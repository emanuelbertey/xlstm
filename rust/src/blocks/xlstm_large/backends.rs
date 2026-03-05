use burn::prelude::*;
use burn::module::Module;

#[derive(Module, Debug, Clone)]
pub struct MLSTMBackend {
    pub chunk_size: usize,
    pub eps: f64,
}

impl MLSTMBackend {
    pub fn new(chunk_size: usize, eps: f64) -> Self {
        Self { chunk_size, eps }
    }

    pub fn forward<B: Backend>(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        i: Tensor<B, 4>,
        f: Tensor<B, 4>,
        state: Option<(Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>)>,
    ) -> (Tensor<B, 4>, Option<(Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>)>) {
        let [_b, _nh, s, _dh] = q.dims();
        let device = q.device();

        if s > 1 && state.is_none() {
            // Parallel execution (training mode, no initial state)
            let h = parallel_stabilized_simple(q, k, v, i, f, self.eps);
            (h, None)
        } else {
            // Recurrent execution — used for inference (step-by-step with carried state)
            let mut current_state = state.unwrap_or_else(|| {
                let [b, nh, _, dh_qk] = q.dims();
                let dh_v = v.dims()[3];
                (
                    Tensor::zeros([b, nh, dh_qk, dh_v], &device),
                    Tensor::zeros([b, nh, dh_qk, 1], &device),
                    Tensor::zeros([b, nh, 1, 1], &device),
                )
            });

            let mut outs = Vec::with_capacity(s);
            for t in 0..s {
                let qt = q.clone().narrow(2, t, 1);
                let kt = k.clone().narrow(2, t, 1);
                let vt = v.clone().narrow(2, t, 1);
                let it = i.clone().narrow(2, t, 1);
                let ft = f.clone().narrow(2, t, 1);

                let (h, next_state) = recurrent_step_stabilized_simple(
                    current_state.0,
                    current_state.1,
                    current_state.2,
                    qt,
                    kt,
                    vt,
                    it,
                    ft,
                    self.eps,
                );
                current_state = next_state;
                outs.push(h);
            }
            let y = Tensor::cat(outs, 2);
            (y, Some(current_state))
        }
    }
}

/// Parallel (quadratic) stabilized mLSTM forward pass.
///
/// Matches `parallel_stabilized_simple` from `xlstm/blocks/mlstm/backends.py` exactly:
///
/// 1. Forget-gate cumsum with a prepended zero column (shape S+1), subtracted and
///    then sliced [1:, 1:] — this is identical to the Python reference.
/// 2. Causal mask uses true `-inf` (f32::NEG_INFINITY) instead of a large finite
///    negative number, guaranteeing exp(-inf) == 0.0 in all dtypes including bfloat16.
///
/// Args (all `(B, NH, S, DH)` or `(B, NH, S, 1)` for gates):
///   queries, keys, values — (B, NH, S, DH)
///   igate_preact, fgate_preact — (B, NH, S, 1)  pre-activations
///   eps_val — small epsilon for numerical stability
///
/// Returns: h_tilde — (B, NH, S, DH)
pub fn parallel_stabilized_simple<B: Backend>(
    queries: Tensor<B, 4>,
    keys: Tensor<B, 4>,
    values: Tensor<B, 4>,
    igate_preact: Tensor<B, 4>,
    fgate_preact: Tensor<B, 4>,
    eps_val: f64,
) -> Tensor<B, 4> {
    let [b, nh, s, dh] = queries.dims();
    let device = queries.device();
    let scale = 1.0 / (dh as f64).sqrt();

    // ── Step 1: log-sigmoid of forget gates ── (B, NH, S, 1)
    let log_fg = burn::tensor::activation::log_sigmoid(fgate_preact);

    // ── Step 2: Cumsum with prepended zero — matches Python ──
    //   Python:
    //     log_fgates_cumsum = cat([zeros(B,NH,1,1), cumsum(log_fg, dim=-2)], dim=-2)
    //     # shape: (B, NH, S+1, 1)
    //   Then builds (S+1 x S+1) matrix by repeat + subtract, and slices [1:, 1:]
    //   so that the diagonal encodes f_t (not 1) and entry [i,j] = sum_{l=j+1}^{i} log_f_l
    let zero_prefix = Tensor::<B, 4>::zeros([b, nh, 1, 1], &device); // (B, NH, 1, 1)
    let log_fg_cumsum = Tensor::cat(vec![zero_prefix, log_fg.cumsum(2)], 2); // (B, NH, S+1, 1)

    // Broadcast to (B, NH, S+1, S+1): row_i - col_j  =>  cumsum[i] - cumsum[j]
    // swap_dims(2,3) turns (S+1,1) into (1,S+1) — broadcasting handles the rest
    let log_fg_matrix_full = log_fg_cumsum.clone() - log_fg_cumsum.swap_dims(2, 3);
    // (B, NH, S+1, S+1)

    // Slice [1:, 1:] to get (B, NH, S, S)  — same as Python [:, :, 1:, 1:]
    let log_fg_matrix = log_fg_matrix_full.narrow(2, 1, s).narrow(3, 1, s);

    // ── Step 3: Combination gate matrix log D = log_fg_matrix + igate_preact^T ──
    // igate_preact: (B, NH, S, 1) — transpose to (B, NH, 1, S) for broadcasting
    let log_d_matrix = log_fg_matrix + igate_preact.swap_dims(2, 3); // (B, NH, S, S)

    // ── Step 4: Causal mask with true -inf ──
    // Using f32::NEG_INFINITY ensures exp(-inf) == 0 exactly, even in bfloat16.
    let causal_mask = Tensor::<B, 2>::ones([s, s], &device)
        .tril(0)
        .reshape([1, 1, s, s]); // (1, 1, S, S)

    let masked_log_d = log_d_matrix
        .mask_fill(causal_mask.clone().equal_elem(0.0), f32::NEG_INFINITY as f64);

    // ── Step 5: Row-wise max for numerical stabilization ── m_i = max_{j<=i}(log_D_ij)
    let m = masked_log_d.clone().max_dim(3); // (B, NH, S, 1)

    // ── Step 6: Stabilized D matrix ── exp(log_D - m), then zero out upper triangle
    let d_matrix = (masked_log_d - m.clone())
        .exp()
        .mask_fill(causal_mask.equal_elem(0.0), 0.0); // (B, NH, S, S)

    // ── Step 7: Scaled QK^T weighted by D ──
    let qk_matrix = queries.clone().matmul(keys.swap_dims(2, 3)) * scale; // (B, NH, S, S)
    let c_matrix = qk_matrix * d_matrix;                                   // (B, NH, S, S)

    // ── Step 8: Normalizer — max(|sum_j C_ij|, exp(-m_i)) + eps ──
    let normalizer = c_matrix.clone().sum_dim(3).abs()
        .max_pair(m.neg().exp()) + eps_val; // (B, NH, S, 1)

    // ── Step 9: Normalize and retrieve values ──
    let c_matrix_normalized = c_matrix / normalizer;
    c_matrix_normalized.matmul(values) // (B, NH, S, DH)
}

/// Single recurrent step of the stabilized mLSTM.
///
/// Matches `recurrent_step_stabilized_simple` from `xlstm/blocks/mlstm/backends.py` exactly.
///
/// State tensors:
///   c_state — (B, NH, DH_qk, DH_v)  memory matrix
///   n_state — (B, NH, DH_qk, 1)     normalizer vector
///   m_state — (B, NH, 1, 1)         log-max stabilizer scalar
///
/// Returns: (h — (B, NH, 1, DH_v), (c_new, n_new, m_new))
pub fn recurrent_step_stabilized_simple<B: Backend>(
    c_state: Tensor<B, 4>,
    n_state: Tensor<B, 4>,
    m_state: Tensor<B, 4>,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    igate_preact: Tensor<B, 4>,
    fgate_preact: Tensor<B, 4>,
    eps_val: f64,
) -> (Tensor<B, 4>, (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>)) {
    let dh = q.dims()[3];
    let scale = 1.0 / (dh as f64).sqrt();

    // log-sigmoid of forget gate: (B, NH, 1, 1)
    let log_fg = burn::tensor::activation::log_sigmoid(fgate_preact);

    // Stabilized max: m_new = max(m + log_fg, igate_preact)
    let m_new = (m_state.clone() + log_fg.clone()).max_pair(igate_preact.clone());

    // Gate activations (stabilized)
    let f_act = (m_state + log_fg - m_new.clone()).exp();   // (B, NH, 1, 1)
    let i_act = (igate_preact - m_new.clone()).exp();        // (B, NH, 1, 1)

    // Scaled key: k / sqrt(DH),  transposed for outer product
    let k_scaled = k * scale;                                // (B, NH, 1, DH)
    let k_t_scaled = k_scaled.swap_dims(2, 3);              // (B, NH, DH, 1)

    // State updates
    // c_new = fg * C + ig * (k^T ⊗ v)    shape: (B, NH, DH_qk, DH_v)
    let c_new = c_state * f_act.clone() + k_t_scaled.clone().matmul(v) * i_act.clone();
    // n_new = fg * n + ig * k^T           shape: (B, NH, DH_qk, 1)
    let n_new = n_state * f_act + k_t_scaled * i_act;

    // Output: h = (q @ C) / max(|q @ n|, exp(-m)) + eps
    let h_num = q.clone().matmul(c_new.clone());             // (B, NH, 1, DH_v)
    let qn_dot = q.matmul(n_new.clone());                    // (B, NH, 1, 1)
    let h_denom = qn_dot.abs().max_pair(m_new.clone().neg().exp()) + eps_val;

    (h_num / h_denom, (c_new, n_new, m_new))
}
