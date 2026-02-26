// mLSTM Backend functions
// Matches Python: blocks/mlstm/backends.py
//
// parallel_stabilized_simple:  full parallel mLSTM forward (for training)
// recurrent_step_stabilized_simple: single-step recurrent mLSTM (for inference)

use burn::prelude::*;

/// Parallel stabilized mLSTM forward pass.
///
/// # Arguments
/// * `queries`  — (B, NH, S, DH)
/// * `keys`     — (B, NH, S, DH)
/// * `values`   — (B, NH, S, DH)
/// * `igate_preact` — (B, NH, S, 1) input gate pre-activations
/// * `fgate_preact` — (B, NH, S, 1) forget gate pre-activations
///
/// # Returns
/// * h_tilde_state — (B, NH, S, DH)
pub fn parallel_stabilized_simple<B: Backend>(
    queries: Tensor<B, 4>,
    keys: Tensor<B, 4>,
    values: Tensor<B, 4>,
    igate_preact: Tensor<B, 4>,
    fgate_preact: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let [b, nh, s, dh] = queries.dims();
    let device = queries.device();
    let eps: f64 = 1e-6;

    // ── forget gate cumulative log-sigmoid ──────────────────────────────
    let log_fgates = log_sigmoid(fgate_preact); // (B, NH, S, 1)

    // Prepend a zero row: (B, NH, 1, 1) then cumulative sum along dim 2
    let zeros = Tensor::<B, 4>::zeros([b, nh, 1, 1], &device);
    let log_fgates_cat = Tensor::cat(vec![zeros, log_fgates.clone()], 2); // (B, NH, S+1, 1)
    let log_fgates_cumsum = cumsum_dim2(log_fgates_cat, s + 1); // (B, NH, S+1, 1)

    // Build the (S, S) forget-gate decay matrix
    // rep_log_fgates_cumsum: repeat along last dim → (B, NH, S+1, S+1)
    let rep = log_fgates_cumsum.clone().repeat_dim(3, s + 1); // (B, NH, S+1, S+1)
    let _log_fg_matrix = rep.clone() - rep.swap_dims(2, 3); // (B, NH, S+1, S+1)

    // Slice [1:, 1:] and apply causal mask
    let log_fg_sub = _log_fg_matrix.narrow(2, 1, s).narrow(3, 1, s); // (B, NH, S, S)

    // Causal mask: lower triangular = true (keep), upper = false (mask to -inf)
    let causal_mask = lower_triangular_bool::<B>(s, &device); // (S, S) Bool
    // Expand to (B, NH, S, S) — Burn's mask_where needs exact shape match
    let causal_mask_4d = causal_mask
        .unsqueeze_dim::<3>(0)
        .unsqueeze_dim::<4>(0)  // (1, 1, S, S)
        .repeat_dim(0, b)
        .repeat_dim(1, nh); // (B, NH, S, S) Bool

    // Invert mask: upper triangle = true (where to replace with -inf)
    let upper_mask = causal_mask_4d.bool_not(); // (B, NH, S, S) Bool

    let neg_inf = Tensor::<B, 4>::full([b, nh, s, s], f32::NEG_INFINITY, &device);
    let log_fg_matrix = log_fg_sub.mask_where(upper_mask, neg_inf);

    // ── gate decay matrix D ─────────────────────────────────────────────
    // log_D = log_fg + igate_preact^T
    let igate_t = igate_preact.swap_dims(2, 3); // (B, NH, 1, S)
    let log_d_matrix = log_fg_matrix.clone() + igate_t; // (B, NH, S, S)

    // Stabilize row-wise
    let max_log_d = log_d_matrix.clone().max_dim(3); // (B, NH, S, 1)
    let log_d_stab = log_d_matrix - max_log_d.clone();
    let d_matrix = log_d_stab.exp(); // (B, NH, S, S)

    // ── combination matrix C ────────────────────────────────────────────
    let scale = 1.0 / (dh as f64).sqrt();
    let keys_scaled = keys * scale; // (B, NH, S, DH)
    let qk_matrix = queries.matmul(keys_scaled.swap_dims(2, 3)); // (B, NH, S, S)
    let c_matrix = qk_matrix * d_matrix; // (B, NH, S, S)

    // Normalizer: max(|sum(C, dim=-1)|, exp(-max_log_D))
    let c_sum = c_matrix.clone().sum_dim(3); // (B, NH, S, 1)
    let c_sum_abs = c_sum.abs();
    let exp_neg_max = (max_log_d.neg()).exp();
    // element-wise maximum
    let normalizer = max_tensor(c_sum_abs, exp_neg_max); // (B, NH, S, 1)

    let c_normalized = c_matrix / (normalizer + eps);

    // ── output ──────────────────────────────────────────────────────────
    c_normalized.matmul(values) // (B, NH, S, DH)
}

/// Single recurrent step of the stabilized mLSTM.
///
/// # Arguments
/// * `c_state` — (B, NH, DH, DH) cell state matrix
/// * `n_state` — (B, NH, DH, 1) normalizer state
/// * `m_state` — (B, NH, 1, 1) max state for stabilization
/// * `q` — (B, NH, 1, DH)
/// * `k` — (B, NH, 1, DH)
/// * `v` — (B, NH, 1, DH)
/// * `igate_preact` — (B, NH, 1, 1)
/// * `fgate_preact` — (B, NH, 1, 1)
///
/// # Returns
/// (h, (c_new, n_new, m_new))
pub fn recurrent_step_stabilized_simple<B: Backend>(
    c_state: Tensor<B, 4>,
    n_state: Tensor<B, 4>,
    m_state: Tensor<B, 4>,
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    v: Tensor<B, 4>,
    igate_preact: Tensor<B, 4>,
    fgate_preact: Tensor<B, 4>,
) -> (Tensor<B, 4>, (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>)) {
    let [_b, _nh, _s, dh] = q.dims();
    let eps: f64 = 1e-6;

    // Squeeze S=1 dim for q,k,v: (B, NH, 1, DH) → (B, NH, DH, 1) via transpose
    let q_col = q.swap_dims(2, 3); // (B, NH, DH, 1)
    let k_col = k.swap_dims(2, 3); // (B, NH, DH, 1)
    let v_col = v.swap_dims(2, 3); // (B, NH, DH, 1)

    // Gates
    let log_fg = log_sigmoid(fgate_preact.clone()); // (B, NH, 1, 1)

    // New max state: m_new = max(log_fg + m, igate)
    let log_fg_plus_m = log_fg.clone() + m_state.clone(); // (B, NH, 1, 1)
    let m_new = max_tensor(log_fg_plus_m.clone(), igate_preact.clone()); // (B, NH, 1, 1)

    let fg_act = (log_fg + m_state - m_new.clone()).exp(); // (B, NH, 1, 1)
    let ig_act = (igate_preact - m_new.clone()).exp(); // (B, NH, 1, 1)

    let scale = 1.0 / (dh as f64).sqrt();
    let k_scaled = k_col.clone() * scale; // (B, NH, DH, 1)

    // c_new = fg * c + ig * (k_scaled @ v^T)
    let kv_outer = k_scaled.clone().matmul(v_col.swap_dims(2, 3)); // (B, NH, DH, DH)
    let c_new = fg_act.clone() * c_state + ig_act.clone() * kv_outer;

    // n_new = fg * n + ig * k_scaled
    let n_new = fg_act * n_state + ig_act * k_scaled;

    // h = (q^T @ c_new) / max( |q^T @ n_new|, exp(-m_new) ) + eps
    let h_num = q_col.clone().swap_dims(2, 3).matmul(c_new.clone()); // (B, NH, 1, DH)
    let qn = q_col.swap_dims(2, 3).matmul(n_new.clone()); // (B, NH, 1, 1)
    let exp_neg_m = (m_new.clone().neg()).exp();
    let h_denom = max_tensor(qn.abs(), exp_neg_m) + eps;
    let h = h_num / h_denom; // (B, NH, 1, DH)

    (h, (c_new, n_new, m_new))
}

// ─── Utility ────────────────────────────────────────────────────────────────

/// log(sigmoid(x)) = -softplus(-x) for numerical stability
fn log_sigmoid<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    let neg_x = x.neg();
    let softplus_neg = (neg_x.exp() + 1.0).log(); // softplus(-x)
    softplus_neg.neg() // -softplus(-x) = log(sigmoid(x))
}

/// Element-wise maximum of two tensors
fn max_tensor<B: Backend, const D: usize>(a: Tensor<B, D>, b: Tensor<B, D>) -> Tensor<B, D> {
    let mask = a.clone().lower_equal(b.clone()); // true where b >= a, so replace a with b
    a.mask_where(mask, b)
}

/// Cumulative sum along dimension 2 for a 4D tensor.
/// Burn doesn't have cumsum so we do it manually.
/*
fn cumsum_dim2<B: Backend>(x: Tensor<B, 4>, len: usize) -> Tensor<B, 4> {
    let [b, nh, _s, d] = x.dims();
    let device = x.device();
    let mut result = Tensor::<B, 4>::zeros([b, nh, len, d], &device);
    // First slice
    let first = x.clone().narrow(2, 0, 1);
    result = result.slice_assign([0..b, 0..nh, 0..1, 0..d], first.clone());
    let mut running = first;
    for t in 1..len {
        let cur = x.clone().narrow(2, t, 1);
        running = running + cur;
        result = result.slice_assign([0..b, 0..nh, t..t+1, 0..d], running.clone());
    }
    result
}
*/
fn cumsum_dim2<B: Backend>(x: Tensor<B, 4>, len: usize) -> Tensor<B, 4> {
    let [_b, _nh, _s, _d] = x.dims();
    let device = x.device();
    
    // Create lower triangular matrix of 1s: (len, len)
    let mask = lower_triangular_bool::<B>(len, &device);
    let l_matrix = Tensor::<B, 2>::zeros([len, len], &device).mask_where(mask, Tensor::ones([len, len], &device));
    
    // matmul broadcast: (1, 1, L, L) @ (B, NH, L, D) -> (B, NH, L, D)
    // We must unsqueeze incrementally: 2D -> 3D -> 4D
    l_matrix.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0).matmul(x)
}

/// Lower triangular boolean mask of size (S, S).
fn lower_triangular_bool<B: Backend>(s: usize, device: &B::Device) -> Tensor<B, 2, Bool> {
    let mut data = vec![false; s * s];
    for i in 0..s {
        for j in 0..=i {
            data[i * s + j] = true;
        }
    }
    Tensor::<B, 1, Bool>::from_bool(
        burn::tensor::TensorData::from(data.as_slice()).convert::<bool>(),
        device,
    )
    .reshape([s, s])
}
