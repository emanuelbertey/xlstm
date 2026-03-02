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
            // Parallel execution
            let h = parallel_stabilized_simple(q, k, v, i, f, self.eps);
            (h, None)
        } else {
            // Recurrent execution (step by step)
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

pub fn parallel_stabilized_simple<B: Backend>(
    queries: Tensor<B, 4>,
    keys: Tensor<B, 4>,
    values: Tensor<B, 4>,
    igate_preact: Tensor<B, 4>,
    fgate_preact: Tensor<B, 4>,
    eps_val: f64,
) -> Tensor<B, 4> {
    let [_b, _nh, s, dh] = queries.dims();
    let scale = 1.0 / (dh as f64).sqrt();

    // 1. Gates
    let log_fg = burn::tensor::activation::log_sigmoid(fgate_preact);
    let log_fg_cumsum = log_fg.cumsum(2);
    
    // 2. Forget gate matrix
    let log_fg_matrix = log_fg_cumsum.clone() - log_fg_cumsum.swap_dims(2, 3);
    
    // 3. Combination gate matrix D (forget + input)
    let log_d_matrix = log_fg_matrix + igate_preact.swap_dims(2, 3);

    // 4. Causal Masking
    let device = queries.device();
    let mask = Tensor::<B, 2>::ones([s, s], &device)
        .tril(0)
        .reshape([1, 1, s, s]);
    
    // m_i = max_{j <= i} (log_D_{ij})
    let masked_log_d = log_d_matrix.mask_fill(mask.clone().equal_elem(0.0), -1e10);
    let m = masked_log_d.clone().max_dim(3);

    // 5. Stabilized attention matrix
    let d_matrix = (masked_log_d - m.clone()).exp().mask_fill(mask.equal_elem(0.0), 0.0);

    // 6. Matmul & Normalize
    let qk_matrix = queries.clone().matmul(keys.swap_dims(2, 3)) * scale;
    let c_matrix = qk_matrix * d_matrix;
    
    // Normalization denominator
    let normalizer = c_matrix.clone().sum_dim(3).abs().max_pair(m.neg().exp()) + eps_val;

    let c_matrix_normalized = c_matrix / normalizer;
    c_matrix_normalized.matmul(values)
}

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

    let log_fg = burn::tensor::activation::log_sigmoid(fgate_preact); 
    
    let m_new = (m_state.clone() + log_fg.clone()).max_pair(igate_preact.clone()); 

    let f_act = (m_state + log_fg - m_new.clone()).exp();
    let i_act = (igate_preact - m_new.clone()).exp(); 

    let k_scaled = k * scale;
    let k_t_scaled = k_scaled.swap_dims(2, 3); 

    let c_new = c_state * f_act.clone() + k_t_scaled.clone().matmul(v) * i_act.clone();
    let n_new = n_state * f_act + k_t_scaled * i_act;

    let h_num = q.clone().matmul(c_new.clone()); 
    let qn_dot = q.matmul(n_new.clone());
    
    let h_denom = qn_dot.abs().max_pair(m_new.clone().neg().exp()) + eps_val;

    (h_num / h_denom, (c_new, n_new, m_new))
}
