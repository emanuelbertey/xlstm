use burn::prelude::*;

pub fn parallel_stabilized_simple<B: Backend>(
    queries: Tensor<B, 4>,
    keys: Tensor<B, 4>,
    values: Tensor<B, 4>,
    igate_preact: Tensor<B, 4>,
    fgate_preact: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let [_b, _nh, s, dh] = queries.dims();
    let eps: f64 = 1e-6;
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
        .unsqueeze::<3>()
        .unsqueeze::<4>();
    
    // m_i = max_{j <= i} (log_D_{ij})
    let masked_log_d = log_d_matrix.mask_fill(mask.clone().equal_elem(0.0), f32::NEG_INFINITY);
    let m = masked_log_d.clone().max_dim(3);

    // 5. Stabilized attention matrix
    let d_matrix = (masked_log_d - m.clone()).exp().mask_fill(mask.equal_elem(0.0), 0.0);

    // 6. Matmul & Normalize
    let keys_scaled = keys * scale;
    let qk_matrix = queries.matmul(keys_scaled.swap_dims(2, 3));
    let c_matrix = qk_matrix * d_matrix;
    
    let qn_dot = c_matrix.clone().sum_dim(3).abs();
    let h_den = qn_dot.max_pair(m.neg().exp()) + eps;

    let c_matrix_normalized = c_matrix / h_den;
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
) -> (Tensor<B, 4>, (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>)) {
    let dh = q.dims()[3];
    let eps: f64 = 1e-6;
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
    
    let h_denom = qn_dot.abs().max_pair(m_new.clone().neg().exp()) + eps;

    (h_num / h_denom, (c_new, n_new, m_new))
}
