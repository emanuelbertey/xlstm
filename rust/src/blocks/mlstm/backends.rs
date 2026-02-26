use burn::prelude::*;

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

    let log_fgates = log_sigmoid(fgate_preact);
    let log_fg_cumsum = cumsum_dim2::<B>(log_fgates, s);
    let zeros = Tensor::zeros([b, nh, 1, 1], &device);
    let log_fg_cumsum_padded = Tensor::cat(vec![zeros, log_fg_cumsum], 2);

    let cumsum_s = log_fg_cumsum_padded.narrow(2, 1, s); 
    let log_fg_matrix = cumsum_s.clone() - cumsum_s.swap_dims(2, 3);

    // Manual mask instead of mask_where
    let row_idx = Tensor::<B, 1, Int>::arange(0..s as i64, &device).reshape([1, 1, s, 1]);
    let col_idx = Tensor::<B, 1, Int>::arange(0..s as i64, &device).reshape([1, 1, 1, s]);
    let ltr = row_idx.greater_equal(col_idx).float();
    let ltr = ltr.repeat_dim(0, b).repeat_dim(1, nh);

    let neg_inf_mask = (ltr.clone() - 1.0) * 1e20;
    let log_fg_matrix = log_fg_matrix * ltr + neg_inf_mask;
    
    let log_d_matrix = log_fg_matrix + igate_preact.swap_dims(2, 3);

    let max_log_d = log_d_matrix.clone().max_dim(3);
    let d_matrix = (log_d_matrix - max_log_d.clone()).exp();

    let qk_matrix = queries.matmul(keys.swap_dims(2, 3)) * (1.0 / (dh as f64).sqrt());
    let c_matrix = qk_matrix * d_matrix;

    // Simplify normalization to be extremely robust for gradients
    let qn_dot = c_matrix.clone().sum_dim(3);
    let normalizer = qn_dot.max_pair((max_log_d.neg()).exp()) + eps;

    c_matrix.matmul(values) / normalizer
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
    let [_b, _nh, _s, dh] = q.dims();
    let eps: f64 = 1e-6;

    let log_fg = log_sigmoid(fgate_preact); 
    let m_new = (log_fg.clone() + m_state.clone()).max_pair(igate_preact.clone()); 

    let fg_act = (m_state + log_fg - m_new.clone()).exp(); 
    let ig_act = (igate_preact - m_new.clone()).exp(); 
    let k_scaled = k.swap_dims(2, 3) * (1.0 / (dh as f64).sqrt()); 

    let c_new = fg_act.clone() * c_state + ig_act.clone() * k_scaled.clone().matmul(v);
    let n_new = fg_act * n_state + ig_act * k_scaled;

    let h_num = q.clone().matmul(c_new.clone()); 
    let h_denom = q.matmul(n_new.clone()).max_pair((m_new.clone().neg()).exp()) + eps;

    (h_num / h_denom, (c_new, n_new, m_new))
}

fn log_sigmoid<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    burn::tensor::activation::softplus(x.neg(), 1.0).neg()
}

fn cumsum_dim2<B: Backend>(x: Tensor<B, 4>, len: usize) -> Tensor<B, 4> {
    let [b, nh, _s, _f] = x.dims();
    let device = x.device();
    let row_idx = Tensor::<B, 1, Int>::arange(0..len as i64, &device).reshape([1, 1, len, 1]);
    let col_idx = Tensor::<B, 1, Int>::arange(0..len as i64, &device).reshape([1, 1, 1, len]);
    let l_matrix = row_idx.greater_equal(col_idx).float();
    let l_matrix = l_matrix.repeat_dim(1, nh).repeat_dim(0, b);
    l_matrix.matmul(x)
}
