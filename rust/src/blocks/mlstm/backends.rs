use burn::prelude::*;

pub fn parallel_stabilized_simple<B: Backend>(
    queries: Tensor<B, 4>,
    keys: Tensor<B, 4>,
    values: Tensor<B, 4>,
    igate_preact: Tensor<B, 4>,
    fgate_preact: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let [_b, _nh, s, dh] = queries.dims();
    let device = queries.device();
    let eps: f64 = 1e-6;
    let scale = 1.0 / (dh as f64).sqrt();

    // 1. Gates y cumsum
    let log_fg = log_sigmoid(fgate_preact);
    let log_fg_cumsum = cumsum_matrix::<B>(log_fg, s, &device);
    
    // 2. Matriz de decaimiento
    let log_fg_matrix = log_fg_cumsum.clone() - log_fg_cumsum.swap_dims(2, 3);
    
    // 3. Estabilización con Input Gate
    let log_d_matrix = log_fg_matrix + igate_preact.swap_dims(2, 3);

    // 4. Máscara Causal (S, S) -> (1, 1, S, S)
    let mask = Tensor::<B, 2>::ones([s, s], &device)
        .tril(0)
        .unsqueeze::<3>()
        .unsqueeze::<4>();
    
    // m_i = max_{j <= i} (log_D_{ij})
    let masked_log_d = log_d_matrix.mask_fill(mask.clone().equal_elem(0.0), -1e10);
    let m = masked_log_d.clone().max_dim(3);

    // 5. Matriz de atención estabilizada
    let d_matrix = (masked_log_d - m.clone()).exp().mask_fill(mask.equal_elem(0.0), 0.0);

    // 6. Cómputo final
    let qk_matrix = queries.matmul(keys.swap_dims(2, 3)) * scale;
    let c_matrix = qk_matrix * d_matrix;
    
    // Clonamos c_matrix porque matmul() consume el original
    let h_num = c_matrix.clone().matmul(values); 
    let qn_dot = c_matrix.sum_dim(3).abs();
    
    let h_den = qn_dot + (m.neg()).exp() + eps;

    h_num / h_den
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

    let log_fg = log_sigmoid(fgate_preact); 
    
    // Evitamos moves clonando los estados que se usan varias veces
    let m_new = (m_state.clone() + log_fg.clone()).max_pair(igate_preact.clone()); 

    let f_act = (m_state + log_fg - m_new.clone()).exp();
    let i_act = (igate_preact - m_new.clone()).exp(); 

    let k_t = k.swap_dims(2, 3); 

    let c_new = c_state * f_act.clone() + k_t.clone().matmul(v) * i_act.clone();
    let n_new = n_state * f_act + k_t * i_act;

    let q_scaled = q * scale;
    let h_num = q_scaled.clone().matmul(c_new.clone()); 
    let qn_dot = q_scaled.matmul(n_new.clone());
    
    let h_denom = qn_dot.abs() + (m_new.clone().neg()).exp() + eps;

    (h_num / h_denom, (c_new, n_new, m_new))
}

fn log_sigmoid<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    burn::tensor::activation::softplus(x.neg(), 1.0).neg()
}

fn cumsum_matrix<B: Backend>(x: Tensor<B, 4>, s: usize, device: &B::Device) -> Tensor<B, 4> {
    let tril = Tensor::<B, 2>::ones([s, s], device)
        .tril(0)
        .unsqueeze::<3>()
        .unsqueeze::<4>();
    tril.matmul(x)
}