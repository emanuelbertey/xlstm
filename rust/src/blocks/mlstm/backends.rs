// mLSTM Backend functions
// Matches Python: blocks/mlstm/backends.py

use burn::prelude::*;

/// Parallel stabilized mLSTM forward pass.
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
    let zeros = Tensor::<B, 4>::zeros([b, nh, 1, 1], &device);
    let log_fgates_cat = Tensor::cat(vec![zeros, log_fgates.clone()], 2);
    let log_fgates_cumsum = cumsum_dim2(log_fgates_cat, s + 1);

    let rep = log_fgates_cumsum.clone().repeat_dim(3, s + 1);
    let _log_fg_matrix = rep.clone() - rep.swap_dims(2, 3);
    let log_fg_sub = _log_fg_matrix.narrow(2, 1, s).narrow(3, 1, s);

    let causal_mask = lower_triangular_bool::<B>(s, &device);
    let causal_mask_4d = causal_mask.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0).repeat_dim(0, b).repeat_dim(1, nh);

    let neg_inf = Tensor::<B, 4>::full([b, nh, s, s], -1e30, &device);
    let log_fg_matrix = log_fg_sub.mask_where(causal_mask_4d.bool_not(), neg_inf);

    let igate_t = igate_preact.swap_dims(2, 3);
    let log_d_matrix = log_fg_matrix + igate_t;

    let max_log_d = log_d_matrix.clone().max_dim(3);
    let d_matrix = (log_d_matrix - max_log_d.clone()).exp();

    let scale = 1.0 / (dh as f64).sqrt();
    let keys_scaled = keys * scale; 
    let qk_matrix = queries.matmul(keys_scaled.swap_dims(2, 3));
    let c_matrix = qk_matrix * d_matrix;

    let c_sum_abs = c_matrix.clone().sum_dim(3).abs();
    let exp_neg_max = (max_log_d.neg()).exp();
    let normalizer = c_sum_abs.max_pair(exp_neg_max); 

    let c_normalized = c_matrix / (normalizer + eps);
    c_normalized.matmul(values) 
}

/// Single recurrent step of the stabilized mLSTM.
/// Python Reference:
///   q, k, v = q.squeeze(2).unsqueeze(-1)  -> (B, NH, DH, 1)
///   m_new = max(log_fg + m, igate)
///   fg_act = exp(log_fg + m - m_new)
///   ig_act = exp(igate - m_new)
///   c_new = fg * c + ig * (k_scaled @ v^T)
///   n_new = fg * n + ig * k_scaled
///   h_num = q^T @ c_new
///   qn = q^T @ n_new
///   h_denom = max(|qn|, exp(-m_new)) + eps
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

    // q, k, v are (B, NH, 1, DH). Squeeze S=1 and make them vectors.
    // In Python: q is (B, NH, DH, 1)
    let k_vec = k.clone().swap_dims(2, 3); // (B, NH, DH, 1)

    let log_fg = log_sigmoid(fgate_preact.clone()); 
    let m_new = (log_fg.clone() + m_state.clone()).max_pair(igate_preact.clone()); 

    let fg_act = (log_fg + m_state - m_new.clone()).exp(); 
    let ig_act = (igate_preact - m_new.clone()).exp(); 

    let scale = 1.0 / (dh as f64).sqrt();
    let k_scaled = k_vec * scale; 

    // c_new = fg * c + ig * (k_scaled @ v_vec^T)
    // v_vec^T is (B, NH, 1, DH) which is just the original 'v'
    let c_new = fg_act.clone() * c_state + ig_act.clone() * k_scaled.clone().matmul(v);

    // n_new = fg * n + ig * k_scaled
    let n_new = fg_act * n_state + ig_act * k_scaled;

    // h_num = q_vec^T @ c_new  -> (B, NH, 1, DH) @ (B, NH, DH, DH)
    let h_num = q.clone().matmul(c_new.clone()); 
    
    // qn = q_vec^T @ n_new -> (B, NH, 1, DH) @ (B, NH, DH, 1) -> (B, NH, 1, 1)
    let qn = q.matmul(n_new.clone()); 
    
    let exp_neg_m = (m_new.clone().neg()).exp();
    let h_denom = qn.abs().max_pair(exp_neg_m) + eps;
    let h = h_num / h_denom; 

    (h, (c_new, n_new, m_new))
}

// ─── Utility ────────────────────────────────────────────────────────────────

fn log_sigmoid<B: Backend, const D: usize>(x: Tensor<B, D>) -> Tensor<B, D> {
    burn::tensor::activation::softplus(x.neg(), 1.0).neg()
}

fn cumsum_dim2<B: Backend>(x: Tensor<B, 4>, len: usize) -> Tensor<B, 4> {
    let device = x.device();
    let mask = lower_triangular_bool::<B>(len, &device);
    let l_matrix = Tensor::<B, 2>::zeros([len, len], &device).mask_where(mask, Tensor::ones([len, len], &device));
    l_matrix.unsqueeze_dim::<3>(0).unsqueeze_dim::<4>(0).matmul(x)
}

fn lower_triangular_bool<B: Backend>(s: usize, device: &B::Device) -> Tensor<B, 2, Bool> {
    let mut data = vec![false; s * s];
    for i in 0..s { for j in 0..=i { data[i * s + j] = true; } }
    Tensor::<B, 1, Bool>::from_bool(burn::tensor::TensorData::from(data.as_slice()).convert::<bool>(), device).reshape([s, s])
}
