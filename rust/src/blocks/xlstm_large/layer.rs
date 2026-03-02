use burn::prelude::*;
use burn::nn;
use burn::module::Module;

use crate::components::ln::{MultiHeadLayerNorm, MultiHeadLayerNormConfig};
use super::config::XLSTMLargeConfig;
use super::components::soft_cap;
use super::backends::{parallel_stabilized_simple, recurrent_step_stabilized_simple};

#[derive(Module, Debug)]
pub struct mLSTMLayer<B: Backend> {
    pub q_proj: nn::Linear<B>,
    pub k_proj: nn::Linear<B>,
    pub v_proj: nn::Linear<B>,
    pub ogate: nn::Linear<B>,
    pub igate: nn::Linear<B>,
    pub fgate: nn::Linear<B>,
    pub outnorm: MultiHeadLayerNorm<B>,
    pub out_proj: nn::Linear<B>,
    pub num_heads: usize,
    pub qk_dim: usize,
    pub v_dim: usize,
    pub gate_soft_cap: f64,
}

impl<B: Backend> mLSTMLayer<B> {
    pub fn init(config: &XLSTMLargeConfig, device: &B::Device) -> Self {
        let qk_dim = config.qk_dim();
        let v_dim = config.v_dim();
        
        let q_proj = nn::LinearConfig::new(config.embedding_dim, qk_dim).with_bias(config.use_bias).init(device);
        let k_proj = nn::LinearConfig::new(config.embedding_dim, qk_dim).with_bias(config.use_bias).init(device);
        let v_proj = nn::LinearConfig::new(config.embedding_dim, v_dim).with_bias(config.use_bias).init(device);
        
        let ogate = nn::LinearConfig::new(config.embedding_dim, v_dim).with_bias(config.use_bias).init(device);
        let igate = nn::LinearConfig::new(config.embedding_dim, config.num_heads).with_bias(true).init(device);
        let fgate = nn::LinearConfig::new(config.embedding_dim, config.num_heads).with_bias(true).init(device);
        
        let outnorm = MultiHeadLayerNormConfig {
            ndim: v_dim,
            eps: config.norm_eps,
            ..Default::default()
        }.init(config.num_heads, device);
        
        let out_proj = nn::LinearConfig::new(v_dim, config.embedding_dim).with_bias(config.use_bias).init(device);
        
        Self {
            q_proj,
            k_proj,
            v_proj,
            ogate,
            igate,
            fgate,
            outnorm,
            out_proj,
            num_heads: config.num_heads,
            qk_dim,
            v_dim,
            gate_soft_cap: config.gate_soft_cap,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, state: Option<(Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>)>) -> (Tensor<B, 3>, Option<(Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>)>) {
        let [b, s, d] = x.dims();
        let device = x.device();
        
        if s > 1 && state.is_none() {
            // Parallel Mode
            let q = self.q_proj.forward(x.clone());
            let k = self.k_proj.forward(x.clone());
            let v = self.v_proj.forward(x.clone());
            let o_preact = self.ogate.forward(x.clone());
            
            let i_preact = soft_cap(self.igate.forward(x.clone()), self.gate_soft_cap);
            let f_preact = soft_cap(self.fgate.forward(x), self.gate_soft_cap);
            
            let q = q.reshape([b, s, self.num_heads, self.qk_dim / self.num_heads]).swap_dims(1, 2);
            let k = k.reshape([b, s, self.num_heads, self.qk_dim / self.num_heads]).swap_dims(1, 2);
            let v = v.reshape([b, s, self.num_heads, self.v_dim / self.num_heads]).swap_dims(1, 2);
            let i_preact = i_preact.swap_dims(1, 2).unsqueeze_dim(3);
            let f_preact = f_preact.swap_dims(1, 2).unsqueeze_dim(3);

            let h = parallel_stabilized_simple(q, k, v, i_preact, f_preact);
            
            let h_norm = self.outnorm.forward(h); // (B, NH, S, DH)
            let h_norm = h_norm.swap_dims(1, 2).reshape([b, s, self.v_dim]);
            
            let og_act = burn::tensor::activation::sigmoid(o_preact);
            let h_out = h_norm * og_act;
            
            (self.out_proj.forward(h_out), None)
        } else {
            // Recurrent Mode (S tokens, potentially pre-filling or stepping)
            let mut current_state = state.unwrap_or_else(|| {
                let dh_v = self.v_dim / self.num_heads;
                let dh_qk = self.qk_dim / self.num_heads;
                (
                    Tensor::zeros([b, self.num_heads, dh_qk, dh_v], &device),
                    Tensor::zeros([b, self.num_heads, dh_qk, 1], &device),
                    Tensor::zeros([b, self.num_heads, 1, 1], &device),
                )
            });

            let mut outs = Vec::with_capacity(s);
            
            // For each token in sequence
            for t in 0..s {
                let xt = x.clone().narrow(1, t, 1); // [B, 1, D]
                
                let q = self.q_proj.forward(xt.clone());
                let k = self.k_proj.forward(xt.clone());
                let v = self.v_proj.forward(xt.clone());
                let o_preact = self.ogate.forward(xt.clone()).reshape([b, self.v_dim]);
                
                let i_preact = soft_cap(self.igate.forward(xt.clone()).reshape([b, self.num_heads]), self.gate_soft_cap);
                let f_preact = soft_cap(self.fgate.forward(xt).reshape([b, self.num_heads]), self.gate_soft_cap);

                let q = q.reshape([b, self.num_heads, self.qk_dim / self.num_heads, 1]);
                let k = k.reshape([b, self.num_heads, self.qk_dim / self.num_heads, 1]);
                let v = v.reshape([b, self.num_heads, self.v_dim / self.num_heads, 1]);
                let i_preact = i_preact.reshape([b, self.num_heads, 1, 1]);
                let f_preact = f_preact.reshape([b, self.num_heads, 1, 1]);

                let (c, n, m) = current_state;
                let (h, next_state) = recurrent_step_stabilized_simple(c, n, m, q.swap_dims(2,3), k.swap_dims(2,3), v.swap_dims(2,3), i_preact, f_preact);
                current_state = next_state;

                let h_norm = self.outnorm.forward(h).reshape([b, 1, self.v_dim]);
                let og_act = burn::tensor::activation::sigmoid(o_preact).reshape([b, 1, self.v_dim]);
                let h_out = h_norm * og_act;
                
                let yt = self.out_proj.forward(h_out); // [B, 1, D]
                outs.push(yt);
            }

            let y = Tensor::cat(outs, 1);
            (y, Some(current_state))
        }
    }
}

