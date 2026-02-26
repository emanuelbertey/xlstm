// CausalConv1d — Depthwise causal convolution
// Matches Python: components/conv.py

use burn::prelude::*;
use burn::module::Module;
use burn::config::Config;
use burn::nn;

#[derive(Config)]
pub struct CausalConv1dConfig {
    pub feature_dim: usize,
    #[config(default = 4)]
    pub kernel_size: usize,
    #[config(default = true)]
    pub bias: bool,
}

#[derive(Module, Debug)]
pub struct CausalConv1d<B: Backend> {
    pub conv: nn::conv::Conv1d<B>,
    pub kernel_size: usize,
    pub feature_dim: usize,
    pub pad: usize,
}

impl CausalConv1dConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalConv1d<B> {
        let pad = self.kernel_size - 1;
        // Depthwise: groups = feature_dim
        // IMPORTANT: We use padding 0 here and pad manually in forward to ensure causality (left padding only)
        let conv = nn::conv::Conv1dConfig::new(self.feature_dim, self.feature_dim, self.kernel_size)
            .with_padding(nn::PaddingConfig1d::Valid) // No padding inside Burn conv
            .with_groups(self.feature_dim)
            .with_bias(self.bias)
            .init(device);
        CausalConv1d {
            conv,
            kernel_size: self.kernel_size,
            feature_dim: self.feature_dim,
            pad,
        }
    }
}

impl<B: Backend> CausalConv1d<B> {
    pub fn empty_state(&self, batch_size: usize, device: &B::Device) -> Tensor<B, 3> {
        Tensor::zeros([batch_size, self.kernel_size, self.feature_dim], device)
    }

    pub fn step(&self, x: Tensor<B, 2>, state: Tensor<B, 3>) -> (Tensor<B, 2>, Tensor<B, 3>) {
        let [_b, k, f] = state.dims();

        // Roll and update state (copying Python roll behavior)
        let state_remaining = state.narrow(1, 1, k - 1); // (B, K-1, F)
        let x_expanded = x.unsqueeze_dim(1); // (B, 1, F)
        let new_state = Tensor::cat(vec![state_remaining, x_expanded], 1); // (B, K, F)

        // Conv manual: SUM( conv_state * weight )
        let weight = self.conv.weight.val() // (F, 1, K)
            .reshape([f, k])
            .swap_dims(0, 1) // (K, F)
            .unsqueeze_dim(0); // (1, K, F)

        let mut y = (new_state.clone() * weight).sum_dim(1).reshape([_b, f]);

        if let Some(bias) = &self.conv.bias {
            y = y + bias.val().reshape([1, f]);
        }
        (y, new_state)
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, _s, f] = x.dims();
        let device = x.device();
        
        // Manual left padding for causality: (B, S+pad, F)
        let left_pad = Tensor::zeros([b, self.pad, f], &device);
        let x_padded = Tensor::cat(vec![left_pad, x], 1); // (B, S+pad, F)
        
        // (B, S+pad, F) -> (B, F, S+pad) for Conv1d
        let x_conv = x_padded.swap_dims(1, 2);
        let y = self.conv.forward(x_conv); // (B, F, S) because padding is Valid and kernel=pad+1
        
        // (B, F, S) -> (B, S, F)
        y.swap_dims(1, 2)
    }
}
