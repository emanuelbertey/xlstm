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
        let conv = nn::conv::Conv1dConfig::new(self.feature_dim, self.feature_dim, self.kernel_size)
            .with_padding(nn::PaddingConfig1d::Explicit(pad))
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
        let state_remaining = state.narrow(1, 1, k - 1);
        let x_expanded = x.clone().unsqueeze_dim(1);
        let new_state = Tensor::cat(vec![state_remaining, x_expanded], 1);
        let state_flipped = new_state.clone().swap_dims(1, 2); // (B, F, K)
        let weight = self.conv.weight.val().reshape([1, f, k]);
        let mut y = (state_flipped * weight).sum_dim(2).reshape([_b, f]);
        if let Some(bias) = &self.conv.bias {
            y = y + bias.val().unsqueeze_dim(0);
        }
        (y, new_state)
    }

    /// Forward: input (B, S, F) → output (B, S, F).
    /// Applies causal (left-padded) depthwise 1D convolution.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_b, s, _f] = x.dims();
        // (B, S, F) → (B, F, S) for Conv1d
        let x = x.swap_dims(1, 2);
        let y = self.conv.forward(x); // (B, F, S + pad)
        // Trim right padding to maintain causality: take first S elements
        let y = y.narrow(2, 0, s);
        // (B, F, S) → (B, S, F)
        y.swap_dims(1, 2)
    }
}
