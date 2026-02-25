// LayerNorm and MultiHeadLayerNorm
// Matches Python: components/ln.py

use burn::prelude::*;
use burn::module::Module;
use burn::config::Config;
use burn::nn;

// ─── LayerNorm ────────────────────────────────────────────────────────────────
// This wraps Burn's built-in LayerNorm. The Python version uses a "residual weight"
// trick where weight = 1 + param (param initialized to 0), but Burn's LayerNorm
// already stores weight initialized to 1, so the effect is equivalent.

#[derive(Config)]
pub struct LayerNormConfig {
    pub ndim: usize,
    #[config(default = true)]
    pub weight: bool,
    #[config(default = false)]
    pub bias: bool,
    #[config(default = 1e-5)]
    pub eps: f64,
}

#[derive(Module, Debug)]
pub struct LayerNorm<B: Backend> {
    pub norm: nn::LayerNorm<B>,
}

impl LayerNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LayerNorm<B> {
        let norm = nn::LayerNormConfig::new(self.ndim)
            .with_epsilon(self.eps)
            .init(device);
        LayerNorm { norm }
    }
}

impl<B: Backend> LayerNorm<B> {
    /// Forward: applies LayerNorm over the last dimension.
    /// Input can be any rank, but the last dim must be `ndim`.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.norm.forward(x)
    }
    
    pub fn forward_2d(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.norm.forward(x)
    }
}

// ─── MultiHeadLayerNorm ───────────────────────────────────────────────────────
// Input: (B, NH, S, DH) — applies group-norm style per-head normalization.
// Implementation: reshape to (B*S, NH*DH) → GroupNorm(num_groups=NH) → reshape back.

#[derive(Config)]
pub struct MultiHeadLayerNormConfig {
    pub ndim: usize,  // total hidden dim = NH * DH
    #[config(default = true)]
    pub weight: bool,
    #[config(default = false)]
    pub bias: bool,
    #[config(default = 1e-5)]
    pub eps: f64,
}

#[derive(Module, Debug)]
pub struct MultiHeadLayerNorm<B: Backend> {
    pub norm: nn::GroupNorm<B>,
    pub ndim: usize,
    pub num_groups: usize,
}

impl MultiHeadLayerNormConfig {
    /// `num_heads` determines the number of groups for group normalization.
    pub fn init<B: Backend>(&self, num_heads: usize, device: &B::Device) -> MultiHeadLayerNorm<B> {
        // GroupNorm with num_groups=num_heads, num_channels=ndim
        let norm = nn::GroupNormConfig::new(num_heads, self.ndim)
            .with_epsilon(self.eps)
            .init(device);
        MultiHeadLayerNorm {
            norm,
            ndim: self.ndim,
            num_groups: num_heads,
        }
    }
}

impl<B: Backend> MultiHeadLayerNorm<B> {
    /// Forward pass on tensor of shape (B, NH, S, DH).
    /// Applies group normalization per head.
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, nh, s, dh] = x.dims();
        // (B, NH, S, DH) → (B, S, NH, DH)
        let x = x.swap_dims(1, 2);
        // (B, S, NH, DH) → (B*S, NH*DH)
        let x = x.reshape([b * s, nh * dh]);
        // GroupNorm expects (B, C) where C = num_channels, groups = NH
        let x = self.norm.forward(x);
        // (B*S, NH*DH) → (B, S, NH, DH) → (B, NH, S, DH)
        x.reshape([b, s, nh, dh]).swap_dims(1, 2)
    }
}
