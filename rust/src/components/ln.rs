use burn::prelude::*;
use burn::module::Module;
use burn::config::Config;
use burn::nn;

// ─── LayerNorm ────────────────────────────────────────────────────────────────

#[derive(Config)]
pub struct LayerNormConfig {
    pub ndim: usize,
    #[config(default = true)]
    pub weight: bool, // Mantenemos el campo para no romper tus otros archivos
    #[config(default = false)]
    pub bias: bool,   // Mantenemos el campo para no romper tus otros archivos
    #[config(default = 1e-5)]
    pub eps: f64,
}

#[derive(Module, Debug)]
pub struct LayerNorm<B: Backend> {
    pub norm: nn::LayerNorm<B>,
}

impl LayerNormConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LayerNorm<B> {
        // Nota: Burn's LayerNorm siempre incluye pesos (weight=1, bias=0) por defecto.
        let norm = nn::LayerNormConfig::new(self.ndim)
            .with_epsilon(self.eps)
            .init(device);
        LayerNorm { norm }
    }
}

impl<B: Backend> LayerNorm<B> {
    /// Soporta específicamente Tensores 3D (B, S, D)
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.norm.forward(x)
    }

    /// Soporta específicamente Tensores 2D (B, D)
    pub fn forward_2d(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.norm.forward(x)
    }
}

// ─── MultiHeadLayerNorm ───────────────────────────────────────────────────────

#[derive(Config)]
pub struct MultiHeadLayerNormConfig {
    pub ndim: usize,
    #[config(default = true)]
    pub weight: bool, // Mantenemos el campo para no romper tus otros archivos
    #[config(default = false)]
    pub bias: bool,   // Mantenemos el campo para no romper tus otros archivos
    #[config(default = 1e-5)]
    pub eps: f64,
}

#[derive(Module, Debug)]
pub struct MultiHeadLayerNorm<B: Backend> {
    pub norm: nn::GroupNorm<B>,
    pub ndim: usize,
    pub num_heads: usize,
}

impl MultiHeadLayerNormConfig {
    pub fn init<B: Backend>(&self, num_heads: usize, device: &B::Device) -> MultiHeadLayerNorm<B> {
        let norm = nn::GroupNormConfig::new(num_heads, self.ndim)
            .with_epsilon(self.eps)
            .init(device);
        
        MultiHeadLayerNorm {
            norm,
            ndim: self.ndim,
            num_heads,
        }
    }
}

impl<B: Backend> MultiHeadLayerNorm<B> {
    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, nh, s, dh] = x.dims();
        let x = x.swap_dims(1, 2).reshape([b * s, nh * dh]);
        let x = self.norm.forward(x);
        x.reshape([b, s, nh, dh]).swap_dims(1, 2)
    }
}