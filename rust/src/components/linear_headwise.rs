// LinearHeadwiseExpand
// Matches Python: components/linear_headwise.py
//
// A structured projection that expands inputs per-head.
// weight shape: (num_heads, out_features_per_head, in_features_per_head)
// y[..., h, :] = x[..., h, :] @ weight[h].T + bias

use burn::prelude::*;
use burn::module::{Module, Param};
use burn::config::Config;
use burn::tensor::Distribution;

#[derive(Config)]
pub struct LinearHeadwiseExpandConfig {
    pub in_features: usize,
    pub num_heads: usize,
    #[config(default = 1.0)]
    pub expand_factor_up: f64,
    #[config(default = false)]
    pub bias: bool,
}

impl LinearHeadwiseExpandConfig {
    pub fn out_features(&self) -> usize {
        (self.expand_factor_up * self.in_features as f64).round() as usize
    }
}

#[derive(Module, Debug)]
pub struct LinearHeadwiseExpand<B: Backend> {
    /// Weight: (num_heads, out_per_head, in_per_head)
    pub weight: Param<Tensor<B, 3>>,
    /// Optional bias: (out_features,)
    pub bias: Option<Param<Tensor<B, 1>>>,
    pub num_heads: usize,
    pub in_features: usize,
    pub out_features: usize,
}

impl LinearHeadwiseExpandConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> LinearHeadwiseExpand<B> {
        let out_features = self.out_features();
        let out_per_head = out_features / self.num_heads;
        let in_per_head = self.in_features / self.num_heads;

        // small init: std = sqrt(2 / (5 * in_per_head))
        let std = (2.0 / (5.0 * in_per_head as f64)).sqrt();
        let weight = Tensor::random(
            [self.num_heads, out_per_head, in_per_head],
            Distribution::Normal(0.0, std),
            device,
        );

        let bias = if self.bias {
            Some(Param::from_tensor(Tensor::zeros([out_features], device)))
        } else {
            None
        };

        LinearHeadwiseExpand {
            weight: Param::from_tensor(weight),
            bias,
            num_heads: self.num_heads,
            in_features: self.in_features,
            out_features,
        }
    }
}

impl<B: Backend> LinearHeadwiseExpand<B> {
    /// Forward: input shape (..., in_features) → output (..., out_features).
    /// The last dim is split into (num_heads, in_per_head), each head is projected,
    /// and the result is concatenated back.
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, _f] = x.dims();
        let in_per_head = self.in_features / self.num_heads;
        let out_per_head = self.out_features / self.num_heads;

        // (B, S, F) → (B, S, NH, in_per_head) → (B, NH, S, in_per_head)
        let x = x.reshape([b, s, self.num_heads, in_per_head]).swap_dims(1, 2);

        // For each head h: out[b,h,s,:] = x[b,h,s,:] @ weight[h].T
        // weight: (NH, O, I), x: (B, NH, S, I)
        // We want: (B, NH, S, I) @ (NH, I, O) → (B, NH, S, O) via batched matmul
        let w = self.weight.val().unsqueeze_dim::<4>(0); // (1, NH, O, I)
        let w = w.swap_dims(2, 3); // (1, NH, I, O)
        let out = x.matmul(w); // (B, NH, S, O)

        // (B, NH, S, O) → (B, S, NH, O) → (B, S, NH*O)
        let out = out.swap_dims(1, 2).reshape([b, s, self.num_heads * out_per_head]);

        if let Some(bias) = &self.bias {
            out + bias.val().unsqueeze_dim::<2>(0).unsqueeze_dim::<3>(0)
                .reshape([1, 1, self.out_features])
        } else {
            out
        }
    }
}
