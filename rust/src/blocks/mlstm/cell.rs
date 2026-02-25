// mLSTM Cell
// Matches Python: blocks/mlstm/cell.py
//
// The cell computes the core mLSTM operation:
// - Takes q, k, v projections
// - Computes input/forget gate pre-activations from concatenated [q, k, v]
// - Delegates to parallel or recurrent backend
// - Applies MultiHeadLayerNorm to the output

use burn::prelude::*;
use burn::module::Module;
use burn::config::Config;
use burn::nn;

use crate::components::ln::{MultiHeadLayerNorm, MultiHeadLayerNormConfig};
use crate::components::init::bias_linspace_init;
use super::backends::{parallel_stabilized_simple, recurrent_step_stabilized_simple};

#[derive(Config)]
pub struct MLSTMCellConfig {
    pub context_length: usize,
    pub embedding_dim: usize,
    pub num_heads: usize,
}

#[derive(Module, Debug)]
pub struct MLSTMCell<B: Backend> {
    /// Input gate: Linear(3*embedding_dim → num_heads)
    pub igate: nn::Linear<B>,
    /// Forget gate: Linear(3*embedding_dim → num_heads)
    pub fgate: nn::Linear<B>,
    /// Output normalization (GroupNorm per head)
    pub outnorm: MultiHeadLayerNorm<B>,
    pub num_heads: usize,
    pub embedding_dim: usize,
}

impl MLSTMCellConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLSTMCell<B> {
        // igate: Linear(3*D, NH)
        let igate = nn::LinearConfig::new(3 * self.embedding_dim, self.num_heads)
            .with_bias(true)
            .init(device);

        // fgate: Linear(3*D, NH), bias initialized with linspace(3, 6)
        let fgate = nn::LinearConfig::new(3 * self.embedding_dim, self.num_heads)
            .with_bias(true)
            .init(device);

        let outnorm = MultiHeadLayerNormConfig {
            ndim: self.embedding_dim,
            weight: true,
            bias: false,
            eps: 1e-5,
        }
        .init(self.num_heads, device);

        // We'll set the biases in a post-init step
        let mut cell = MLSTMCell {
            igate,
            fgate,
            outnorm,
            num_heads: self.num_heads,
            embedding_dim: self.embedding_dim,
        };

        // Reset parameters as in Python
        cell.reset_parameters(device);
        cell
    }
}

impl<B: Backend> MLSTMCell<B> {
    /// Reset gate parameters to match the Python initialization.
    fn reset_parameters(&mut self, device: &B::Device) {
        // Forget gate bias: linspace(3.0, 6.0)
        let fgate_bias = bias_linspace_init::<B>(3.0, 6.0, self.num_heads, device);
        self.fgate.bias = Some(burn::module::Param::from_tensor(fgate_bias));

        // Forget gate weight: zeros
        let fgate_w = Tensor::zeros(self.fgate.weight.dims(), device);
        self.fgate.weight = burn::module::Param::from_tensor(fgate_w);

        // Input gate weight: zeros, bias: normal(0, 0.1)
        let igate_w = Tensor::zeros(self.igate.weight.dims(), device);
        self.igate.weight = burn::module::Param::from_tensor(igate_w);
        let igate_bias = Tensor::random(
            [self.num_heads],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            device,
        );
        self.igate.bias = Some(burn::module::Param::from_tensor(igate_bias));
    }

    /// Parallel forward pass.
    ///
    /// # Arguments
    /// * `q` — (B, S, H) queries
    /// * `k` — (B, S, H) keys
    /// * `v` — (B, S, H) values
    ///
    /// # Returns
    /// * (B, S, H) normalized hidden state
    pub fn forward(&self, q: Tensor<B, 3>, k: Tensor<B, 3>, v: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, _h] = q.dims();
        let dh = self.embedding_dim / self.num_heads;

        // Concatenate for gate input
        let if_gate_input = Tensor::cat(vec![q.clone(), k.clone(), v.clone()], 2); // (B, S, 3*H)

        // Reshape to multi-head: (B, S, NH, DH) → (B, NH, S, DH)
        let q = q.reshape([b, s, self.num_heads, dh]).swap_dims(1, 2);
        let k = k.reshape([b, s, self.num_heads, dh]).swap_dims(1, 2);
        let v = v.reshape([b, s, self.num_heads, dh]).swap_dims(1, 2);

        // Gate pre-activations: (B, S, NH)
        let igate_preact = self.igate.forward(if_gate_input.clone());
        let igate_preact = igate_preact.swap_dims(1, 2).unsqueeze_dim(3); // (B, NH, S, 1)

        let fgate_preact = self.fgate.forward(if_gate_input);
        let fgate_preact = fgate_preact.swap_dims(1, 2).unsqueeze_dim(3); // (B, NH, S, 1)

        // Backend
        let h_state = parallel_stabilized_simple(q, k, v, igate_preact, fgate_preact); // (B, NH, S, DH)

        // Output norm
        let h_state_norm = self.outnorm.forward(h_state); // (B, NH, S, DH)
        // (B, NH, S, DH) → (B, S, NH, DH) → (B, S, H)
        h_state_norm.swap_dims(1, 2).reshape([b, s, self.embedding_dim])
    }

    /// Recurrent step.
    ///
    /// # Arguments
    /// * `q` — (B, 1, H)
    /// * `k` — (B, 1, H)  
    /// * `v` — (B, 1, H)
    /// * `mlstm_state` — Option<(c, n, m)> where c:(B,NH,DH,DH), n:(B,NH,DH,1), m:(B,NH,1,1)
    ///
    /// # Returns
    /// (output (B,1,H), (c_new, n_new, m_new))
    pub fn step(
        &self,
        q: Tensor<B, 3>,
        k: Tensor<B, 3>,
        v: Tensor<B, 3>,
        mlstm_state: Option<(Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>)>,
    ) -> (Tensor<B, 3>, (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>)) {
        let [b, s, _h] = q.dims();
        let dh = self.embedding_dim / self.num_heads;
        let device = q.device();

        let if_gate_input = Tensor::cat(vec![q.clone(), k.clone(), v.clone()], 2);

        let q = q.reshape([b, s, self.num_heads, dh]).swap_dims(1, 2);
        let k = k.reshape([b, s, self.num_heads, dh]).swap_dims(1, 2);
        let v = v.reshape([b, s, self.num_heads, dh]).swap_dims(1, 2);

        let igate_preact = self.igate.forward(if_gate_input.clone())
            .swap_dims(1, 2).unsqueeze_dim(3);
        let fgate_preact = self.fgate.forward(if_gate_input)
            .swap_dims(1, 2).unsqueeze_dim(3);

        let (c_state, n_state, m_state) = match mlstm_state {
            Some(state) => state,
            None => {
                let c = Tensor::zeros([b, self.num_heads, dh, dh], &device);
                let n = Tensor::zeros([b, self.num_heads, dh, 1], &device);
                let m = Tensor::zeros([b, self.num_heads, 1, 1], &device);
                (c, n, m)
            }
        };

        let (h_state, new_state) = recurrent_step_stabilized_simple(
            c_state, n_state, m_state, q, k, v, igate_preact, fgate_preact,
        );

        let h_state_norm = self.outnorm.forward(h_state);
        let output = h_state_norm.swap_dims(1, 2).reshape([b, s, self.embedding_dim]);

        (output, new_state)
    }
}
