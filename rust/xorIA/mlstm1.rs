/*
# mLSTM: Matrix Long Short-Term Memory

This module implements the mLSTM (matrix LSTM) cell and layer as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The mLSTM extends the traditional LSTM by using a matrix memory state and exponential gating,
allowing for enhanced storage capacities and improved performance on long-range dependencies.
*/

use burn::{
    config::Config,
    module::{Module, Param},
    nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig},
    tensor::{Tensor, backend::Backend},
};
use num_traits::{FromPrimitive, ToPrimitive};
use burn::tensor::activation;
/// State for mLSTM containing cell matrix and hidden state
#[derive(Clone, Debug)]
pub struct MLstmstate<B: Backend> {
    /// Cell state - matrix of shape [`batch_size`, `num_heads`, `head_dim`, `head_dim`]
    pub cell: Tensor<B, 4>,
    /// Hidden state - vector of shape [`batch_size`, `hidden_size`]
    pub hidden: Tensor<B, 2>,
    /// Normalizer state - vector of shape [`batch_size`, `num_heads`, `head_dim`]
    pub normalizer: Tensor<B, 3>,
    /// Global max gate state for numeric stability - shape [`batch_size`, `num_heads`, 1]
    pub max_gate_log: Tensor<B, 3>,
}

impl<B: Backend> MLstmstate<B> {
    /// Create a new mLSTM state
    pub const fn new(
        cell: Tensor<B, 4>,
        hidden: Tensor<B, 2>,
        normalizer: Tensor<B, 3>,
        max_gate_log: Tensor<B, 3>,
    ) -> Self {
        Self {
            cell,
            hidden,
            normalizer,
            max_gate_log,
        }
    }

    /// Detach the state from the computational graph
    pub fn detach(self) -> Self {
        Self {
            cell: self.cell.detach(),
            hidden: self.hidden.detach(),
            normalizer: self.normalizer.detach(),
            max_gate_log: self.max_gate_log.detach(),
        }
    }
}

/// Configuration for mLSTM
#[derive(Config, Debug)]
pub struct MLstmconfig {
    /// Size of input features
    pub d_input: usize,
    /// Size of hidden state
    pub d_hidden: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of heads for multi-head mLSTM
    #[config(default = "4")]
    pub num_heads: usize,
    /// Dropout probability
    #[config(default = "0.0")]
    pub dropout: f64,
    /// Weight initializer (GPT-style gold standard)
    #[config(default = "Initializer::Normal{mean: 0.0, std: 0.02}")]
    pub initializer: Initializer,
    /// Weight standard deviation for some initializers (Not used for Normal)
    #[config(default = "0.02")]
    pub weight_stdev: f64,
    /// Forget gate neutral (log(1.0)=0)
    #[config(default = "0.0")]
    pub forget_bias: f32,
    /// Input gate bias
    #[config(default = "-1.0")] 
    pub input_bias: f32,
    /// Epsilon for numerical stability f
    #[config(default = "1e-6")]
    pub epsilon: f32,
    /// Minimum value for exponential clamping
    #[config(default = "-30.0")]
    pub exp_clamp_min: f32,
    /// Maximum value for exponential clamping
    #[config(default = "30.0")]
    pub exp_clamp_max: f32,
    /// Initial value for the stabilizer (Log-space neutral approx -infinity)
    #[config(default = "-10.0")]
    pub stabilizer_init: f32,
    /// Attention scale (Standard 1/sqrt(head_dim) = 0.125 for 256/4)
    #[config(default = "0.125")]
    pub attention_scale: f32,
    /// Projection expansion factor for internal matrix memory
    #[config(default = "2.0")]
    pub proj_factor: f32,
}
impl MLstmconfig {
    /// Initialize a new mLSTM
    pub fn init<B: Backend>(&self, device: &B::Device) -> MLstm<B> {
        let layers = (0..self.num_layers)
            .map(|i| {
                let input_size = if i == 0 { self.d_input } else { self.d_hidden };
                MLstmcell::new(input_size, self.d_hidden, self.num_heads, self, device)
            })
            .collect();

        MLstm {
            layers,
            dropout_layer: DropoutConfig::new(self.dropout).init(),
            d_input: self.d_input,
            d_hidden: self.d_hidden,
            num_layers: self.num_layers,
            num_heads: self.num_heads,
            dropout: self.dropout,
            forget_bias: self.forget_bias,
            input_bias: self.input_bias,
            epsilon: self.epsilon,
            exp_clamp_min: self.exp_clamp_min,
            exp_clamp_max: self.exp_clamp_max,
            stabilizer_init: self.stabilizer_init,
            proj_factor: self.proj_factor,
        }
    }
}

/// mLSTM layer implementation
#[derive(Module, Debug)]
pub struct MLstm<B: Backend> {
    /// Stack of mLSTM cells
    pub layers: alloc::vec::Vec<MLstmcell<B>>,
    /// Dropout module for inter-layer dropout
    pub dropout_layer: Dropout,
    /// Input size
    pub d_input: usize,
    /// Hidden size
    pub d_hidden: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of heads
    pub num_heads: usize,
    /// Dropout probability
    pub dropout: f64,
    /// Forget gate neutral (log(1.0)=0)
    pub forget_bias: f32,
    /// Input gate bias
    pub input_bias: f32,
    /// Epsilon for numerical stability
    pub epsilon: f32,
    /// Minimum value for exponential clamping
    pub exp_clamp_min: f32,
    /// Maximum value for exponential clamping
    pub exp_clamp_max: f32,
    /// Initial value for the stabilizer
    pub stabilizer_init: f32,
    /// Projection expansion factor
    pub proj_factor: f32,
}

impl<B: Backend> MLstm<B> {
    /// Forward pass through mLSTM consuming and returning states
    ///
    /// # Arguments
    /// * `input_seq` - Input tensor of shape [`batch_size`, `seq_length`, `input_size`]
    /// * `states` - States to consume (will be moved)
    ///
    /// # Returns
    /// * Output tensor of shape [`batch_size`, `seq_length`, `hidden_size`]
    /// * New states
    pub fn forward(
        &self,
        input_seq: &Tensor<B, 3>,
        states: Option<alloc::vec::Vec<MLstmstate<B>>>,
    ) -> (Tensor<B, 3>, alloc::vec::Vec<MLstmstate<B>>)
    where
        <B as Backend>::FloatElem: ToPrimitive + FromPrimitive + Copy,
    {
        self.forward_ext(input_seq, states, false)
    }

    /// Extended forward pass with  freezing support
    pub fn forward_ext(
        &self,
        input_seq: &Tensor<B, 3>,
        states: Option<alloc::vec::Vec<MLstmstate<B>>>,
        frozen: bool,
    ) -> (Tensor<B, 3>, alloc::vec::Vec<MLstmstate<B>>)
    where
        <B as Backend>::FloatElem: ToPrimitive + FromPrimitive + Copy,
    {
        let device = input_seq.device();
        let [batch_size, _seq_length, _] = input_seq.dims();

        // Inicializar estados
        let mut hidden_states = states.unwrap_or_else(|| self.init_hidden(batch_size, &device));
        let mut layer_input = input_seq.clone();

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let old_state = hidden_states[layer_idx].clone();
            let (h_seq, new_state) = layer.forward_sequence_ext(&layer_input, old_state, frozen);
            
            hidden_states[layer_idx] = new_state;

            layer_input = if layer_idx < self.num_layers - 1 && self.dropout > 0.0 {
                self.dropout_layer.forward(h_seq)
            } else {
                h_seq
            };
        }

        (layer_input, hidden_states)
    }

    /// Initialize hidden states
    fn init_hidden(&self, batch_size: usize, device: &B::Device) -> alloc::vec::Vec<MLstmstate<B>> {
        let internal_hidden_size = (self.d_hidden as f32 * self.proj_factor) as usize;
        let head_dim = internal_hidden_size / self.num_heads;
        
        (0..self.num_layers)
            .map(|_| {
                MLstmstate::new(
                    Tensor::zeros([batch_size, self.num_heads, head_dim, head_dim], device),
                    Tensor::zeros([batch_size, self.d_hidden], device),
                    Tensor::zeros([batch_size, self.num_heads, head_dim], device),
                    Tensor::zeros([batch_size, self.num_heads, 1], device).add_scalar(self.stabilizer_init),
                )
            })
            .collect()
    }
}

/// mLSTM cell implementation with matrix memory
#[derive(Module, Debug)]
pub struct MLstmcell<B: Backend> {
    /// Weight matrix for input to gates
    pub weight_ih: Param<Tensor<B, 2>>,
    /// Bias for gates
    pub bias: Param<Tensor<B, 1>>,
    /// Query projection
    pub w_q: Linear<B>,
    /// Key projection
    pub w_k: Linear<B>,
    /// Value projection
    pub w_v: Linear<B>,
    /// Output projection (Down-Projection)
    pub w_proj: Linear<B>,
    /// Input size
    pub input_size: usize,
    /// Target hidden size (output size)
    pub hidden_size: usize,
    /// Internal expanded hidden size
    pub internal_hidden_size: usize,
    /// Number of heads
    pub num_heads: usize,
    /// Epsilon for numerical stability
    pub epsilon: f32,
    /// Minimum value for exponential clamping
    pub exp_clamp_min: f32,
    /// Maximum value for exponential clamping
    pub exp_clamp_max: f32,
    /// Attention scale
    pub attention_scale: f32,
}

impl<B: Backend> MLstmcell<B> {
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_heads: usize,
        config: &MLstmconfig,
        device: &B::Device,
    ) -> Self {
        let internal_hidden_size = (hidden_size as f32 * config.proj_factor) as usize;
        let mut bias_data = alloc::vec![0.0; 3 * num_heads];
        for i in 0..num_heads {
            bias_data[i] = config.input_bias;             // Input gate: Aligned with config
            bias_data[i + num_heads] = config.forget_bias; // Forget gate: Aligned with config
            bias_data[i + 2 * num_heads] = 1.0;            // Output gate: Default 1.0 (matching mlstm1.rs)
        }
        let bias = Tensor::from_floats(bias_data.as_slice(), device);

        let head_dim = internal_hidden_size / num_heads;
        let attention_scale = 1.0 / (head_dim as f32).sqrt(); // <-- ESCALA DINÁMICA DE ATENCIÓN
        
        // Initialize Q, K, V with configured initializer to target internal expanded memory
        let w_q = LinearConfig::new(input_size, internal_hidden_size)
            .with_bias(false)
            .with_initializer(config.initializer.clone())
            .init(device);
        let w_k = LinearConfig::new(input_size, internal_hidden_size)
            .with_bias(false)
            .with_initializer(config.initializer.clone())
            .init(device);
        let w_v = LinearConfig::new(input_size, internal_hidden_size)
            .with_bias(false)
            .with_initializer(config.initializer.clone())
            .init(device);
        let weight_ih = config.initializer.init_with(
            [3 * num_heads, input_size],
            Some(input_size),
            Some(3 * num_heads),
            device,
        );
        let w_proj = LinearConfig::new(internal_hidden_size, hidden_size)
            .with_bias(true)
            .with_initializer(config.initializer.clone())
            .init(device);

        Self {
            weight_ih,
            bias: Param::from_tensor(bias),
            w_q,
            w_k,
            w_v,
            w_proj,
            input_size,
            hidden_size,
            internal_hidden_size,
            num_heads,
            epsilon: config.epsilon,
            exp_clamp_min: config.exp_clamp_min,
            exp_clamp_max: config.exp_clamp_max,
            attention_scale,
        }
    }

    /// Forward pass through mLSTM cell consuming the state
    ///
    /// # Arguments
    /// * `input` - Input tensor [`batch_size`, `input_size`]
    /// * `state` - State to consume (moved)
    ///
    /// # Returns
    /// * New hidden state (for output)
    /// * New complete state
    pub fn forward_sequence(
        &self,
        input_seq: &Tensor<B, 3>,
        state: MLstmstate<B>,
    ) -> (Tensor<B, 3>, MLstmstate<B>)
    where
        <B as Backend>::FloatElem: num_traits::ToPrimitive + num_traits::FromPrimitive + Copy,
    {
        self.forward_sequence_ext(input_seq, state, false)
    }

    pub fn forward_sequence_ext(
        &self,
        input_seq: &Tensor<B, 3>,
        state: MLstmstate<B>,
        frozen: bool,
    ) -> (Tensor<B, 3>, MLstmstate<B>)
    where
        <B as Backend>::FloatElem: num_traits::ToPrimitive + num_traits::FromPrimitive + Copy,
    {
        let [batch_size, seq_len, _] = input_seq.dims();
        let head_dim = self.internal_hidden_size / self.num_heads;
        let device = input_seq.device();

        // 1. Parallel Projections (Q, K, V)
        let q = self.w_q.forward(input_seq.clone())
            .reshape::<4, _>([batch_size, seq_len, self.num_heads, head_dim])
            .swap_dims(1, 2); // [B, H, S, D_h]
        let k = self.w_k.forward(input_seq.clone())
            .reshape::<4, _>([batch_size, seq_len, self.num_heads, head_dim])
            .swap_dims(1, 2);
        let v = self.w_v.forward(input_seq.clone())
            .reshape::<4, _>([batch_size, seq_len, self.num_heads, head_dim])
            .swap_dims(1, 2);

        // 2. Parallel Gates (Scalar per head)
        let weight_ih_val = self.weight_ih.val().transpose();
        let bias_val = self.bias.val();

        // Proyección directa a num_heads
        let gates = input_seq.clone().matmul(weight_ih_val.reshape::<3, _>([1, self.input_size, 3 * self.num_heads])) 
                    + bias_val.reshape::<3, _>([1, 1, 3 * self.num_heads]);
        
        let i_log = gates.clone().slice([0..batch_size, 0..seq_len, 0..self.num_heads]).swap_dims(1, 2).reshape::<4, _>([batch_size, self.num_heads, seq_len, 1]); // [B, H, S, 1]
        let f_log = gates.clone().slice([0..batch_size, 0..seq_len, self.num_heads..2*self.num_heads]).swap_dims(1, 2).reshape::<4, _>([batch_size, self.num_heads, seq_len, 1]);
        
        // Estabilización por Clamping (PAPER ACCURATE)
        let i_log = i_log.clamp(self.exp_clamp_min, self.exp_clamp_max);
        let f_log = f_log.clamp(self.exp_clamp_min, self.exp_clamp_max);
        let o_log = gates.clone().slice([0..batch_size, 0..seq_len, 2*self.num_heads..3*self.num_heads])
            .swap_dims(1, 2)
            .reshape::<4, _>([batch_size, self.num_heads, seq_len, 1]); // [B, H, S, 1]
        
        let o_log = o_log.clamp(self.exp_clamp_min, self.exp_clamp_max);


        // Forget gate log-space stable (PAPER ACCURATE: pure log-projection)
        let f_log_val = f_log; 
        
        // 3. Parallel Stabilization Logic (Log-Space) - PAPER ACCURATE (10/10)
        // i_log y f_log ya son escalares por cabeza [B, H, S, 1]
        let i_log_scalar = i_log; 
        let f_log_scalar = f_log_val; 
        
        // Manual CumSum using triangular matrix: [1, 1, S, S] @ [B, H, S, 1] -> [B, H, S, 1]
        let mask_tri = Tensor::<B, 2>::tril(Tensor::ones([seq_len, seq_len], &device), 0);
        let f_log_cumsum = mask_tri.clone().reshape::<4, _>([1, 1, seq_len, seq_len]).matmul(f_log_scalar.clone());
        
        // Matrix of decay weights in 4D: [B, H, S_t, S_k]
        // log_f_matrix[t, k] = sum_{j=k+1}^t log f_j = F[t] - F[k]
        let f_t = f_log_cumsum.clone(); // [B, H, S, 1]
        let f_k = f_log_cumsum.clone().swap_dims(2, 3); // [B, H, 1, S]
        let log_f_matrix = f_t - f_k;
        
        // log_weights[t, k] = log_f_matrix + i_log[k]
        let i_k = i_log_scalar.clone().swap_dims(2, 3); // [B, H, 1, S]
        let log_weights = log_f_matrix + i_k;
        
        // Causal Masking
        let mask_4d = mask_tri.reshape::<4, _>([1, 1, seq_len, seq_len]);
        let log_weights_masked = log_weights.mask_fill(mask_4d.equal(Tensor::zeros([1, 1, seq_len, seq_len], &device)), -1e30);
        
        // Contribución del estado inicial: m_0 + sum log f
        // NOTA: No sumamos log_scale aquí porque se aplica en el producto final (evita doble escalado)
        let m_0 = state.max_gate_log.clone().reshape::<4, _>([batch_size, self.num_heads, 1, 1]); 
        let log_initial_contrib = f_log_cumsum.clone() + m_0; // [B, H, S, 1]
        
        // m_t global = max( max_k(weights), log_initial_contrib )
        let max_seq = log_weights_masked.clone().max_dim(3); // [B, H, S, 1]
        let m_t_global = max_seq.max_pair(log_initial_contrib.clone()); // [B, H, S, 1]
        
        // Exponenciales estables para la memoria (Escalares por cabeza)
        let weights = (log_weights_masked - m_t_global.clone()).exp(); // [B, H, S, S]
        let initial_scale = (log_initial_contrib - m_t_global.clone()).exp(); // [B, H, S, 1]
        
        // --- Puerta de salida (PAPER ACCURATE: Exponential gating stabilized) ---
        let o_gate = (o_log - m_t_global.clone()).exp();
        
        // H_parallel = (weights @ V)  -- wait, weights is [t, k]
        // We need sum_k weights[t, k] * (v[k] * k[k]^T) ? No. 
        // Standard attention form: H = (weights @ (K^T \odot V))? No.
        // xLSTM Matrix memory:
        // C_t = sum (v_k * k_k^T * decay)
        // h_t = q_t * C_t = sum (q_t * v_k * k_k^T * decay)
        //     = sum ((q_t * v_k^T) * k_k)? No. 
        //     = sum (scalar(q_t, k_k) * v_k) ? No, that's regular attention.
        //     = q_t * (sum v_k k_k^T) 
        //     = sum (q_t k_k) v_k ? No, matrix multiply order.
        //     = q_t * V * K^T ?
        // Let's check dimensions.
        // q: [B, H, S, D], k: [B, H, S, D], v: [B, H, S, D]
        // C is [D, D].
        // C = K^T * V ? ([D, S] * [S, D] -> [D, D])
        // h = Q * C = Q * K^T * V?
        // Assoc: (Q K^T) V. Yes.
        // So we compute Attention(Q, K, V) with our specific decay weights.
        // weights[t, k] acts as the "attention score" A[t, k].
        
        // --- Numerator (C_t @ q_t) ---
        // h_parallel = sum_k weights[t, k] * (q_t @ k_k^T) * v_k
       
       /*
        // 1. Producto punto q * k_k^T para todas las combinaciones t, k
        let qk = q.clone().matmul(k.clone().swap_dims(2, 3)); // [B, H, S, S]
        */
        // 1. Producto punto q * k_k^T (K ya viene escalada)
        let qk = q.clone().matmul(k.clone().swap_dims(2, 3)) * self.attention_scale; // [B, H, S, S]

        // 2. Aplicamos los pesos de decaimiento escalares a las puntuaciones de atención
        let attention_scores = weights.clone() * qk; // [B, H, S, S]
       
        // 3. Resultado final con valores v
        let h_parallel = attention_scores.clone().matmul(v.clone()); // [B, H, S, D]
        
        // 4. Contribución del estado inicial (Eq. 21)
        let h_initial = (q.clone() * self.attention_scale)
            .matmul(state.cell.clone().swap_dims(2, 3)) * initial_scale.clone();

        // --- Denominator (Eq. 21: max(|nt^T qt|, 1)) ---
        // n_parallel = sum_k weights[t, k] * k_k
        let n_parallel = weights.clone().matmul(k.clone()); // [B, H, S, D]
        let n_dot_q_parallel = (n_parallel * (q.clone() * self.attention_scale)).sum_dim(3).reshape::<4, _>([batch_size, self.num_heads, seq_len, 1]); // [B, H, S, 1]
        
        let n_initial_dot_q = ((q.clone() * self.attention_scale) * state.normalizer.clone().reshape::<4, _>([batch_size, self.num_heads, 1, head_dim]))
            .sum_dim(3)
            .reshape::<4, _>([batch_size, self.num_heads, seq_len, 1]) * initial_scale.clone();

        let denominator = n_dot_q_parallel + n_initial_dot_q;

        // Estabilización final escalar (PAPER ACCURATE: max(|n^T q|, 1))
        let denominator_stable = denominator.clone().abs().max_pair(Tensor::ones_like(&denominator)); 
        
        let h_normalized = (h_parallel + h_initial) / denominator_stable;
        
        // --- Output Gate ---
        let h_gated = h_normalized * o_gate;
        
        // Recombinar cabezas para la salida final antes de proyección
        let h_seq_internal = h_gated.swap_dims(1, 2).reshape::<3, _>([batch_size, seq_len, self.internal_hidden_size]);

        // Compactación final (Down-Projection) a hidden_size
        let y_t = self.w_proj.forward(h_seq_internal.clone());

        // --- State Update for Next Step (FAITHFUL TO PAPER) ---
        let last_idx = seq_len - 1;
        
        // 1. m_T
        let final_m = m_t_global.slice([0..batch_size, 0..self.num_heads, last_idx..seq_len, 0..1]).reshape::<3, _>([batch_size, self.num_heads, 1]);
        
        // 2. n_T = exp(F_T - m_T) * n_0 + sum_k (weights[T, k] * k_k)
        let last_scale = initial_scale.slice([0..batch_size, 0..self.num_heads, last_idx..seq_len, 0..1]).reshape::<3, _>([batch_size, self.num_heads, 1]);
        let n_initial_contrib = state.normalizer.clone() * last_scale.clone().expand([batch_size, self.num_heads, head_dim]);
        
        let last_weights = weights.slice([0..batch_size, 0..self.num_heads, last_idx..seq_len, 0..seq_len]); 
        let n_parallel_contrib = last_weights.clone().matmul(k.clone()).reshape::<3, _>([batch_size, self.num_heads, head_dim]);
        let final_norm = n_initial_contrib + n_parallel_contrib;

        // 3. C_T = exp(F_T - m_T) * C_0 + sum_k (weights[T, k] * v_k @ k_k^T)
        let c_initial_contrib = state.cell.clone() * last_scale.reshape::<4, _>([batch_size, self.num_heads, 1, 1]);
        
        // sum_k weights[T, k] * (v_k @ k_k^T) --> (v_weighted^T @ k)
        let v_weighted = v * last_weights.reshape::<4, _>([batch_size, self.num_heads, seq_len, 1]);
        let c_parallel_contrib = v_weighted.swap_dims(2, 3).matmul(k); 
        let final_cell = c_initial_contrib + c_parallel_contrib;

        let final_state = if frozen { 
            state 
        } else { 
            MLstmstate::new(
                final_cell, 
                y_t.clone().slice([0..batch_size, last_idx..last_idx+1, 0..self.hidden_size]).reshape([batch_size, self.hidden_size]), 
                final_norm, 
                final_m
            ) 
        };
        (y_t, final_state)
    }

    /// Forward pass through mLSTM cell with optional state freezing
    pub fn forward_step(
        &self,
        input: &Tensor<B, 2>,
        state: MLstmstate<B>,
        frozen: bool,
    ) -> (Tensor<B, 2>, MLstmstate<B>)
    where
        <B as Backend>::FloatElem: num_traits::ToPrimitive + num_traits::FromPrimitive + Copy,
    {
        let cell = state.cell.clone();
        let normalizer = state.normalizer.clone();
        let max_gate_log = state.max_gate_log.clone();
        let [batch_size, _] = input.dims();
        let head_dim = self.internal_hidden_size / self.num_heads;

        // Gates calculation
        let gates = input.clone().matmul(self.weight_ih.val().transpose())
            + self.bias.val().reshape::<2, _>([1, 3 * self.num_heads]);

        let chunks = gates.chunk(3, 1);
        let i_log = chunks[0].clone().reshape::<2, _>([batch_size, self.num_heads]).clamp(self.exp_clamp_min, self.exp_clamp_max);
        let f_log = chunks[1].clone().reshape::<2, _>([batch_size, self.num_heads]).clamp(self.exp_clamp_min, self.exp_clamp_max);
        let o_log = chunks[2].clone().reshape::<2, _>([batch_size, self.num_heads]).clamp(self.exp_clamp_min, self.exp_clamp_max);


        // Projections
        let q = self.w_q.forward(input.clone()).reshape::<4, _>([batch_size, self.num_heads, 1, head_dim]);
        let k = self.w_k.forward(input.clone()).reshape::<4, _>([batch_size, self.num_heads, 1, head_dim]);
        let v = self.w_v.forward(input.clone()).reshape::<4, _>([batch_size, self.num_heads, head_dim, 1]);

        let m_t_minus_1 = max_gate_log.clone().reshape::<2, _>([batch_size, self.num_heads]); 
        // m_t = max(f_log + m_prev, i_log)
        let m_t = if frozen { m_t_minus_1.clone() } else { 
            (f_log.clone() + m_t_minus_1.clone()).max_pair(i_log.clone()) 
        };
        
        let f_stable = (f_log + m_t_minus_1 - m_t.clone()).exp();
        let i_stable = (i_log - m_t.clone()).exp();
        
        // --- Cálculo de C y N ---
        let (c_new, n_new) = if frozen {
            // En modo congelado, la matriz no cambia
            (cell.clone(), normalizer.clone())
        } else {
            let f_exp = f_stable.clone().reshape::<4, _>([batch_size, self.num_heads, 1, 1]).expand([batch_size, self.num_heads, head_dim, head_dim]); 
            let i_exp = i_stable.clone().reshape::<4, _>([batch_size, self.num_heads, 1, 1]).expand([batch_size, self.num_heads, head_dim, head_dim]);
            let cell_update = v.clone().matmul(k.clone());
            let cn = cell.clone() * f_exp + cell_update * i_exp;
            
            let nn = normalizer.clone() * f_stable.clone().reshape::<3, _>([batch_size, self.num_heads, 1]).expand([batch_size, self.num_heads, head_dim]) 
                      + k.reshape::<3, _>([batch_size, self.num_heads, head_dim]) * i_stable.clone().reshape::<3, _>([batch_size, self.num_heads, 1]).expand([batch_size, self.num_heads, head_dim]);
            (cn, nn)
        };

        // Read Memory: C_t @ q_t (Con escalado de atención)
        let h_heads = (q.clone() * self.attention_scale).matmul(c_new.clone().swap_dims(2, 3)).squeeze::<3>(2);
        
        let q_step_scaled = (q.clone() * self.attention_scale).reshape::<3, _>([batch_size, self.num_heads, head_dim]);
        let denominator = (n_new.clone() * q_step_scaled).sum_dim(2).reshape([batch_size, self.num_heads, 1]);
        
        let denominator_stable = denominator.clone().abs().max_pair(Tensor::ones_like(&denominator));
        let h_normalized = h_heads / denominator_stable;
        
        // o_gate stabilized (PAPER ACCURATE)
        let o_gate = (o_log.reshape::<3, _>([batch_size, self.num_heads, 1]) - m_t.clone().reshape::<3, _>([batch_size, self.num_heads, 1])).exp();
        let h_new = (h_normalized * o_gate).reshape::<2, _>([batch_size, self.internal_hidden_size]);

        // Compactación final (Down-Projection)
        let y_new = self.w_proj.forward(h_new.clone());

        if frozen {
            (y_new, state)
        } else {
            let m_final = m_t.reshape::<3, _>([batch_size, self.num_heads, 1]);
            (y_new.clone(), MLstmstate::new(c_new, y_new, n_new, m_final))
        }
    }

    pub fn forward(
        &self,
        input: &Tensor<B, 2>,
        state: MLstmstate<B>,
    ) -> (Tensor<B, 2>, MLstmstate<B>)
    where
        <B as Backend>::FloatElem: num_traits::ToPrimitive + num_traits::FromPrimitive + Copy,
    {
        self.forward_step(input, state, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;

    type TestBackend = burn_ndarray::NdArray<f32>;

    #[test]
    fn test_mlstm_forward() {
        let device = Default::default();
        let config = MLstmconfig::new(64, 128, 2).with_dropout(0.1);
        let mlstm = config.init::<TestBackend>(&device);

        let input = Tensor::<TestBackend, 3>::random([4, 10, 64], Distribution::Default, &device);

        let (output, states) = mlstm.forward(&input, None);

        assert_eq!(output.dims(), [4, 10, 128]);
        assert_eq!(states.len(), 2);
        assert_eq!(states[0].hidden.dims(), [4, 128]);
        assert_eq!(states[0].cell.dims(), [4, 4, 64, 64]);
    }
// // 
// //
    #[test]
    fn test_mlstm_cell() {
        let device = Default::default();
        let num_heads = 4;
        let config = MLstmconfig::new(32, 64, 1);
        let cell = MLstmcell::new(32, 64, num_heads, &config, &device);

        let input = Tensor::<TestBackend, 2>::random([4, 32], Distribution::Default, &device);
        let state = MLstmstate::new(
            Tensor::<TestBackend, 4>::zeros([4, num_heads, 32, 32], &device),
            Tensor::<TestBackend, 2>::zeros([4, 64], &device),
            Tensor::<TestBackend, 3>::zeros([4, num_heads, 32], &device),
            Tensor::<TestBackend, 3>::zeros([4, num_heads, 1], &device),
        );

        let (h_new, new_state) = cell.forward(&input, state);

        assert_eq!(h_new.dims(), [4, 64]);
        assert_eq!(new_state.cell.dims(), [4, 4, 32, 32]);
        assert_eq!(new_state.hidden.dims(), [4, 64]);
    }

    #[test]
    fn test_mlstm_state_reuse() {
        let device = Default::default();
        let config = MLstmconfig::new(32, 64, 1);
        let mlstm = config.init::<TestBackend>(&device);

        let input1 = Tensor::<TestBackend, 3>::random([2, 5, 32], Distribution::Default, &device);
        let input2 = Tensor::<TestBackend, 3>::random([2, 5, 32], Distribution::Default, &device);

        // First forward pass
        let (output1, states) = mlstm.forward(&input1, None);

        // Second forward pass reusing states
        let (output2, _final_states) = mlstm.forward(&input2, Some(states));

        assert_eq!(output1.dims(), [2, 5, 64]);
        assert_eq!(output2.dims(), [2, 5, 64]);
    }
}