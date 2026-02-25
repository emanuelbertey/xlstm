// --- VARIABLES DE AJUSTE (CANEZERA) ---
pub const MINGRU_CLAMP: f64 = 20.0;
pub const MINGRU_GATE_BIAS: f32 = -1.0;
pub const MINGRU_HIDDEN_GAIN: f64 = 1.0;
// --------------------------------------

// minGRU - Minimal GRU paralelo
// zt  = σ(Linear(xt))
// h̃t  = Linear(xt)          ← fórmula exacta del paper
// ht  = (1-zt)⊙ht-1 + zt⊙h̃t

use burn::{
    config::Config,
    module::Module,
    nn::{Initializer, Linear, LinearConfig},
    tensor::{Tensor, activation, backend::Backend},
};
use num_traits::{FromPrimitive, ToPrimitive};

#[derive(Clone, Debug)]
pub struct MinGruState<B: Backend> {
    pub hidden: Tensor<B, 2>,
}

impl<B: Backend> MinGruState<B> {
    pub fn new(hidden: Tensor<B, 2>) -> Self { Self { hidden } }
    pub fn detach(self) -> Self { Self { hidden: self.hidden.detach() } }
}

#[derive(Config, Debug)]
pub struct MinGruConfig {
    pub d_input: usize,
    pub d_hidden: usize,
    pub num_layers: usize,
    #[config(default = "0.0")]
    pub dropout: f64,
    #[config(default = "15.0")]
    pub clamp_max: f64,
    #[config(default = "-3.0")]
    pub gate_bias: f32,
    #[config(default = "Initializer::Normal{mean: 0.0, std: 0.02}")]
    pub initializer: Initializer,
}

impl MinGruConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> MinGru<B> {
        let layers = (0..self.num_layers)
            .map(|i| {
                let input_size = if i == 0 { self.d_input } else { self.d_hidden };
                MinGruLayer::new(input_size, self.d_hidden, self.gate_bias, &self.initializer, device)
            })
            .collect();
        MinGru { layers, d_hidden: self.d_hidden, num_layers: self.num_layers }
    }
}

#[derive(Module, Debug)]
pub struct MinGru<B: Backend> {
    pub layers: alloc::vec::Vec<MinGruLayer<B>>,
    pub d_hidden: usize,
    pub num_layers: usize,
}

impl<B: Backend> MinGru<B> {
    pub fn forward(
        &self,
        input_seq: Tensor<B, 3>,
        states: Option<alloc::vec::Vec<MinGruState<B>>>,
    ) -> (Tensor<B, 3>, alloc::vec::Vec<MinGruState<B>>)
    where
        <B as Backend>::FloatElem: ToPrimitive + FromPrimitive + Copy,
    {
        self.forward_ext(input_seq, states, false)
    }

    pub fn forward_ext(
        &self,
        input_seq: Tensor<B, 3>,
        states: Option<alloc::vec::Vec<MinGruState<B>>>,
        frozen: bool,
    ) -> (Tensor<B, 3>, alloc::vec::Vec<MinGruState<B>>)
    where
        <B as Backend>::FloatElem: ToPrimitive + FromPrimitive + Copy,
    {
        let [batch_size, _seq_len, _] = input_seq.dims();
        let device = input_seq.device();

        // Inicializamos o recuperamos estados
        let hidden_states = states.unwrap_or_else(|| {
            core::iter::repeat_with(|| MinGruState::new(Tensor::zeros([batch_size, self.d_hidden], &device)))
                .take(self.num_layers)
                .collect()
        });

        // Procesamos todas las capas de forma encadenada
        // El bucle de tiempo ya es paralelo dentro de layer.forward
        let mut new_states = alloc::vec::Vec::with_capacity(self.num_layers);
        let output = self.layers.iter().zip(hidden_states).fold(input_seq, |x, (layer, state)| {
            let (out, ns) = layer.forward(x, state, frozen);
            new_states.push(ns);
            out
        });

        (output, new_states)
    }
}

#[derive(Module, Debug)]
pub struct MinGruLayer<B: Backend> {
    pub linear_z: Linear<B>,  // update gate
    pub linear_h: Linear<B>,  // candidate
    pub d_hidden: usize,
    pub clamp_max: f64,
}

impl<B: Backend> MinGruLayer<B> {
    pub fn new(input_size: usize, hidden_size: usize, gate_bias: f32, initializer: &Initializer, device: &B::Device) -> Self {
        let linear_z = LinearConfig::new(input_size, hidden_size)
            .with_bias(true)
            .with_initializer(initializer.clone())
            .init(device);
            
        let mut linear_z = linear_z;
        // Bias negativo fuerte → memoria persistente (zt muy pequeño → ht ≈ ht-1)
        let b_t = Tensor::ones([hidden_size], device) * gate_bias;
        linear_z.bias = Some(burn::module::Param::from_tensor(b_t));

        let linear_h = LinearConfig::new(input_size, hidden_size)
            .with_bias(false)
            .with_initializer(initializer.clone())
            .init(device);

        Self { linear_z, linear_h, d_hidden: hidden_size, clamp_max: MINGRU_CLAMP }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        state: MinGruState<B>,
        frozen: bool,
    ) -> (Tensor<B, 3>, MinGruState<B>)
    where
        <B as Backend>::FloatElem: ToPrimitive + FromPrimitive + Copy,
    {
        let [batch_size, seq_len, _] = x.dims();
        let device = x.device();

        // 1. Proyecciones Paralelas
        // z_gate = Linear_z(x), h_tilde = Linear_h(x)
        let z_logits = self.linear_z.forward(x.clone()).clamp(-self.clamp_max, self.clamp_max);
        let h_tilde = self.linear_h.forward(x).clamp(-self.clamp_max, self.clamp_max);

        // 2. Coeficientes en Espacio Logarítmico para Estabilidad Máxima
        // log(a_t) = log(1 - sigmoid(z_t)) = log_sigmoid(-z_t)
        // log(b_t) = log(sigmoid(z_t)) = log_sigmoid(z_t)
        let log_a = activation::log_sigmoid(z_logits.clone().neg()); // log(retención)
        let log_b = activation::log_sigmoid(z_logits);               // log(entrada)

        // 3. Parallel Scan via Log-CumSum
        // 3. O(S) Parallel Scan via cumulative computation (manual cumsum)
        let mask_tri = Tensor::<B, 2>::tril(Tensor::ones([seq_len, seq_len], &device), 0)
            .reshape::<3, _>([1, seq_len, seq_len]);
        
        // p_t = exp(cumsum(log_a))
        let log_p = mask_tri.clone().matmul(log_a); // [B, S, H]
        let p_t = log_p.clone().exp();
        
        // v_t = log_b.exp() * h_tilde
        let v_t = log_b.exp() * h_tilde;
        
        // terms = v_t / P_t
        let epsilon = 1e-12;
        let terms = v_t / (p_t.clone() + epsilon); 
        
        // running_sum = cumsum(terms)
        let running_sum = mask_tri.matmul(terms); // [B, S, H]
        
        // h_seq_parallel = P_t * running_sum
        let h_seq_parallel = p_t.clone() * running_sum;

        // 4. Contribución del Estado Inicial
        // h_0 es [B, H]. Se escala por P_t
        let h_0 = state.hidden.clone().reshape::<3, _>([batch_size, 1, self.d_hidden]);
        let h_seq_initial = p_t * h_0;

        let h_seq = h_seq_parallel + h_seq_initial;

        // 5. Update State (último elemento)
        let last_h = h_seq.clone().slice([0..batch_size, seq_len-1..seq_len, 0..self.d_hidden])
            .reshape([batch_size, self.d_hidden]);
        let final_state = if frozen { state } else { MinGruState::new(last_h) };
        (h_seq, final_state)
    }
}

fn ones_like<B: Backend, const D: usize>(x: &Tensor<B, D>) -> Tensor<B, D> {
    Tensor::ones(x.dims(), &x.device())
}
