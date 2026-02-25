use burn::{
    module::Module,
    nn::{
        conv::{Conv2d, Conv2dConfig},
        Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, Initializer,
    },
    tensor::{backend::Backend, Tensor, activation},
    config::Config,
};
use crate::slstm::{SLstm, SLstmstate, SLstmconfig};
use crate::mlstm::{MLstm, MLstmstate, MLstmconfig};
use crate::mingru::{MinGru, MinGruState, MinGruConfig};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BlockType {
    SLSTM,
    MLSTM,
    MINGRU,
}

#[derive(Module, Debug)]
pub enum LSTMVariant<B: Backend> {
    SLSTM(SLstm<B>),
    MLSTM(MLstm<B>),
    MINGRU(MinGru<B>),
}

#[derive(Clone, Debug)]
pub enum LSTMState<B: Backend> {
    SLSTM(alloc::vec::Vec<SLstmstate<B, 2>>),
    MLSTM(alloc::vec::Vec<MLstmstate<B>>),
    MINGRU(alloc::vec::Vec<MinGruState<B>>),
}

impl<B: Backend> LSTMState<B> {
    pub fn detach(self) -> Self {
        match self {
            Self::SLSTM(states) => Self::SLSTM(states.into_iter().map(|s| s.detach()).collect()),
            Self::MLSTM(states) => Self::MLSTM(states.into_iter().map(|s| s.detach()).collect()),
            Self::MINGRU(states) => Self::MINGRU(states.into_iter().map(|s| s.detach()).collect()),
        }
    }
}

#[derive(Config, Debug)]
pub struct XLstmblockConfig {
    pub d_input: usize,
    pub d_hidden: usize,
    pub num_layers: usize,
    pub block_type: BlockType,
    #[config(default = "4")]
    pub num_heads: usize,
    #[config(default = "0.0")]
    pub dropout: f64,
    #[config(default = "false")]
    pub use_conv: bool,
    #[config(default = "4")]
    pub conv_kernel_size: usize,
    #[config(default = "false")]
    pub use_mlp: bool,
    #[config(default = "Initializer::Normal{mean: 0.0, std: 0.02}")]
    pub initializer: Initializer,
}

impl XLstmblockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> XLstmblock<B> {
        let lstm = match self.block_type {
            BlockType::SLSTM => LSTMVariant::SLSTM(
                SLstmconfig::new(self.d_input, self.d_hidden, self.num_layers)
                    .with_initializer(self.initializer.clone())
                    .init(device),
            ),
            BlockType::MLSTM => LSTMVariant::MLSTM(
                MLstmconfig::new(self.d_input, self.d_hidden, self.num_layers)
                    .with_num_heads(self.num_heads)
                    .with_dropout(self.dropout)
                    .with_initializer(self.initializer.clone())
                    .init(device),
            ),
            BlockType::MINGRU => LSTMVariant::MINGRU(
                MinGruConfig::new(self.d_input, self.d_hidden, self.num_layers)
                    .with_initializer(self.initializer.clone())
                    .init(device),
            ),
        };

        let norm = LayerNormConfig::new(self.d_hidden).init(device);
        let proj = LinearConfig::new(self.d_hidden, self.d_hidden)
            .with_initializer(self.initializer.clone())
            .init(device);

        let conv = if self.use_conv {
            Some(
                Conv2dConfig::new([self.d_input, self.d_input], [1, self.conv_kernel_size])
                    .with_padding(burn::nn::PaddingConfig2d::Explicit(0, self.conv_kernel_size - 1))
                    .init(device),
            )
        } else {
            None
        };

        let (mlp_fc1, mlp_fc2) = if self.use_mlp {
            let fc1 = LinearConfig::new(self.d_hidden, self.d_hidden * 4)
                .with_initializer(self.initializer.clone())
                .init(device);
            let fc2 = LinearConfig::new(self.d_hidden * 4, self.d_hidden)
                .with_initializer(self.initializer.clone())
                .init(device);
            (Some(fc1), Some(fc2))
        } else {
            (None, None)
        };

        XLstmblock {
            lstm,
            norm,
            proj,
            mlp_fc1,
            mlp_fc2,
            conv,
            dropout: DropoutConfig::new(self.dropout).init(),
            hidden_size: self.d_hidden,
        }
    }
}

#[derive(Module, Debug)]
pub struct XLstmblock<B: Backend> {
    pub lstm: LSTMVariant<B>,
    pub norm: LayerNorm<B>,
    pub proj: Linear<B>,
    pub mlp_fc1: Option<Linear<B>>,
    pub mlp_fc2: Option<Linear<B>>,
    pub conv: Option<Conv2d<B>>,
    pub dropout: Dropout,
    pub hidden_size: usize,
}

impl<B: Backend> XLstmblock<B> {
    /// Forward pass through xLSTM block
    pub fn forward(
        &self,
        input_seq: Tensor<B, 3>,
        state: Option<LSTMState<B>>,
    ) -> (Tensor<B, 3>, Option<LSTMState<B>>)
    where
        <B as Backend>::FloatElem: num_traits::ToPrimitive + num_traits::FromPrimitive + Copy,
        B: Backend,
    {
        let mut conv_in = input_seq.clone();
        
        if let Some(conv) = &self.conv {
            let [b, s, c] = conv_in.dims();
            let mut x_conv = conv_in.swap_dims(1, 2).reshape([b, c, 1, s]);
            x_conv = conv.forward(x_conv);
            x_conv = x_conv.slice([0..b, 0..c, 0..1, 0..s]);
            x_conv = activation::gelu(x_conv);
            conv_in = x_conv.reshape([b, c, s]).swap_dims(1, 2);
        }

        let (mut out, new_state) = match (&self.lstm, state) {
            (LSTMVariant::SLSTM(lstm), Some(LSTMState::SLSTM(s))) => {
                let (o, ns) = lstm.forward(&conv_in, Some(s));
                (o, Some(LSTMState::SLSTM(ns)))
            }
            (LSTMVariant::SLSTM(lstm), _) => {
                let (o, ns) = lstm.forward(&conv_in, None);
                (o, Some(LSTMState::SLSTM(ns)))
            }
            (LSTMVariant::MLSTM(lstm), Some(LSTMState::MLSTM(s))) => {
                let (o, ns) = lstm.forward(&conv_in, Some(s));
                (o, Some(LSTMState::MLSTM(ns)))
            }
            (LSTMVariant::MLSTM(lstm), _) => {
                let (o, ns) = lstm.forward(&conv_in, None);
                (o, Some(LSTMState::MLSTM(ns)))//forward_ext
            }
            (LSTMVariant::MINGRU(gru), Some(LSTMState::MINGRU(s))) => {
                let (o, ns) = gru.forward(conv_in.clone(), Some(s));
                (o, Some(LSTMState::MINGRU(ns)))
            }
            (LSTMVariant::MINGRU(gru), _) => {
                let (o, ns) = gru.forward(conv_in.clone(), None);
                (o, Some(LSTMState::MINGRU(ns)))
            }
        };

        out = self.norm.forward(out);
        out = activation::gelu(out);
        out = self.proj.forward(out);
        
        if let (Some(fc1), Some(fc2)) = (&self.mlp_fc1, &self.mlp_fc2) {
            let mlp_hid = fc1.forward(out.clone());
            let mlp_hid = activation::gelu(mlp_hid);
            out = out + fc2.forward(mlp_hid);
        }
        
        out = self.dropout.forward(out);
        out = out + input_seq.clone();

        (out, new_state)
    }

    /// Refined forward pass with internal loops (Loop-RNN)
    pub fn forward_refine(
        &self,
        input_seq: Tensor<B, 3>,
        state: Option<LSTMState<B>>,
        n_loops: usize,
    ) -> (Tensor<B, 3>, Option<LSTMState<B>>)
    where
        <B as Backend>::FloatElem: num_traits::ToPrimitive + num_traits::FromPrimitive + Copy,
        B: Backend,
    {
        // Pre-procesamiento: Convolución (igual que en forward)
        let mut conv_in = input_seq.clone();
        if let Some(conv) = &self.conv {
            let [b, s, c] = conv_in.dims();
            let mut x_conv = conv_in.swap_dims(1, 2).reshape([b, c, 1, s]);
            x_conv = conv.forward(x_conv);
            x_conv = x_conv.slice([0..b, 0..c, 0..1, 0..s]);
            x_conv = activation::gelu(x_conv);
            conv_in = x_conv.reshape([b, c, s]).swap_dims(1, 2);
        }

        let mut out = conv_in; // Empezamos el loop con la señal convolucionada
        let mut current_state = state;
        
        for i in 0..n_loops {
            let is_last = i == n_loops - 1;
            let frozen = !is_last;
            
            // Re-aplicar el procesamiento del bloque pero con control de congelación
            let (next_out, next_state) = match (&self.lstm, current_state.clone()) {
                (LSTMVariant::SLSTM(lstm), Some(LSTMState::SLSTM(s))) => {
                    let (o, ns) = lstm.forward_ext(&out, Some(s), frozen);
                    (o, Some(LSTMState::SLSTM(ns)))
                }
                (LSTMVariant::SLSTM(lstm), _) => {
                    let (o, ns) = lstm.forward_ext(&out, None, frozen);
                    (o, Some(LSTMState::SLSTM(ns)))
                }
                (LSTMVariant::MLSTM(lstm), Some(LSTMState::MLSTM(s))) => {
                    let (o, ns) = lstm.forward_ext(&out, Some(s), frozen);
                    (o, Some(LSTMState::MLSTM(ns)))
                }
                (LSTMVariant::MLSTM(lstm), _) => {
                    let (o, ns) = lstm.forward_ext(&out, None, frozen);
                    (o, Some(LSTMState::MLSTM(ns)))
                }
                (LSTMVariant::MINGRU(gru), Some(LSTMState::MINGRU(s))) => {
                    let (o, ns) = gru.forward_ext(out.clone(), Some(s), frozen);
                    (o, Some(LSTMState::MINGRU(ns)))
                }
                (LSTMVariant::MINGRU(gru), _) => {
                    let (o, ns) = gru.forward_ext(out.clone(), None, frozen);
                    (o, Some(LSTMState::MINGRU(ns)))
                }
            };
            
            // Post-procesamiento
            let mut final_out = self.norm.forward(next_out);
            final_out = activation::gelu(final_out);
            final_out = self.proj.forward(final_out);
            
            if let (Some(fc1), Some(fc2)) = (&self.mlp_fc1, &self.mlp_fc2) {
                let mlp_hid = fc1.forward(final_out.clone());
                let mlp_hid = activation::gelu(mlp_hid);
                final_out = final_out + fc2.forward(mlp_hid);
            }
            
            out = self.dropout.forward(final_out); 
            // Residual connection: se añade al final de cada loop sobre el input original del bloque
            out = out + input_seq.clone();
            current_state = next_state;
        }
        
        (out, current_state)
    }

    pub const fn get_type(&self) -> BlockType {
        match &self.lstm {
            LSTMVariant::SLSTM(_) => BlockType::SLSTM,
            LSTMVariant::MLSTM(_) => BlockType::MLSTM,
            LSTMVariant::MINGRU(_) => BlockType::MINGRU,
        }
    }
}
