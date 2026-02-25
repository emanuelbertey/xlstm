/*
# xLSTM: Extended Long Short-Term Memory

This library implements the xLSTM model as described in the paper:
"xLSTM: Extended Long Short-Term Memory" by Beck et al. (2024).

The xLSTM combines sLSTM (Scalar LSTM) and mLSTM (Matrix LSTM) in a novel
architecture to achieve state-of-the-art performance on various sequence
modeling tasks.

*/

extern crate alloc;

mod block;
mod mlstm;
mod model;
mod slstm;
mod mingru;

pub use block::{BlockType, XLstmblock, XLstmblockConfig, LSTMVariant, LSTMState};
pub use mlstm::{MLstm, MLstmcell, MLstmconfig, MLstmstate};
pub use model::{LearningRateConfig, LstmType, XLstm, XLstmconfig};
pub use slstm::{SLstm, SLstmcell, SLstmconfig, SLstmstate};
pub use mingru::{MinGru, MinGruConfig, MinGruLayer, MinGruState};

pub const VERSION: &str = "0.1.0";
