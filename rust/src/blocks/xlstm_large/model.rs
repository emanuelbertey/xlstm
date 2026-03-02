use burn::prelude::*;
use burn::nn;
use burn::module::Module;

use super::config::XLSTMLargeConfig;
use super::components::{RMSNorm, soft_cap};
use super::layer::mLSTMLayer;

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    pub proj_up_gate: nn::Linear<B>,
    pub proj_up: nn::Linear<B>,
    pub proj_down: nn::Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn init(config: &XLSTMLargeConfig, device: &B::Device) -> Self {
        let up_dim = config.up_proj_dim();
        Self {
            proj_up_gate: nn::LinearConfig::new(config.embedding_dim, up_dim).with_bias(config.use_bias).init(device),
            proj_up: nn::LinearConfig::new(config.embedding_dim, up_dim).with_bias(config.use_bias).init(device),
            proj_down: nn::LinearConfig::new(up_dim, config.embedding_dim).with_bias(config.use_bias).init(device),
        }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let gate = burn::tensor::activation::silu(self.proj_up_gate.forward(x.clone()));
        let up = self.proj_up.forward(x);
        self.proj_down.forward(gate * up)
    }
}

#[derive(Module, Debug)]
pub struct mLSTMBlock<B: Backend> {
    pub norm_mlstm: RMSNorm<B>,
    pub mlstm_layer: mLSTMLayer<B>,
    pub norm_ffn: RMSNorm<B>,
    pub ffn: FeedForward<B>,
}

impl<B: Backend> mLSTMBlock<B> {
    pub fn init(config: &XLSTMLargeConfig, device: &B::Device) -> Self {
        Self {
            norm_mlstm: RMSNorm::init(config.embedding_dim, true, config.use_bias, config.norm_eps, device),
            mlstm_layer: mLSTMLayer::init(config, device),
            norm_ffn: RMSNorm::init(config.embedding_dim, true, config.use_bias, config.norm_eps, device),
            ffn: FeedForward::init(config, device),
        }
    }

    pub fn forward(
        &self, 
        x: Tensor<B, 3>, 
        state: Option<(Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>)>
    ) -> (Tensor<B, 3>, Option<(Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>)>) {
        let x_norm = self.norm_mlstm.forward(x.clone());
        let (x_mlstm, next_state) = self.mlstm_layer.forward(x_norm, state);
        let x = x + x_mlstm;

        let x_ffn_norm = self.norm_ffn.forward(x.clone());
        let x_ffn = self.ffn.forward(x_ffn_norm);
        (x + x_ffn, next_state)
    }
}

#[derive(Clone, Debug)]
pub struct XLSTMLargeState<B: Backend> {
    pub block_states: Vec<(Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>)>,
}

#[derive(Module, Debug)]
pub struct XLSTMLarge<B: Backend> {
    pub embedding: nn::Embedding<B>,
    pub blocks: Vec<mLSTMBlock<B>>,
    pub out_norm: Option<RMSNorm<B>>,
    pub lm_head: nn::Linear<B>,
    pub output_logit_soft_cap: f64,
}

impl<B: Backend> XLSTMLarge<B> {
    pub fn init(config: &XLSTMLargeConfig, device: &B::Device) -> Self {
        let embedding = nn::EmbeddingConfig::new(config.vocab_size, config.embedding_dim).init(device);
        let blocks = (0..config.num_blocks).map(|_| mLSTMBlock::init(config, device)).collect();
        let out_norm = if config.add_out_norm {
            Some(RMSNorm::init(config.embedding_dim, true, config.use_bias, config.norm_eps, device))
        } else {
            None
        };
        let lm_head = nn::LinearConfig::new(config.embedding_dim, config.vocab_size).with_bias(false).init(device);
        
        Self {
            embedding,
            blocks,
            out_norm,
            lm_head,
            output_logit_soft_cap: config.output_logit_soft_cap,
        }
    }

    pub fn forward(
        &self, 
        x: Tensor<B, 2, Int>, 
        state: Option<XLSTMLargeState<B>>
    ) -> (Tensor<B, 3>, Option<XLSTMLargeState<B>>) {
        let [b, s] = x.dims();
        let mut x = self.embedding.forward(x);
        
        match state {
            Some(st) => {
                let mut next_block_states = Vec::with_capacity(self.blocks.len());
                for (block, b_state) in self.blocks.iter().zip(st.block_states.into_iter()) {
                    let (out, next_b_state) = block.forward(x, Some(b_state));
                    x = out;
                    if let Some(nbs) = next_b_state {
                        next_block_states.push(nbs);
                    }
                }
                
                if let Some(norm) = &self.out_norm {
                    x = norm.forward(x);
                }
                let mut logits = self.lm_head.forward(x);
                if self.output_logit_soft_cap > 0.0 {
                    logits = soft_cap(logits, self.output_logit_soft_cap);
                }
                (logits, Some(XLSTMLargeState { block_states: next_block_states }))
            }
            None => {
                for block in &self.blocks {
                    let (out, _) = block.forward(x, None);
                    x = out;
                }
                if let Some(norm) = &self.out_norm {
                    x = norm.forward(x);
                }
                let mut logits = self.lm_head.forward(x);
                if self.output_logit_soft_cap > 0.0 {
                    logits = soft_cap(logits, self.output_logit_soft_cap);
                }
                (logits, None)
            }
        }
    }

    pub fn empty_state(&self, batch_size: usize, device: &B::Device) -> XLSTMLargeState<B> {
        let mut block_states = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let dh_v = block.mlstm_layer.v_dim / block.mlstm_layer.num_heads;
            let dh_qk = block.mlstm_layer.qk_dim / block.mlstm_layer.num_heads;
            block_states.push((
                Tensor::zeros([batch_size, block.mlstm_layer.num_heads, dh_qk, dh_v], device),
                Tensor::zeros([batch_size, block.mlstm_layer.num_heads, dh_qk, 1], device),
                Tensor::zeros([batch_size, block.mlstm_layer.num_heads, 1, 1], device),
            ));
        }
        XLSTMLargeState { block_states }
    }
}


