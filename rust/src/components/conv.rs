use burn::prelude::*;
use burn::nn;
use burn::module::Module;
use burn::config::Config;

#[derive(Config)]
pub struct CausalConv1dConfig {
    pub feature_dim: usize,
    pub kernel_size: usize,
    #[config(default = true)]
    pub bias: bool,
}

#[derive(Module, Debug)]
pub struct CausalConv1d<B: Backend> {
    pub conv: nn::conv::Conv1d<B>,
    pub kernel_size: usize,
    pub feature_dim: usize,
}

impl CausalConv1dConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> CausalConv1d<B> {
        let conv = nn::conv::Conv1dConfig::new(self.feature_dim, self.feature_dim, self.kernel_size)
            .with_bias(self.bias)
            .with_groups(self.feature_dim) 
            .init(device);

        CausalConv1d {
            conv,
            kernel_size: self.kernel_size,
            feature_dim: self.feature_dim,
        }
    }
}

impl<B: Backend> CausalConv1d<B> {
    pub fn empty_state(&self, batch_size: usize, device: &B::Device) -> Tensor<B, 3> {
        Tensor::zeros([batch_size, self.kernel_size, self.feature_dim], device)
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [b, s, f] = x.dims();
        let device = x.device();
        let padding = self.kernel_size - 1;
        
        let zeros = Tensor::zeros([b, f, padding], &device);
        let x_padded = Tensor::cat(vec![zeros, x.swap_dims(1, 2)], 2);
        
        let y = self.conv.forward(x_padded);
        y.swap_dims(1, 2).narrow(1, 0, s)
    }

    pub fn step(&self, x: Tensor<B, 2>, state: Tensor<B, 3>) -> (Tensor<B, 2>, Tensor<B, 3>) {
        let [batch_size, k, f] = state.dims();
        let new_state = Tensor::cat(vec![state.narrow(1, 1, k - 1), x.unsqueeze_dim(1)], 1);
        
        let weight = self.conv.weight.val().reshape([f, k]).unsqueeze_dim::<3>(0);
        
        let new_state_swapped = new_state.clone().swap_dims(1, 2); 
        let mut y = (new_state_swapped * weight).sum_dim(2).reshape([batch_size, f]);
        
        if let Some(bias) = &self.conv.bias {
            y = y + bias.val().reshape([1, f]);
        }
        (y, new_state)
    }
}
