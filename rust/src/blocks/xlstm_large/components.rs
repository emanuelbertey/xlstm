use burn::prelude::*;
use burn::module::Module;

pub fn soft_cap<B: Backend, const D: usize>(values: Tensor<B, D>, cap_value: f64) -> Tensor<B, D> {
    values.div_scalar(cap_value).tanh().mul_scalar(cap_value)
}

#[derive(Module, Debug)]
pub struct RMSNorm<B: Backend> {
    pub weight: Option<burn::module::Param<Tensor<B, 1>>>,
    pub bias: Option<burn::module::Param<Tensor<B, 1>>>,
    pub eps: f64,
}

impl<B: Backend> RMSNorm<B> {
    pub fn init(dim: usize, use_weight: bool, use_bias: bool, eps: f64, device: &B::Device) -> Self {
        let weight = if use_weight {
            Some(burn::module::Param::from_tensor(Tensor::ones([dim], device)))
        } else {
            None
        };
        let bias = if use_bias {
            Some(burn::module::Param::from_tensor(Tensor::zeros([dim], device)))
        } else {
            None
        };
        Self { weight, bias, eps }
    }

    pub fn forward<const D: usize>(&self, x: Tensor<B, D>) -> Tensor<B, D> {
        let rms = x.clone().powf_scalar(2.0).mean_dim(D - 1).add_scalar(self.eps).sqrt();
        let norm = x.div(rms);
        
        let mut out = norm;
        if let Some(w) = &self.weight {
            let w_val = w.val();
            let mut shape = [1; D];
            shape[D - 1] = w_val.dims()[0];
            out = out.mul(w_val.reshape(shape));
        }
        if let Some(b) = &self.bias {
            let b_val = b.val();
            let mut shape = [1; D];
            shape[D - 1] = b_val.dims()[0];
            out = out.add(b_val.reshape(shape));
        }
        out
    }
}
