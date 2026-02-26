use xlstm::blocks::mlstm::layer::{MLSTMLayer, MLSTMLayerConfig};
use burn::tensor::{Tensor, Distribution};
use burn_autodiff::Autodiff;
use burn::backend::ndarray::NdArray;
use burn::prelude::*;

type TestBackend = NdArray<f32>;
type AdBackend = Autodiff<TestBackend>;

fn main() {
    let device = Default::default();
    let batch_size = 1;
    let seq_len = 1; 
    let embedding_dim = 16;
    
    println!("--- TEST mLSTM: AISLAMIENTO TOTAL (SEQ_LEN=1) ---");
    
    let config = MLSTMLayerConfig {
        embedding_dim,
        num_heads: 4,
        conv1d_kernel_size: 4,
        qkv_proj_blocksize: 4,
        proj_factor: 2.0,
        bias: true,
        dropout: 0.0,
        context_length: 128,
    };
    
    let layer_p: MLSTMLayer<AdBackend> = config.init(&device);
    let mut layer_r = config.init(&device);
    
    // Sincronizar pesos (Copiamos de P a R)
    layer_r.proj_up = layer_p.proj_up.clone();
    layer_r.q_proj = layer_p.q_proj.clone();
    layer_r.k_proj = layer_p.k_proj.clone();
    layer_r.v_proj = layer_p.v_proj.clone();
    layer_r.conv1d = layer_p.conv1d.clone();
    layer_r.mlstm_cell = layer_p.mlstm_cell.clone();
    layer_r.learnable_skip = layer_p.learnable_skip.clone();
    layer_r.proj_down = layer_p.proj_down.clone();

    let input_raw = Tensor::<TestBackend, 3>::random([batch_size, seq_len, embedding_dim], Distribution::Normal(0.0, 1.0), &device);
    
    // --- PASADA 1: PARALELA ---
    let input_p = Tensor::<AdBackend, 3>::from_inner(input_raw.clone()).require_grad();
    let output_p = layer_p.forward(input_p.clone());
    let grads_p = output_p.clone().sum().backward();
    
    let g_w_p = layer_p.proj_up.weight.grad(&grads_p).unwrap();
    let g_in_p = input_p.grad(&grads_p).unwrap();

    // --- PASADA 2: RECURRENTE ---
    let input_r = Tensor::<AdBackend, 3>::from_inner(input_raw).require_grad();
    let state = layer_r.empty_state(batch_size, &device);
    let x_t = input_r.clone().slice([0..batch_size, 0..1, 0..embedding_dim]).reshape([batch_size, embedding_dim]);
    
    let (y_t, _) = layer_r.step(x_t, state);
    
    let grads_r = y_t.clone().sum().backward(); // .clone() aquí para poder comparar y_t después
    let g_w_r = layer_r.proj_up.weight.grad(&grads_r).unwrap();
    let g_in_r = input_r.grad(&grads_r).unwrap();

    // --- COMPARAR ---
    let val_diff = (output_p.reshape([batch_size, embedding_dim]) - y_t).abs().mean().into_scalar();
    let grad_w_diff = (g_w_p - g_w_r).abs().mean().into_scalar();
    let grad_in_diff = (g_in_p - g_in_r).abs().mean().into_scalar();

    println!("Diferencia VALOR:       {:.10}", val_diff);
    println!("Diferencia GRAD PESO:   {:.10}", grad_w_diff);
    println!("Diferencia GRAD INPUT:  {:.10}", grad_in_diff);

    if grad_w_diff < 1e-4 {
        println!("\n✅ GRADIENTES COINCIDEN EN SEQ_LEN=1");
    } else {
        println!("\n❌ ERROR ESTRUCTURAL DETECTADO");
        std::process::exit(1);
    }
}
