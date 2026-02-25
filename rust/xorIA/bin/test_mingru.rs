use xlstm::{MinGru, MinGruConfig};
use burn::tensor::{Tensor, Distribution};
use burn::tensor::backend::Backend;
use burn::backend::Autodiff;
use burn::optim::{AdamConfig, Optimizer};
use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use rand::Rng;

type TestBackend = burn_ndarray::NdArray<f32>;

#[derive(Module, Debug)]
pub struct ParityModel<B: Backend> {
    mingru: MinGru<B>,
    head: Linear<B>,
}

fn run_grad() {
    let device = Default::default();
    let batch_size = 1;
    let seq_len = 100; 
    let input_size = 8;
    let hidden_size = 64;
    
    type AdBackend = Autodiff<TestBackend>;
    
    let config = MinGruConfig::new(input_size, hidden_size, 1);
    let mingru: MinGru<AdBackend> = config.init(&device);
    
    let x = Tensor::<AdBackend, 1>::random(
        [batch_size * seq_len * input_size], 
        Distribution::Normal(0.0, 1.0), 
        &device
    ).reshape([batch_size, seq_len, input_size]).require_grad();
    
    let (out, _) = mingru.forward(x.clone(), None);
    let h_last = out.slice([0..batch_size, seq_len-1..seq_len, 0..hidden_size]).sum();
    
    let grads = h_last.backward();
    let x_grad = x.grad(&grads).expect("Debe existir gradiente");
    
    let grad_first = x_grad.clone().slice([0..batch_size, 0..1, 0..input_size]).abs().mean().into_scalar();
    let grad_last = x_grad.clone().slice([0..batch_size, seq_len-1..seq_len, 0..input_size]).abs().mean().into_scalar();
    
    println!("Análisis de Gradiente (S={}):", seq_len);
    println!("  - t=0    (Long-term) : {:.10}", grad_first);
    println!("  - t=Last (Short-term): {:.10}", grad_last);
    
    if grad_first > 1e-10 {
        println!("✅ El gradiente llega al inicio!");
    } else {
        println!("⚠️ Desvanecimiento detectado.");
    }
}

fn run_stress_learning() {
    let device = Default::default();
    let batch_size = 64;
    let seq_len = 16; 
    let input_size = 1;
    let hidden_size = 64;
    
    type AdBackend = Autodiff<TestBackend>;
    
    let mut rng = rand::rng();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for _ in 0..batch_size {
        let mut count = 0;
        for _ in 0..seq_len {
            let val = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
            xs.push(val);
            if val > 0.0 { count += 1; }
        }
        ys.push(if count % 2 == 0 { 0i64 } else { 1i64 });
    }
    
    let x = Tensor::<AdBackend, 1>::from_floats(xs.as_slice(), &device).reshape([batch_size, seq_len, 1]);
    let y = Tensor::<AdBackend, 1, burn::tensor::Int>::from_ints(ys.as_slice(), &device);
    
    let config = MinGruConfig::new(input_size, hidden_size, 1);
    let mingru = config.init(&device);
    let head = LinearConfig::new(hidden_size, 2).init(&device);
    
    let mut model = ParityModel { mingru, head };
    let mut optim = AdamConfig::new().init();
    
    let loss_fn = burn::nn::loss::CrossEntropyLossConfig::new().init(&device);
    
    println!("\nEntrenando MinGRU (Task: Parity-16)...");
    let initial_loss: f32;
    
    let (out, _) = model.mingru.forward(x.clone(), None);
    let last_h = out.slice([0..batch_size, seq_len-1..seq_len, 0..hidden_size]).reshape([batch_size, hidden_size]);
    let logits = model.head.forward(last_h);
    initial_loss = loss_fn.forward(logits, y.clone()).into_data().as_slice::<f32>().unwrap()[0];

    for epoch in 1..=50 {
        let (out, _) = model.mingru.forward(x.clone(), None);
        let last_h = out.slice([0..batch_size, seq_len-1..seq_len, 0..hidden_size]).reshape([batch_size, hidden_size]);
        
        let logits = model.head.forward(last_h);
        let loss = loss_fn.forward(logits, y.clone());
        let loss_val = loss.clone().into_data().as_slice::<f32>().unwrap()[0];
        
        let grads = loss.backward();
        let grads_params = burn::optim::GradientsParams::from_grads(grads, &model);
        model = optim.step(5e-3, model, grads_params);
        
        if epoch % 10 == 0 || epoch == 1 {
            println!("  Época {:3}: Loss = {:.4}", epoch, loss_val);
        }
        
        if epoch == 50 {
            if loss_val < initial_loss * 0.95 {
                println!("✅ MinGRU está aprendiendo.");
            } else {
                println!("⚠️ Mejora de pérdida baja.");
            }
        }
    }
}

fn main() {
    println!("--- MinGRU Stress Test (v3) ---");
    run_grad();
    run_stress_learning();
}
