#![recursion_limit = "256"]

/*!
Text Generation with xLSTM using Character-Level Tokenization
Updated to use the new xLSTM Rust port.
*/

use burn::optim::decay::WeightDecayConfig;
use burn::{
    module::Module,
    optim::AdamConfig,
    record::{CompactRecorder, Recorder},
    tensor::{activation::softmax, Tensor, backend::{AutodiffBackend, Backend}, Int},
    nn::loss::CrossEntropyLossConfig,
};
use burn::grad_clipping::GradientClippingConfig;
use burn::tensor::TensorData;
use burn_autodiff::Autodiff;
use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;
use burn_ndarray::NdArray;
use tokenizers::decoders::metaspace::Metaspace as MetaspaceDecoder;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::tokenizer::Tokenizer as HFTokenizer;
use tokenizers::models::TrainerWrapper;

use xlstm::{LearningRateConfig, LstmType, XLstm, XLstmConfig, XLstmState};
use xlstm::xlstm_block_stack::XLSTMBlockStackConfig;
use xlstm::blocks::mlstm::block::MLSTMBlockConfig;
use xlstm::blocks::mlstm::layer::MLSTMLayerConfig;
use xlstm::blocks::slstm::block::SLSTMBlockConfig;
use xlstm::blocks::slstm::layer::SLSTMLayerConfig;
use xlstm::components::feedforward::GatedFeedForwardConfig;

type MyBackend = Autodiff<NdArray<f32>>;

/// Tokenizador profesional usando la librería 'tokenizers' de Hugging Face
pub struct Tokenizer {
    tokenizer: HFTokenizer,
}

impl Tokenizer {
    pub fn from_text(text: &str, vocab_size: usize) -> Result<Self, Box<dyn Error>> {
        let model = BPE::builder()
            .byte_fallback(true)
            .build()
            .map_err(|e| format!("Error building BPE: {}", e))?;
            
        let mut tokenizer = HFTokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(Metaspace::new('▁', PrependScheme::Always, true)));
        tokenizer.with_decoder(Some(MetaspaceDecoder::new('▁', PrependScheme::Always, true)));

        let trainer = BpeTrainerBuilder::default()
            .show_progress(true)
            .vocab_size(vocab_size)
            .min_frequency(2)
            .build();

        let mut trainer_wrapper = TrainerWrapper::from(trainer);
        let temp_file = "temp_train.txt";
        fs::write(temp_file, text)?;
        tokenizer.train_from_files(&mut trainer_wrapper, vec![temp_file.to_string()])
            .map_err(|e| format!("Error en entrenamiento: {}", e))?;
        fs::remove_file(temp_file)?;

        Ok(Self { tokenizer })
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        self.tokenizer.save(path, true).map_err(|e| format!("{}", e))?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let mut tokenizer = HFTokenizer::from_file(path).map_err(|e| format!("{}", e))?;
        tokenizer.with_decoder(Some(MetaspaceDecoder::new('▁', PrependScheme::Always, true)));
        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        let encoding = self.tokenizer.encode(text, false).unwrap();
        encoding.get_ids().iter().map(|&id| id as usize).collect()
    }

    pub fn decode(&self, indices: &[usize]) -> String {
        let u32_indices: Vec<u32> = indices.iter().map(|&idx| idx as u32).collect();
        self.tokenizer.decode(&u32_indices, true).unwrap()
    }

    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    pub fn id_to_token(&self, id: usize) -> Option<String> {
        self.tokenizer.id_to_token(id as u32)
    }
}

fn create_batch<B: Backend>(
    tokens: &[usize],
    start_idx: usize,
    batch_size: usize,
    seq_length: usize,
    device: &B::Device,
) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
    let mut x_indices = Vec::with_capacity(batch_size * seq_length);
    let mut y_indices = Vec::with_capacity(batch_size * seq_length);

    for i in 0..batch_size {
        let current_start = start_idx + i;
        for j in 0..seq_length {
            let x_idx = if current_start + j < tokens.len() { tokens[current_start + j] } else { 0 };
            let y_idx = if current_start + j + 1 < tokens.len() { tokens[current_start + j + 1] } else { 0 };
            x_indices.push(x_idx as i64);
            y_indices.push(y_idx as i64);
        }
    }

    let x = Tensor::<B, 2, Int>::from_data(
        TensorData::new(x_indices, [batch_size, seq_length]),
        device,
    );
    let y = Tensor::<B, 2, Int>::from_data(
        TensorData::new(y_indices, [batch_size, seq_length]),
        device,
    );

    (x, y)
}

fn sample_from_logits<B: Backend>(
    logits: Tensor<B, 2>, 
    temperature: f32,
    top_k: usize,
    top_p: f32
) -> usize {
    let probs = softmax(logits, 1);
    let mut probs_vec: Vec<(usize, f32)> = probs.to_data()
        .as_slice::<f32>()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, &x)| (i, x))
        .collect();

    probs_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    let k = top_k.min(probs_vec.len()).max(1);
    let mut filtered_probs = Vec::with_capacity(k);
    let mut cumulative_prob = 0.0;
    for (i, p) in probs_vec.into_iter() {
        filtered_probs.push((i, p));
        cumulative_prob += p;
        if filtered_probs.len() >= k || cumulative_prob >= top_p {
            break;
        }
    }

    let indices: Vec<usize> = filtered_probs.iter().map(|(i, _)| *i).collect();
    let mut weights: Vec<f32> = filtered_probs.iter().map(|(_, p)| *p).collect();

    if temperature <= 1e-6 {
        return indices[0];
    }

    for p in weights.iter_mut() {
        *p = (p.max(1e-10).ln() / temperature).exp();
    }

    let sum: f32 = weights.iter().sum();
    use rand::Rng;
    let mut rng = rand::rng(); 
    let sample: f32 = rng.random::<f32>() * sum; 
    let mut cumulative = 0.0;
    for (i, &p) in weights.iter().enumerate() {
        cumulative += p;
        if sample <= cumulative {
            return indices[i];
        }
    }
    indices[0]
}

fn generate_text<B: Backend>(
    model: &XLstm<B>,
    tokenizer: &Tokenizer,
    seed_text: &str,
    length: usize,
    device: &B::Device,
) -> String {
    let mut generated_ids = tokenizer.encode(seed_text);
    if generated_ids.is_empty() {
        generated_ids.push(0);
    }
    
    let mut current_state = model.empty_state(1, device);

    // Warm up state with seed
    for &id in &generated_ids {
        let input = Tensor::<B, 1, Int>::from_data(TensorData::new(vec![id as i64], [1]), device);
        let (_, next_state) = model.step(input, current_state);
        current_state = next_state;
    }

    let mut result_ids = Vec::new();
    let mut last_id = *generated_ids.last().unwrap();

    for _ in 0..length {
        let input = Tensor::<B, 1, Int>::from_data(TensorData::new(vec![last_id as i64], [1]), device);
        let (logits, next_state) = model.step(input, current_state);
        current_state = next_state;

        let next_token = sample_from_logits(logits, 0.4, 40, 0.9);
        result_ids.push(next_token);
        last_id = next_token;
    }

    tokenizer.decode(&result_ids)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Uso: cargo run --bin msltmchat -- <archivo.txt>");
        std::process::exit(1);
    }

    let text_file = &args[1];
    let tokenizer_path = "tokenizer.json";
    let model_path = "xlstm_chat_model"; 

    let target_vocab_size = 1024;
    let tokenizer = if Path::new(tokenizer_path).exists() {
        Tokenizer::load(tokenizer_path)?
    } else {
        let text = fs::read_to_string(text_file)?;
        let tokenizer = Tokenizer::from_text(&text, target_vocab_size)?;
        tokenizer.save(tokenizer_path)?;
        tokenizer
    };

    let text = fs::read_to_string(text_file)?;
    let tokens = tokenizer.encode(&text);
    let vocab_size = tokenizer.vocab_size();
    let embedding_dim = 128;
    let num_blocks = 4;
    let seq_length = 64;
    let batch_size = 8;
    let num_epochs = 20;

    let device = Default::default();

    let mlstm_cfg = MLSTMBlockConfig {
        mlstm: MLSTMLayerConfig {
            embedding_dim,
            num_heads: 4,
            conv1d_kernel_size: 4,
            qkv_proj_blocksize: 4,
            proj_factor: 2.0,
            bias: true,
            dropout: 0.1,
            context_length: 128,
        },
    };

    let slstm_cfg = SLSTMBlockConfig {
        slstm: SLSTMLayerConfig {
            embedding_dim,
            num_heads: 4,
            conv1d_kernel_size: 4,
            dropout: 0.1,
        },
        feedforward: Some(GatedFeedForwardConfig {
            embedding_dim,
            proj_factor: 1.3,
            bias: true,
            dropout: 0.1,
        }),
    };

    let config = XLstmConfig {
        vocab_size,
        add_embedding_dropout: true,
        block_stack: XLSTMBlockStackConfig {
            embedding_dim,
            num_blocks,
            context_length: 128,
            add_post_blocks_norm: true,
            bias: true,
            dropout: 0.1,
            mlstm_block: Some(mlstm_cfg),
            slstm_block: Some(slstm_cfg),
            slstm_at: vec![1, 3],
        },
    };

    let model_file = format!("{}.mpk", model_path);
    let mut model: xlstm::XLstm<MyBackend> = if Path::new(&model_file).exists() {
        println!("Cargando modelo...");
        let recorder = CompactRecorder::new();
        let record = recorder.load(model_file.into(), &device)?;
        config.init(&device).load_record(record)
    } else {
        println!("Iniciando entrenamiento desde cero...");
        config.init(&device)
    };

    let mut optim = AdamConfig::new()
        .with_weight_decay(Some(WeightDecayConfig::new(1e-5)))
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();

    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let num_sequences = tokens.len().saturating_sub(seq_length + 1);
    let num_batches = num_sequences / (batch_size * seq_length);

    for epoch in 0..num_epochs {
        let mut total_loss = 0.0;
        for batch_idx in 0..num_batches {
            let start_idx = batch_idx * batch_size * seq_length;
            let (input, targets) = create_batch::<MyBackend>(&tokens, start_idx, batch_size, seq_length, &device);

            let logits = model.forward(input);
            let logits_flat = logits.reshape([batch_size * seq_length, vocab_size]);
            let targets_flat = targets.reshape([batch_size * seq_length]);

            let loss = loss_fn.forward(logits_flat, targets_flat);
            total_loss += loss.clone().into_data().as_slice::<f32>().unwrap()[0];

            let grads = loss.backward();
            let lr = 1e-3;
            model = model.optimizer_step(lr, &mut optim, grads);

            if batch_idx % 10 == 0 {
                print!("\rEpoch {}/{} Batch {}/{} Loss: {:.4}", epoch+1, num_epochs, batch_idx, num_batches, total_loss / (batch_idx+1) as f32);
                io::stdout().flush()?;
            }
        }
        println!("\nEpoch {} completa. Generando...", epoch+1);
        let generated = generate_text(&model.valid(), &tokenizer, "El ", 50, &device);
        println!("Generado: {}", generated);

        let recorder = CompactRecorder::new();
        model.clone().save_file(model_path, &recorder)?;
    }

    Ok(())
}
