#![recursion_limit = "256"]

/*!
xLSTM Pro Trainer - High Performance Data Streaming
    
This tool is designed to train xLSTM models on large datasets (directories, 
multiple file formats, and gigabyte-sized files) without exhausting RAM.
*/

use burn::grad_clipping::GradientClippingConfig;
use burn::optim::decay::WeightDecayConfig;
use burn::{
    module::AutodiffModule,
    module::Module,
    optim::AdamConfig,
    record::{CompactRecorder, Recorder},
    tensor::{activation::softmax, Tensor, backend::{AutodiffBackend, Backend}},
    nn::loss::CrossEntropyLossConfig,
};
use burn::tensor::TensorData;
use burn_autodiff::Autodiff;
use std::error::Error;
use std::fs;
use std::io::{self, Write, Read, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;
use burn_ndarray::NdArray;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::decoders::byte_level::ByteLevel as ByteLevelDecoder;
use tokenizers::models::bpe::{BpeTrainerBuilder, BPE};
use tokenizers::tokenizer::Tokenizer as HFTokenizer;
use tokenizers::models::TrainerWrapper;
use xlstm::{LearningRateConfig, LstmType, XLstm, XLstmconfig, BlockType};
use std::collections::HashSet;

type MyBackend = Autodiff<NdArray<f32>>;

/// Professional BPE Tokenizer
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
        tokenizer.with_pre_tokenizer(Some(ByteLevel::default()));
        tokenizer.with_decoder(Some(ByteLevelDecoder::default()));

        // Agregar token especial <|endoftext|>
        let special_token = "<|endoftext|>";
        tokenizer.add_special_tokens(&[tokenizers::AddedToken::from(special_token, true)]);

        let trainer = BpeTrainerBuilder::default()
            .show_progress(true)
            .vocab_size(vocab_size)
            .min_frequency(0)
            .initial_alphabet(ByteLevel::alphabet().into_iter().collect::<HashSet<_>>())
            .special_tokens(vec![tokenizers::AddedToken::from(special_token, true)])
            .build();

        let mut trainer_wrapper = TrainerWrapper::from(trainer);
        let temp_file = "temp_train_pro.txt";
        fs::write(temp_file, text)?;
        tokenizer.train_from_files(&mut trainer_wrapper, vec![temp_file.to_string()])
            .map_err(|e| format!("Error in training: {}", e))?;
        fs::remove_file(temp_file)?;

        Ok(Self { tokenizer })
    }

    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        self.tokenizer.save(path, true).map_err(|e| format!("Error saving: {}", e))?;
        Ok(())
    }

    pub fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let mut tokenizer = HFTokenizer::from_file(path).map_err(|e| format!("Error loading: {}", e))?;
        tokenizer.with_decoder(Some(ByteLevelDecoder::default()));
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

    pub fn token_to_id(&self, token: &str) -> Option<usize> {
        self.tokenizer.token_to_id(token).map(|id| id as usize)
    }
}

/// Recursively find files with specific extensions
fn find_files(dir: &Path, extensions: &[&str]) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(find_files(&path, extensions));
            } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if extensions.contains(&ext) {
                    files.push(path);
                }
            }
        }
    }
    files
}

/// Iterator for large files that yields fragments (buffers) to save RAM
struct FileFragmentIterator {
    reader: BufReader<fs::File>,
    buffer_size: usize,
    finished: bool,
}

impl FileFragmentIterator {
    fn new(path: &Path, buffer_size_mb: usize) -> io::Result<Self> {
        let file = fs::File::open(path)?;
        Ok(Self {
            reader: BufReader::new(file),
            buffer_size: buffer_size_mb * 1024 * 1024,
            finished: false,
        })
    }
}

impl Iterator for FileFragmentIterator {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished { return None; }

        let mut buffer = vec![0u8; self.buffer_size];
        let mut total_read = 0;

        while total_read < self.buffer_size {
            match self.reader.read(&mut buffer[total_read..]) {
                Ok(0) => { self.finished = true; break; }
                Ok(n) => total_read += n,
                Err(ref e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(_) => { self.finished = true; break; }
            }
        }

        if total_read == 0 { return None; }
        buffer.truncate(total_read);

        // Ensure we don't split a UTF-8 character
        while !buffer.is_empty() && String::from_utf8(buffer.clone()).is_err() {
            buffer.pop();
        }

        if buffer.is_empty() { return None; }
        String::from_utf8(buffer).ok()
    }
}

fn create_batch<B: AutodiffBackend>(
    tokens: &[usize],
    start_idx: usize,
    batch_size: usize,
    seq_length: usize,
    stride: usize,
    vocab_size: usize,
    device: &B::Device,
) -> (Tensor<B, 3>, Tensor<B, 2, burn::tensor::Int>) {
    let mut x_indices = Vec::with_capacity(batch_size * seq_length);
    let mut y_indices = Vec::with_capacity(batch_size * seq_length);

    for i in 0..batch_size {
        let current_start = start_idx + i * stride;
        
        // RELLENO (PADDING): Si no hay suficientes tokens, rellenamos con ceros (o el último token)
        for j in 0..seq_length {
            let idx_x = if current_start + j < tokens.len() { tokens[current_start + j] } else { 0 };
            let idx_y = if current_start + j + 1 < tokens.len() { tokens[current_start + j + 1] } else { 0 };
            
            x_indices.push(idx_x as i64);
            y_indices.push(idx_y as i64);
        }
    }

    let eye = Tensor::<B::InnerBackend, 2>::eye(vocab_size, device);
    let indices_inner = Tensor::<B::InnerBackend, 1, burn::tensor::Int>::from_data(
        TensorData::new(x_indices, [batch_size * seq_length]),
        device,
    );

    let x = Tensor::<B, 3>::from_inner(
        eye.select(0, indices_inner).reshape([batch_size, seq_length, vocab_size])
    );
    
    let y = Tensor::<B, 2, burn::tensor::Int>::from_data(
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
) -> usize
where
    <B as Backend>::FloatElem: num_traits::ToPrimitive,
{
    let probs = softmax(logits, 1);
    let mut probs_vec: Vec<(usize, f32)> = probs.to_data()
        .as_slice::<<B as Backend>::FloatElem>()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, &x)| (i, num_traits::ToPrimitive::to_f32(&x).unwrap_or(0.0)))
        .collect();

    probs_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    // Check for NaNs
    if probs_vec.iter().any(|(_, p)| p.is_nan()) {
        return 0; // Fallback safety
    }

    let k = top_k.min(probs_vec.len()).max(1);
    let mut filtered_probs: Vec<(usize, f32)> = Vec::with_capacity(k);
    
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

    if temperature <= 1e-6 { return indices[0]; }

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
        if sample <= cumulative { return indices[i]; }
    }

    indices[0]
}

fn generate_text<B: Backend>(
    model: &XLstm<B>,
    tokenizer: &Tokenizer,
    seed_text: &str,
    length: usize,
    vocab_size: usize,
    device: &B::Device,
) -> String
where
    <B as Backend>::FloatElem: num_traits::ToPrimitive + num_traits::FromPrimitive,
{
    let mut generated_ids = tokenizer.encode(seed_text);
    let seed_tokens = generated_ids.clone();
    if seed_tokens.is_empty() { return seed_text.to_string(); }

    let eye = Tensor::<B, 2>::eye(vocab_size, device);
    let mut current_state = None; 
    let mut current_tokens = seed_tokens.clone();

    let eof_id = tokenizer.token_to_id("<|endoftext|>");

    for i in 0..length {
        let tokens_to_process = if i == 0 { current_tokens.clone() } else { vec![*current_tokens.last().unwrap()] };
        let seq_len = tokens_to_process.len();
        let indices = Tensor::<B, 1, burn::tensor::Int>::from_data(
            TensorData::new(tokens_to_process.iter().map(|&t| t as i64).collect(), [seq_len]),
            device,
        );

        let input = eye.clone().select(0, indices).reshape([1, seq_len, vocab_size]);
        
        //        let (output, next_state) = model.forward(input, current_state);// Usamos la nueva función independiente para refinamiento en generación
        let (output, next_state) = model.forward(input, current_state);
        current_state = Some(next_state);

        let dims = output.dims();
        let last_logits = output.slice([0..1, (dims[1] - 1)..dims[1], 0..dims[2]]).reshape([1, dims[2]]);
        
        let next_token = sample_from_logits(last_logits, 0.7, 40, 0.9);

        if Some(next_token) == eof_id {
             println!("======= <|endoftext|> ======\n");
             println!("Tokens generados: {}\n", i);
            break;
        }

        current_tokens.push(next_token);
        generated_ids.push(next_token);
    }

    tokenizer.decode(&generated_ids[seed_tokens.len()..])
}


fn main() -> Result<(), Box<dyn Error>> {
    println!("xLSTM Pro Trainer - Automatización de Directorios");
    println!("================================================\n");

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Uso: cargo run --bin pro_trainer -- <archivo_o_carpeta>");
        std::process::exit(1);
    }

    let input_path = Path::new(&args[1]);
    let tokenizer_path = "tokenizer_pro.json";
    let model_path = "xlstm_pro_model";
    let extensions = ["txt", "gd", /*"html", "cpp", "c", "py", "h","md", */ "gdshader"];

    let all_files = if input_path.is_dir() {
        println!("Explorando directorio: {:?}", input_path);
        find_files(input_path, &extensions)
    } else {
        vec![input_path.to_path_buf()]
    };

    if all_files.is_empty() {
        return Err("No se encontraron archivos válidos (.txt, .gd, .html)".into());
    }
    println!("Archivos encontrados: {}", all_files.len());

    let target_vocab_size = 4096;
    let tokenizer_file = Path::new(tokenizer_path);
    
    let mut needs_training = true;
    let mut tokenizer = None;

    if tokenizer_file.exists() {
        if let Ok(loaded_t) = Tokenizer::load(tokenizer_path) {
            let current_vocab = loaded_t.vocab_size();
            if current_vocab >= target_vocab_size {
                println!("✅ Tokenizador cargado con vocabulario suficiente: {}", current_vocab);
                tokenizer = Some(loaded_t);
                needs_training = false;
            } else {
                println!("⚠️ Tokenizador existente insuficiente ({} < {}). Re-entrenando...", current_vocab, target_vocab_size);
            }
        }
    }

    let tokenizer = if needs_training {
        println!("🚀 Entrenando nuevo tokenizador con {} archivos...", all_files.len().min(100));
        let mut sample_text = String::new();
        let special_token = "<|endoftext|>";
        sample_text.push_str(special_token);
        
        for file_path in all_files.iter().take(100) {
            if let Ok(mut file) = fs::File::open(file_path) {
                let mut content = String::new();
                let mut reader = BufReader::new(file).take(500_000); // 500KB por archivo
                if reader.read_to_string(&mut content).is_ok() {
                    sample_text.push_str(&content);
                    sample_text.push_str(special_token);
                }
            }
        }
        
        if sample_text.trim().len() < 1000 {
            return Err("Error: Muestras insuficientes para alcanzar 4096 tokens.".into());
        }
        
        println!("  -> Bytes totales para entrenamiento: {}", sample_text.len());
        let t = Tokenizer::from_text(&sample_text, target_vocab_size)?;
        t.save(tokenizer_path)?;
        println!("✨ Tokenizador guardado (Vocab: {})", t.vocab_size());
        t
    } else {
        tokenizer.unwrap()
    };

    let vocab_size = tokenizer.vocab_size();
    println!("Resumen: Vocabulario={}, Archivos={}", vocab_size, all_files.len());
    
    let hidden_size = 256;
    let num_layers = 1;
    let seq_length = 256;
    let batch_size = 16;
    let stride = 256;
    let num_epochs = 50;

    let device = Default::default();
    let dropout = 0.1;
    let num_heads = 8;

    // DEFINICIÓN DEL ORDEN DE BLOQUES
    let block_layout = vec![
      // BlockType::MLSTM, 
        BlockType::MLSTM, 
        BlockType::MLSTM, 
        BlockType::MLSTM, 
        BlockType::MLSTM,
        BlockType::MLSTM, 
    ];
    let num_blocks = block_layout.len();

    let lr_config = LearningRateConfig::per_block_type(
        1e-4, // sLSTM
        8e-7, // mLSTM
        1e-3, // minGRU
        1e-4, // Others
    );

    let config = XLstmconfig::new(vocab_size, hidden_size, num_layers, num_blocks, vocab_size)
        .with_dropout(dropout)
        .with_num_heads(num_heads)
        .with_lstm_type(LstmType::Custom(block_layout.clone()))
        .with_use_projection(true);   

    println!("Configuración: {} bloques (H={})", num_blocks, hidden_size);
    println!("Layout: {:?}", block_layout);

    let model_file = format!("{}.mpk", model_path);
    let mut model = if Path::new(&model_file).exists() {
        println!("Cargando modelo previo '{}'...", model_file);
        let recorder = CompactRecorder::new();
        let record = recorder.load(model_file.into(), &device)
            .map_err(|e| format!("Error al cargar modelo: {}", e))?;
        config.init::<MyBackend>(&device).load_record(record)
    } else {
        println!("No se encontró modelo previo. Iniciando desde cero.");
        config.init::<MyBackend>(&device)
    };

    println!("\n¿Qué deseas hacer?");
    println!("1. Entrenar (continuar o empezar)");
    println!("2. Inferir (generar texto interactivo)");
    print!("Selección > ");
    io::stdout().flush()?;
    let mut choice = String::new();
    io::stdin().read_line(&mut choice)?;

    let mut skip_batches = 0;
    if choice.trim() == "1" {
        print!("Saltar hasta el batch (0 para empezar desde el inicio) > ");
        io::stdout().flush()?;
        let mut skip_input = String::new();
        io::stdin().read_line(&mut skip_input)?;
        skip_batches = skip_input.trim().parse::<usize>().unwrap_or(0);
    } else if choice.trim() == "2" {
        println!("\nModo Inferencia Chat");
        println!("====================");
        println!("Comandos:");
        println!("  - Escribe tu semilla para generar");
        println!("  - 'len <n>' para cambiar longitud (ej: len 500)");
        println!("  - 'salir' para finalizar\n");

        let mut current_len = 200;
        loop {
            print!("Semilla (len: {}) > ", current_len);
            io::stdout().flush()?;
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let input = input.trim();
            
            if input.is_empty() { continue; }
            if input == "salir" { break; }

            if input.starts_with("len ") {
                if let Ok(new_len) = input[4..].trim().parse::<usize>() {
                    current_len = new_len;
                    println!("  -> Longitud cambiada a: {}", current_len);
                    continue;
                }
            }

            println!("Generando...");
            let start_gen = Instant::now();
            let generated = generate_text(&model.valid(), &tokenizer, input, current_len, vocab_size, &device);
            let elapsed_gen = start_gen.elapsed().as_secs_f32();
            
            if generated.is_empty() {
                println!("  ⚠️ El modelo generó texto vacío ({:.3}s).", elapsed_gen);
            } else {
                println!("  ✨ Generación completada en {:.3}s", elapsed_gen);
            }
            // Limpieza solo visual para el chat
            let clean_text = generated
                .replace('▁', " ")
                .replace('Ġ', " ")
                .replace('Ċ', "\n")
                .replace('↲', "\n")
                .replace('␣', " ")
                .replace('\t', "    "); // Convertir tabs a espacios para la consola
            
            // Si el texto parece vacío pero hay tokens, es que son solo espacios/control
            if clean_text.trim().is_empty() && !generated.is_empty() {
                println!("│ [Contenido: {} espacios/tabs]", generated.len());
            }
            
            // The original `let clean_text = generated;` was redundant after the new `clean_text` definition.
            // It's removed to avoid shadowing and use the newly cleaned text.

            println!("\n╭──────────────────────────────────────────────────────────╮");
            println!("│ TEXTO GENERADO:");
            println!("├──────────────────────────────────────────────────────────┤");
            
            for line in clean_text.lines() {
                let mut current_line = line.to_string();
                if current_line.is_empty() {
                    println!("│");
                    continue;
                }
                while current_line.len() > 58 {
                    let chunk: String = current_line.chars().take(58).collect();
                    println!("│ {}", chunk);
                    current_line = current_line.chars().skip(58).collect();
                }
                println!("│ {}", current_line);
            }
            println!("╰──────────────────────────────────────────────────────────╯\n");
        }
        return Ok(());
    }

    let mut optim = AdamConfig::new()
        .with_beta_1(0.9)
        .with_beta_2(0.999)
        .with_epsilon(1e-8)
        .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
        .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
        .init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);

    println!("\nIniciando entrenamiento por fragmentos...");

    let mut global_batches_skipped = 0;

    for epoch in 0..num_epochs {
        println!("\n--- Época {}/{} ---", epoch + 1, num_epochs);
        let mut total_loss = 0.0f32;
        let mut processed_in_epoch = 0;

        let mut last_save_time = Instant::now();
        let mut token_buffer = Vec::new();
        let tokens_per_batch = batch_size * stride;
        let tokens_needed_for_batch = tokens_per_batch + seq_length; 

        for (f_idx, file_path) in all_files.iter().enumerate() {
            // Solo imprimimos si el archivo es relevante o para dar feedback de progreso
            if f_idx % 10 == 0 || all_files.len() < 100 {
                print!("\r  [{}/{}] Analizando: {:<40}", f_idx + 1, all_files.len(), file_path.file_name().unwrap_or_default().to_string_lossy());
                io::stdout().flush().unwrap();
            }

            let fragments = FileFragmentIterator::new(file_path, 4).map_err(|e| e.to_string())?;
            
            for (_fr_idx, fragment) in fragments.enumerate() {
                let new_tokens = tokenizer.encode(&fragment);
                if new_tokens.is_empty() && !fragment.trim().is_empty() {
                    eprintln!("\n⚠️ ADVERTENCIA: Fragmento produjo 0 tokens. ¿Tokenizador roto?");
                }
                token_buffer.extend(new_tokens);

                // Procesamos mientras el búfer tenga suficientes tokens para al menos un batch completo
                while token_buffer.len() >= tokens_needed_for_batch + 1 {
                    let batch_start_time = Instant::now();
                    
                    if global_batches_skipped < skip_batches {
                        global_batches_skipped += 1;
                        token_buffer.drain(0..tokens_per_batch);
                        if global_batches_skipped % 100 == 0 || global_batches_skipped == skip_batches {
                            print!("\r    ⏩ Saltando batches: {}/{}... ", global_batches_skipped, skip_batches);
                            io::stdout().flush().unwrap();
                        }
                        continue;
                    }

                    let (input, target) = create_batch::<MyBackend>(
                        &token_buffer, 0, batch_size, seq_length, stride, vocab_size, &device
                    );

                    let (logits, _) = model.forward(input, None);
                    let logits_flat = logits.reshape([batch_size * seq_length, vocab_size]);
                    let target_flat = target.reshape::<1, _>([batch_size * seq_length]);

                    let loss = loss_fn.forward(logits_flat, target_flat);
                    let loss_val = loss.clone().into_data().as_slice::<f32>().unwrap()[0];
                    
                    let grads = loss.backward();
                    model = model.optimizer_step(&lr_config, &mut optim, grads);

                    // --- DEBUG: Inspeccionar exactamente lo que entra al modelo ---
                    if processed_in_epoch == 0 {
                        println!("\n--- 🔍 PRIMER BATCH ENTRENADO (EP:{}, GLOBAL:{}) ---", epoch, global_batches_skipped);
                        for b_idx in 0..2.min(batch_size) {
                            let start = b_idx * stride;
                            let indices = &token_buffer[start..start + seq_length];
                            let decoded = tokenizer.decode(indices);
                            let clean_debug = decoded
                                .replace('▁', " ")
                                .replace('Ġ', " ")
                                .replace('Ċ', "\n")
                                .replace('\t', "    ");
                            
                            println!("SEQ {}:", b_idx);
                            println!("--------------------------------------------------");
                            println!("{}", clean_debug);
                            println!("--------------------------------------------------");
                        }
                        println!("------------------------------------------------------------\n");
                    }

                    total_loss += loss_val;
                    processed_in_epoch += 1;
                    global_batches_skipped += 1;

                    // Consumimos los tokens procesados del búfer
                    token_buffer.drain(0..tokens_per_batch);

                    // Reporte fluido
                    let elapsed = batch_start_time.elapsed().as_secs_f32();
                    print!("\r    Global Batch {} | Loss: {:.4} | Time: {:.3}s | Buf: {}     ", 
                        global_batches_skipped, loss_val, elapsed, token_buffer.len());
                    io::stdout().flush().unwrap();

                    // --- GUARDADO Y GENERACIÓN AUTOMÁTICA ---
                    if last_save_time.elapsed().as_secs() >= 180 {
                        println!("\n\n    💾 Guardado automático (Intervalo 4 min)...");
                        let recorder = CompactRecorder::new();
                        let _ = model.clone().save_file(model_path, &recorder);
                        
                        println!("    🔍 Generando muestra de prueba:");
                        let seed = "func _ready():";
                        let start_gen = Instant::now();
                        let generated = generate_text(&model.valid(), &tokenizer, seed, 200, vocab_size, &device);
                        let elapsed_gen = start_gen.elapsed().as_secs_f32();
                        println!("    ✨ Generado en {:.3}s:", elapsed_gen);
                        
                        let full_text = format!("{}{}", seed, generated);
                        let clean_text = full_text
                            .replace('▁', " ")
                            .replace('Ġ', " ")
                            .replace('Ċ', "\n")
                            .replace('\t', "    ");

                        println!("    ┌──────────────────────────────────────────────────");
                        for line in clean_text.lines() {
                            println!("    │ {}", line);
                        }
                        println!("    └──────────────────────────────────────────────────\n");
                        
                        last_save_time = Instant::now();
                    }
                }
            }
            // --- FIX: Separador entre archivos para evitar que se peguen las palabras ---
            token_buffer.extend(tokenizer.encode("<|endoftext|>")); 
        }
        println!("\n  ✅ Época finalizada. Guardando...");
        let recorder = CompactRecorder::new();
        let _ = model.clone().save_file(model_path, &recorder);
    }

    Ok(())
}
