#![recursion_limit = "256"]

/*!
Text Generation with xLSTM using Character-Level Tokenization

This example demonstrates how to use xLSTM for text generation
using a simple character-level tokenizer that can be saved/loaded as JSON.

Author: Based on xlstm-rs project
Date: January 2026
*/

use burn::grad_clipping::GradientClippingConfig;

use burn::optim::decay::WeightDecayConfig;
use burn::{
    module::AutodiffModule,
    module::Module,
    optim::AdamConfig,
    record::{CompactRecorder, Recorder},
    tensor::{activation::softmax, Tensor, backend::{AutodiffBackend, Backend}},
};
use burn::tensor::TensorData;
use burn_autodiff::Autodiff;
use burn_wgpu::{Wgpu, WgpuDevice};
use std::error::Error;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::time::Instant;

use tokenizers::decoders::metaspace::Metaspace as MetaspaceDecoder;
use tokenizers::models::bpe::{BPE, BpeTrainerBuilder};
use tokenizers::pre_tokenizers::metaspace::{Metaspace, PrependScheme};
use tokenizers::tokenizer::Tokenizer as HFTokenizer;
use tokenizers::models::TrainerWrapper;
use std::collections::HashSet;

use xlstm::{LearningRateConfig, LstmType, XLstm, XLstmconfig};

type MyBackend = Autodiff<Wgpu<f32, i32>>;

/// Tokenizador profesional usando la librería 'tokenizers' de Hugging Face
pub struct Tokenizer {
    tokenizer: HFTokenizer,
}

impl Tokenizer {
    /// Crea un nuevo tokenizador BPE entrenado desde un texto
    pub fn from_text(text: &str, vocab_size: usize) -> Result<Self, Box<dyn Error>> {
        // Usar byte_fallback para manejar caracteres desconocidos
        let model = BPE::builder()
            .byte_fallback(true)
            .build()
            .map_err(|e| format!("Error building BPE: {}", e))?;
            
        let mut tokenizer = HFTokenizer::new(model);
        
        // Usar Metaspace (como en GPT-2/RoBERTa) para preservar espacios
        tokenizer.with_pre_tokenizer(Some(Metaspace::new(
            '▁', 
            PrependScheme::Always,
            true,
        )));

        // AGREGAR DECODER PARA QUE 'decode' NO META ESPACIOS ENTRE SUB-TOKENS
        tokenizer.with_decoder(Some(MetaspaceDecoder::new('▁', PrependScheme::Always, true)));

        // Asegurar que caracteres de control básicos estén en el alfabeto inicial
        let mut alphabet = HashSet::new();
        alphabet.insert('\n');
        alphabet.insert(' ');

        let trainer = BpeTrainerBuilder::default()
            .show_progress(true)
            .vocab_size(vocab_size)
            .min_frequency(0) // Permitir todos para empezar
            .initial_alphabet(alphabet)
            .build();

        // Envolver el entrenador de manera genérica usando el trait From
        let mut trainer_wrapper = TrainerWrapper::from(trainer);

        // Entrenar desde el archivo temporal
        let temp_file = "temp_train.txt";
        fs::write(temp_file, text)?;
        tokenizer.train_from_files(&mut trainer_wrapper, vec![temp_file.to_string()])
            .map_err(|e| format!("Error en entrenamiento: {}", e))?;
        fs::remove_file(temp_file)?;

        Ok(Self { tokenizer })
    }

    /// Guarda el tokenizador en un archivo
    pub fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        self.tokenizer.save(path, true)
            .map_err(|e| format!("Error al guardar: {}", e))?;
        println!("Tokenizador guardado en: {}", path);
        Ok(())
    }

    /// Carga el tokenizador desde un archivo
    pub fn load(path: &str) -> Result<Self, Box<dyn Error>> {
        let mut tokenizer = HFTokenizer::from_file(path)
            .map_err(|e| format!("Error al cargar: {}", e))?;
            
        // IMPORTANTE: Incluso al cargar, debemos asegurar el decoder para que no rompa palabras
        tokenizer.with_decoder(Some(MetaspaceDecoder::new('▁', PrependScheme::Always, true)));
        
        println!("Tokenizador cargado desde: {}", path);
        Ok(Self { tokenizer })
    }

    /// Convierte texto a índices
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let encoding = self.tokenizer.encode(text, false).unwrap();
        encoding.get_ids().iter().map(|&id| id as usize).collect()
    }

    /// Convierte índices a texto
    pub fn decode(&self, indices: &[usize]) -> String {
        let u32_indices: Vec<u32> = indices.iter().map(|&idx| idx as u32).collect();
        self.tokenizer.decode(&u32_indices, true).expect("Error al decodificar")
    }

    /// Obtiene el tamaño del vocabulario
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Obtiene el string de un token por su índice
    pub fn id_to_token(&self, id: usize) -> Option<String> {
        self.tokenizer.id_to_token(id as u32)
    }
}

/// Crea un batch de entrenamiento (one-hot) de forma eficiente usando una matriz identidad
fn create_batch<B: AutodiffBackend>(
    tokens: &[usize],
    start_idx: usize,
    batch_size: usize,
    seq_length: usize,
    stride: usize, // NEW ARGUMENT
    vocab_size: usize,
    device: &B::Device,
) -> (Tensor<B, 3>, Tensor<B, 2, burn::tensor::Int>) {
    let mut x_indices = Vec::with_capacity(batch_size * seq_length);
    let mut y_indices = Vec::with_capacity(batch_size * seq_length);

    for i in 0..batch_size {
        let current_start = start_idx + i * stride; // FIX: Multiply by stride
        for j in 0..seq_length {
            x_indices.push(tokens[current_start + j] as i32);
            y_indices.push(tokens[current_start + j + 1] as i32);
        }
    }

    let eye = Tensor::<B::InnerBackend, 2>::eye(vocab_size, device);
    let indices_inner = Tensor::<B::InnerBackend, 1, burn::tensor::Int>::from_data(
        TensorData::new(x_indices, [batch_size * seq_length]),
        device,
    );

    let x = Tensor::<B, 3>::from_inner(
        eye.select(0, indices_inner)
           .reshape([batch_size, seq_length, vocab_size])
    );
    
    let y = Tensor::<B, 2, burn::tensor::Int>::from_data(
        TensorData::new(y_indices, [batch_size, seq_length]),
        device,
    );

    (x, y)
}

/// Selecciona un token usando muestreo estocástico con Top-K, Top-P (Nucleus) y temperatura
fn sample_from_logits<B: Backend>(
    logits: Tensor<B, 2>, 
    temperature: f32,
    top_k: usize,
    top_p: f32
) -> usize
where
    <B as Backend>::FloatElem: num_traits::ToPrimitive,
{
    // Aplicar softmax para obtener probabilidades base
    let probs = softmax(logits, 1);
    let mut probs_vec: Vec<(usize, f32)> = probs.to_data()
        .as_slice::<<B as Backend>::FloatElem>()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, &x)| (i, num_traits::ToPrimitive::to_f32(&x).unwrap_or(0.0)))
        .collect();

    // Ordenar de mayor a menor probabilidad
    probs_vec.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    
    // --- TOP-K ---
    let k = top_k.min(probs_vec.len()).max(1);
    let mut filtered_probs: Vec<(usize, f32)> = Vec::with_capacity(k);
    
    // --- TOP-P (Nucleus Sampling) ---
    let mut cumulative_prob = 0.0;
    for (i, p) in probs_vec.into_iter() {
        filtered_probs.push((i, p));
        cumulative_prob += p;
        if filtered_probs.len() >= k || cumulative_prob >= top_p {
            break;
        }
    }

    // Extraer solo los pesos para el muestreo
    let indices: Vec<usize> = filtered_probs.iter().map(|(i, _)| *i).collect();
    let mut weights: Vec<f32> = filtered_probs.iter().map(|(_, p)| *p).collect();

    // Si la temperatura es muy baja, actuar de forma determinista (Greedy)
    if temperature <= 1e-6 {
        return indices[0];
    }

    // Aplicar temperatura
    for p in weights.iter_mut() {
        *p = (p.max(1e-10).ln() / temperature).exp();
    }

    let sum: f32 = weights.iter().sum();
    use rand::Rng;
    let mut rng = rand::rng(); // Usar API moderna rand::rng()
    let sample: f32 = rng.random::<f32>() * sum; // Usar random() en lugar de gen()

    let mut cumulative = 0.0;

    for (i, &p) in weights.iter().enumerate() {
        cumulative += p;
        if sample <= cumulative {
            return indices[i];
        }
    }

    indices[0]
}

/// Genera texto de forma recurrente manteniendo el estado interno del modelo
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
    
    if seed_tokens.is_empty() {
        return seed_text.to_string();
    }

    let eye = Tensor::<B, 2>::eye(vocab_size, device);
    
    // 1. IMPORTANTE: No definas el tipo de current_state. 
    // Al ser una variable local, Rust infiere el tipo privado de la librería automáticamente.
    let mut current_state = None; 
    let mut current_tokens = seed_tokens.clone();

    // Procesar la semilla primero para calentar el estado
    // (Esto ya se hace en el loop abajo si i=0, pero vamos a ajustar el loop para generar 'length' tokens NUEVOS)
    
    // El loop original iteraba 'length' veces.
    // Vamos a mantener la lógica pero usando el nuevo sampleo y decoding.

    for i in 0..length {
        // En el primer paso (i == 0) procesamos toda la semilla para inicializar la memoria.
        // Después, solo procesamos el último token generado para mantener eficiencia O(1).
        let tokens_to_process = if i == 0 {
            current_tokens.clone()
        } else {
            vec![*current_tokens.last().unwrap()]
        };

        let seq_len = tokens_to_process.len();
        let indices = Tensor::<B, 1, burn::tensor::Int>::from_data(
            TensorData::new(tokens_to_process.iter().map(|&t| t as i32).collect(), [seq_len]),
            device,
        );

        let input = eye.clone()
            .select(0, indices)
            .reshape([1, seq_len, vocab_size]);
   
        // 2. Ejecutamos el forward
        let (output, next_state) = model.forward(input, current_state);
        
        // 3. Actualizamos el estado
        current_state = Some(next_state);

        // 4. Extraemos el último paso de los logits
        let dims = output.dims();
        let last_logits = output
            .slice([0..1, (dims[1] - 1)..dims[1], 0..dims[2]])
            .reshape([1, dims[2]]);

        // 5. Muestreo con temperatura y Top-P/ Top-K
        // Muestreo equilibrado (Top-P 0.9 para evitar repeticiones, Temp 0.7 para fluidez)
        let next_token = sample_from_logits(last_logits, 0.4, 40, 0.9);

        current_tokens.push(next_token);
        generated_ids.push(next_token);
        
        // Efecto visual inmediato para saltos de línea
        if let Some(t) = tokenizer.id_to_token(next_token) {
            if t.contains('Ċ') {
                print!("\n"); 
            }
        }
    }

    // 6. Decodificar solo los tokens generados (sin la semilla)
    tokenizer.decode(&generated_ids[seed_tokens.len()..])
}


fn main() -> Result<(), Box<dyn Error>> {
    println!("xLSTM Text Generation con Tokenizador");
    println!("======================================\n");

    // Parsear argumentos
    let args: Vec<String> = std::env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Uso: cargo run --bin mainchat -- <archivo.txt>");
        eprintln!("Ejemplo: cargo run --bin mainchat -- input.txt");
        std::process::exit(1);
    }

    let text_file = &args[1];
    let tokenizer_path = "tokenizer.json";
    let model_path = "xlstm_chat_model_mlstm";

    // Intentar leer vocab_size de argumentos o usar 2000 por defecto
    let target_vocab_size = 1024;

    // Cargar o crear tokenizador
    let tokenizer = if Path::new(tokenizer_path).exists() {
        println!("Cargando tokenizador existente...");
        Tokenizer::load(tokenizer_path)?
    } else {
        println!("Entrenando nuevo tokenizador profesional (BPE) desde {}...", text_file);
        let text = fs::read_to_string(text_file)?;
        let tokenizer = Tokenizer::from_text(&text, target_vocab_size)?;
        tokenizer.save(tokenizer_path)?;
        tokenizer
    };

    println!("Tamaño del vocabulario: {}\n", tokenizer.vocab_size());

    // Cargar texto de entrenamiento
    println!("Cargando texto de entrenamiento...");
    let text = fs::read_to_string(text_file)?;
    let tokens = tokenizer.encode(&text);
    println!("Tokens totales: {}\n", tokens.len());

    // Hiperparámetros - PROTECCIÓN DE RAM
    let vocab_size = tokenizer.vocab_size();
    let hidden_size = 256; // Suficiente para BPE
    let num_layers = 1;
    let num_blocks = 2;
    let output_size = vocab_size; 
    let dropout = 0.0;

    let seq_length = 128; 
    let batch_size = 16; 
    let stride = 64;     // Más data por época
    let num_epochs = 50;
    let num_heads = 4;
    // Learning rates por bloque (igual que main.rs)
    let lr_config = LearningRateConfig::per_block_type(
        5e-4, // sLSTM learning rate
        1e-4, // mLSTM learning rate (Very aggressive for faster convergence)
        5e-4, // Other components learning rate
    );

    println!("Configuración del modelo:");
    println!("  Bloques: {}", num_blocks);
    println!("  Hidden size: {}", hidden_size);
    println!("  Seq length: {}", seq_length);
    println!("  Batch size: {}", batch_size);
    println!("  Epochs: {}\n", num_epochs);

    // Device
    let device = WgpuDevice::default();

    // Configuración del modelo - vocab_size es el input_size (one-hot)
   // let config = XLstmconfig::new(vocab_size, hidden_size, num_layers, num_blocks, output_size)
     //   .with_dropout(dropout)
      //  .with_lstm_type(LstmType::Alternate)
       // .with_use_projection(true);


// Configuración
     let config = XLstmconfig::new(vocab_size, hidden_size, num_layers, num_blocks, output_size)
        .with_dropout(dropout)
        .with_num_heads(num_heads)
        //.with_lstm_type(LstmType::Alternate)
        .with_lstm_type(LstmType::MLSTM) // <--- Forzar solo mLSTM
        .with_use_projection(true);   

    // Verificar si existe un modelo guardado (una sola vez)
    let model_file = format!("{}.mpk", model_path);
    let existe_modelo = Path::new(&model_file).exists();
    
    let mut continuar_entrenamiento = false;
    if existe_modelo {
        print!("¿Deseas seguir entrenando el modelo cargado? (s/n): ");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        if input.trim().to_lowercase() == "s" {
            continuar_entrenamiento = true;
        }
    }

    let model = if existe_modelo && !continuar_entrenamiento {
        println!("¡Modelo encontrado! Cargando pesos para generación...");
        let recorder = CompactRecorder::new();
        let record = recorder
            .load(model_file.into(), &device)
            .map_err(|e| format!("Error al cargar modelo: {}", e))?;
        
        let loaded_model = config.init::<MyBackend>(&device).load_record(record);
        println!("Modelo cargado exitosamente!\n");
        loaded_model
    } else {
        let mut model = if continuar_entrenamiento {
            println!("Cargando modelo previo para continuar entrenamiento...");
            let recorder = CompactRecorder::new();
            let record = recorder
                .load(model_file.into(), &device)
                .map_err(|e| format!("Error al cargar modelo: {}", e))?;
            config.init::<MyBackend>(&device).load_record(record)
        } else {
            println!("No se encontró modelo guardado. Iniciando entrenamiento desde cero...\n");
            config.init::<MyBackend>(&device)
        };

        // Imprimir el primer embedding (one-hot vector) para inspección ANTES de procesar todo el texto
        if !tokens.is_empty() {
            let first_token_idx = tokens[0];
            let first_token_str = tokenizer.id_to_token(first_token_idx).unwrap_or("?".to_string());
            
            println!("--- INSPECCIÓN DE EMBEDDING PROFESIONAL (BPE) ---");
            println!("  Token (BPE): '{}'", first_token_str);
            println!("  Token Index: {}", first_token_idx);
            println!("  Dimensión del vector (Vocab size): {}", vocab_size);
            println!("----------------------------------------------------\n");
        }

        let num_sequences = tokens.len().saturating_sub(seq_length);
        // Ajustar num_sequences según el stride
        let num_actual_sequences = (num_sequences + stride - 1) / stride;
        
        println!("Tokens para procesar: {}", tokens.len());
        println!("Secuencias únicas calculadas (Stride {}): {}\n", stride, num_actual_sequences);

        // Crear modelo
        println!("Creando modelo xLSTM con bloques alternados sLSTM/mLSTM...");
        //let mut model = config.init::<MyBackend>(&device);
        model.print_architecture();
        println!();

        // Crear optimizador
        let mut optim = AdamConfig::new()
            .with_beta_1(0.9)
            .with_beta_2(0.999)
            .with_epsilon(1e-8)
            .with_weight_decay(Some(WeightDecayConfig::new(1e-4)))
            .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
            .init();

        println!("Iniciando entrenamiento...\n");

        // Training loop - Solo procesar baches COMPLETOS para evitar errores de dimensión
        let num_batches = num_actual_sequences / batch_size; 

         for epoch in 0..num_epochs {
            let mut total_loss = 0.0f32;
            let mut num_losses = 0;
            let mut correct = 0;
            let mut total = 0;
            let mut current_state = None;
            for batch_idx in 0..num_batches {

                let current_batch_start_seq = batch_idx * batch_size;
                let current_batch_size = (batch_size).min(num_actual_sequences - current_batch_start_seq);
                let epoch_start = Instant::now();
                if current_batch_size == 0 {
                    break;
                }

                // Generar batch instantáneo (usando Inner Backend para ahorrar RAM)
                let (input_batch, target_batch) = create_batch::<MyBackend>(
                    &tokens,
                    current_batch_start_seq * stride,
                    current_batch_size,
                    seq_length,
                    stride, // Pass stride
                    vocab_size,
                    &device,
                );
                 if batch_idx == 0 {
                // Hacemos un forward silencioso para llenar las matrices del mLSTM
                let (_, warm_state) = model.predict_last(input_batch.clone(), None);
                current_state = Some(warm_state);
                println!("> Estado inicializado con éxito en el Batch 0");
            }   
                // Forward pass
               // let (logits, _) = model.forward(input_batch.clone(), None);
               let (logits, next_state) = model.forward(input_batch.clone(), current_state);
                current_state = Some(next_state.into_iter().map(|s| {
    s.map(|state| state.detach())
}).collect());
                // --- OPTIMIZACIÓN: COSTE Y ACCURACY NATIVOS SOBRE TODA LA SECUENCIA ---
                
                // --- OPTIMIZACIÓN: COSTE Y ACCURACY NATIVOS SOBRE TODA LA SECUENCIA ---
                
                // Aplanar para cálculo eficiente
                let logits_flat = logits.reshape::<2, _>([current_batch_size * seq_length, vocab_size]);
                let target_flat = target_batch.reshape::<1, _>([current_batch_size * seq_length]);

                // Usar inner backend para los targets para que no consuman memoria de gradientes
                let eye_inner = Tensor::<Wgpu<f32, i32>, 2>::eye(vocab_size, &device);
                let target_one_hot = Tensor::<MyBackend, 2>::from_inner(
                    eye_inner.select(0, target_flat.clone().inner())
                             .reshape([current_batch_size * seq_length, vocab_size])
                );

                // 2. Calcular Cross-Entropy nativo sobre toda la secuencia
                let log_probs = (softmax(logits_flat.clone(), 1) + 1e-10).log();
                let loss_tensor = -(target_one_hot * log_probs).sum_dim(1).mean();
                
                let loss_f32 = loss_tensor.clone().into_data().as_slice::<f32>().unwrap()[0];
                total_loss += loss_f32;
                num_losses += 1;

                // 3. Calcular Accuracy nativo sobre toda la secuencia
                let predicted_indices = logits_flat.argmax(1).reshape([current_batch_size * seq_length]);
                let matches = predicted_indices.equal(target_flat);
                let correct_batch = matches.int().sum().into_data().as_slice::<i32>().unwrap()[0];
                
                correct += correct_batch as usize;
                total += current_batch_size * seq_length;

                // --- FIN OPTIMIZACIÓN ---

                let grads = loss_tensor.backward();
                model = model.optimizer_step(&lr_config, &mut optim, grads);

                // Reportar progreso cada 10 batches para que se vea el movimiento fluido
                if batch_idx % 1 == 0 || batch_idx == num_batches - 1 {
                    let elapsed = epoch_start.elapsed().as_secs_f32();
                    print!("\r  -> Batch [{}/{}] Loss: {:.4}  Acc: {:.2}% ({:.1}s)    ", 
                        batch_idx + 1, num_batches, total_loss / (batch_idx + 1) as f32,
                       100.0 * correct as f32 / total as f32, elapsed);
                    io::stdout().flush().unwrap();
                }
            }
            println!(); 

            let avg_loss = total_loss / num_losses as f32;
            let accuracy = 100.0 * correct as f32 / total as f32;

            if epoch % 1 == 0 {
                println!(
                    "Epoch [{:3}/{}], Loss: {:.4}, Accuracy (Avg): {:.2}%",
                    epoch + 1,
                    num_epochs,
                    avg_loss,
                    accuracy
                );
            }

            if epoch % 1 == 0 {

                // GUARDADO POR ÉPOCA (ADICIONAL)
                let recorder = CompactRecorder::new();
                let _ = model.clone().save_file(model_path, &recorder);

                // Generar texto de ejemplo con temperatura y SEMILLA ALEATORIA
                if epoch % 1 == 0 {
                    use rand::Rng;
                    let mut rng = rand::rng(); 
                    
                    // Elegir un punto de inicio al azar para la semilla (15 tokens para contexto real)
                    let seed_len = 15;
                    let start_random = if tokens.len() > seed_len + 1 {
                        rng.random_range(0..tokens.len() - seed_len - 1) 
                    } else {
                        0
                    };
                    
                    let seed_tokens: Vec<usize> = tokens[start_random..start_random + seed_len].to_vec();
                    // Limpieza de la semilla (eliminando decoradores de BPE)
                    let seed = tokenizer.decode(&seed_tokens)
                        .replace('▁', " ")
                        .replace('Ġ', " ")
                        .replace('Ċ', "\n")
                        .replace("  ", " "); // Quitar espacios extra que ensucian
                    
                    println!("  -> Semilla ({} tokens): \"{}\"", seed_len, seed.trim());
                    let generated = generate_text(
                        &model, // Pasamos referencia sin clonar
                        &tokenizer,
                        &seed,
                        100, // Generar 100 palabras para ver la capacidad real
                        vocab_size,
                        &device,
                    );
                    println!("  Generado: {}{}\n", seed, generated);

                    // --- LOGGER: Guardar en archivo para ver la evolución ---
                    let log_path = "training_history_mlstm.txt";
                    let mut file = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(log_path)?;
                    
                    writeln!(file, "====================================================")?;
                    writeln!(file, "ÉPOCA: {} | LOSS: {:.4} | ACC: {:.2}%", epoch + 1, avg_loss, accuracy)?;
                    writeln!(file, "SEMILLA: {}", seed)?;
                    writeln!(file, "----------------------------------------------------")?;
                    writeln!(file, "{}", generated)?;
                    writeln!(file, "====================================================\n\n")?;
                }
            }
        }

        println!("\n¡Entrenamiento completado!");

        // Guardar modelo
        println!("Guardando modelo...");
        let recorder = CompactRecorder::new();
        model
            .clone()
            .save_file(model_path, &recorder)
            .map_err(|e| format!("Error al guardar modelo: {}", e))?;
        println!("Modelo guardado en: {}.mpk\n", model_path);

        model
    };

    // Modo interactivo - Loop para generar texto
    println!("\n╔════════════════════════════════════════════════════════╗");
    println!("║        MODO INTERACTIVO - GENERACIÓN DE TEXTO         ║");
    println!("╚════════════════════════════════════════════════════════╝\n");
    println!("Comandos:");
    println!("  - Escribe un texto semilla y presiona Enter para generar");
    println!("  - Escribe 'salir' o 'exit' para terminar");
    println!("  - Escribe 'auto' para generar con semilla automática\n");

    loop {
        print!("Semilla > ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();

        if input.is_empty() {
            continue;
        }

        if input.eq_ignore_ascii_case("salir") || input.eq_ignore_ascii_case("exit") {
            println!("\n¡Hasta luego!");
            break;
        }

        let seed = if input.eq_ignore_ascii_case("auto") {
            text.chars().take(20).collect::<String>()
        } else {
            input.to_string()
        };

        println!("\n┌─ Generando texto...");
        println!("│ Semilla: {}", seed);
        println!("└─ Longitud: 200 caracteres\n");

        let generated = generate_text(
            &model.valid(),
            &tokenizer,
            &seed,
            200,
            vocab_size,
            &device,
        );

        println!("╭─────────────────────────────────────────────────────────╮");
        println!("│ TEXTO GENERADO:");
        println!("├─────────────────────────────────────────────────────────┤");
        
        // Dividir en líneas de máximo 55 caracteres para mejor visualización
        let mut chars_count = 0;
        print!("│ ");
        for ch in generated.chars() {
            print!("{}", ch);
            chars_count += 1;
            if ch == '\n' || chars_count >= 55 {
                if ch != '\n' {
                    println!();
                }
                print!("│ ");
                chars_count = 0;
            }
        }
        if chars_count > 0 {
            println!();
        }
        println!("╰─────────────────────────────────────────────────────────╯\n");
    }

    Ok(())
}
