use xlstm::{MLstm, MLstmconfig, MLstmstate};
use burn::tensor::{Tensor, Distribution};
use burn::backend::Autodiff;

type TestBackend = burn_ndarray::NdArray<f32>;

fn run_equivalence() {
    let device = Default::default();
    let batch_size = 2;
    let seq_len = 5;
    let input_size = 16;
    let hidden_size = 32;
    let num_heads = 4;
    
    let config = MLstmconfig::new(input_size, hidden_size, 1)
        .with_num_heads(num_heads)
        .with_dropout(0.0);
    
    let mlstm: MLstm<TestBackend> = config.init(&device);
    let cell = &mlstm.layers[0];
    
    let input_seq: Tensor<TestBackend, 3> = Tensor::random(
        [batch_size, seq_len, input_size], 
        Distribution::Default, 
        &device
    );
    
    let internal_hidden_size = (hidden_size as f32 * config.proj_factor) as usize;
    let head_dim = internal_hidden_size / num_heads;
    
    let initial_state = MLstmstate::new(
        Tensor::<TestBackend, 4>::zeros([batch_size, num_heads, head_dim, head_dim], &device), // cell
        Tensor::<TestBackend, 2>::zeros([batch_size, hidden_size], &device), // hidden
        Tensor::<TestBackend, 3>::zeros([batch_size, num_heads, head_dim], &device), // normalizer
        Tensor::<TestBackend, 3>::zeros([batch_size, num_heads, 1], &device), // max_gate_log
    );
    
    // MODO PARALELO
    let (output_parallel, final_state_parallel) = cell.forward_sequence(&input_seq, initial_state.clone());
    
    // MODO RECURRENTE
    let mut current_state = initial_state;
    let mut outputs_recurrent: Vec<Tensor<TestBackend, 3>> = Vec::with_capacity(seq_len);
    
    for t in 0..seq_len {
        let input_t: Tensor<TestBackend, 2> = input_seq.clone()
            .slice([0..batch_size, t..t+1, 0..input_size])
            .reshape([batch_size, input_size]);
        
        let (output_t, new_state) = cell.forward(&input_t, current_state);
        outputs_recurrent.push(output_t.clone().reshape::<3, _>([batch_size, 1, hidden_size]));
        current_state = new_state;
    }
    
    let output_recurrent: Tensor<TestBackend, 3> = Tensor::cat(outputs_recurrent, 1);
    let final_state_recurrent = current_state;
    
    let output_diff: f32 = (output_parallel.clone() - output_recurrent.clone()).abs().mean().into_scalar();
    println!("Diferencia media en outputs: {:.2e}", output_diff);
    
    let cell_diff: f32 = (final_state_parallel.cell.clone() - final_state_recurrent.cell.clone()).abs().mean().into_scalar();
    println!("Diferencia media en cell states: {:.2e}", cell_diff);
    
    let norm_diff: f32 = (final_state_parallel.normalizer.clone() - final_state_recurrent.normalizer.clone()).abs().mean().into_scalar();
    println!("Diferencia media en normalizers: {:.2e}", norm_diff);
    
    if output_diff < 1e-4 {
        println!("✅ Test de equivalencia dual PASADO!");
    } else {
        println!("❌ ERROR: Los outputs no son equivalentes.");
        std::process::exit(1);
    }
}

fn run_grad_mlstm() {
    let device = Default::default();
    let batch_size = 1;
    let seq_len = 12;
    let input_size = 16;
    let hidden_size = 32;
    let num_heads = 4;

    type AdBackend = Autodiff<TestBackend>;

    let config = MLstmconfig::new(input_size, hidden_size, 1)
        .with_num_heads(num_heads)
        .with_dropout(0.0);
    
    let mlstm: MLstm<AdBackend> = config.init(&device);
    
    // Input con gradientes
    let x = Tensor::<AdBackend, 3>::random(
        [batch_size, seq_len, input_size], 
        Distribution::Normal(0.0, 1.0), 
        &device
    ).require_grad();

    // Forward a través de la secuencia paralela
    let (h_seq, _) = mlstm.forward(&x, None);
    
    // Tomamos el gradiente del último paso (cross-entropy dummy via sum)
    let h_last = h_seq.slice([0..batch_size, seq_len-1..seq_len, 0..hidden_size]).sum();
    
    let grads = h_last.backward();
    let x_grad = x.grad(&grads).expect("Debe existir gradiente para x en mLSTM");
    let grad_val = x_grad.abs().mean().into_scalar();

    println!("Gradiente REAL mLSTM (dual) |d last / d x|: {:.6}", grad_val);
    
    if grad_val > 1e-7 {
        println!("✅ Gradiente mLSTM saludable!");
    } else {
        println!("⚠️ ADVERTENCIA: Gradiente de mLSTM muy bajo ({:.2e})", grad_val);
    }
}

fn run_long_distance_grad() {
    let device = Default::default();
    let batch_size = 1;
    let seq_len = 512;
    let input_size = 16;
    let hidden_size = 32;
    let num_heads = 4;

    type AdBackend = Autodiff<TestBackend>;

    let config = MLstmconfig::new(input_size, hidden_size, 1)
        .with_num_heads(num_heads)
        .with_dropout(0.0);
    
    let mlstm: MLstm<AdBackend> = config.init(&device);
    
    // Input con gradientes
    let x = Tensor::<AdBackend, 3>::random(
        [batch_size, seq_len, input_size], 
        Distribution::Normal(0.0, 1.0), 
        &device
    ).require_grad();

    // Forward a través de la secuencia
    let (h_seq, _) = mlstm.forward(&x, None);
    
    // Tomamos la suma del ÚLTIMO paso de tiempo para ver cómo el error viaja hacia atrás
    let h_last = h_seq.slice([0..batch_size, seq_len-1..seq_len, 0..hidden_size]).sum();
    
    let grads = h_last.backward();
    let x_grad = x.grad(&grads).expect("Debe existir gradiente para x en mLSTM");
    
    // Extraemos el gradiente del PRIMER token (índice 0)
    let grad_start = x_grad.slice([0..batch_size, 0..1, 0..input_size]).abs().mean().into_scalar();

    println!("Gradiente de Larga Distancia (Token {} -> Token 0): {:.10}", seq_len - 1, grad_start);
    
    if grad_start > 1e-10 {
        println!("✅ TEST PASADO: El gradiente llegó vivo al inicio de la secuencia!");
        println!("   Esto demuestra que mLSTM evita el desvanecimiento de gradiente mejor que una RNN estándar.");
    } else {
        println!("⚠️ TEST FALLIDO: El gradiente se desvaneció ({:.2e})", grad_start);
    }
}

fn run_convergence_test() {
    use burn::optim::{AdamConfig, Optimizer};
    use burn::nn::loss::CrossEntropyLossConfig;
    use burn::tensor::TensorData;
    
    let device = Default::default();
    let batch_size = 2;
    let seq_len = 10;
    let vocab_size = 16;
    let hidden_size = 32;
    let num_heads = 4;

    type AdBackend = Autodiff<TestBackend>;

    // Configuración con projection (factor 2.0 por defecto interno)
    let config = MLstmconfig::new(vocab_size, hidden_size, 1) // Entrada=16, Salida=32
        .with_num_heads(num_heads)
        .with_dropout(0.0);
    
    let mut mlstm: MLstm<AdBackend> = config.init(&device);
    
    // Proyección manual final a vocabulario para poder calcular la pérdida
    let final_proj = burn::nn::LinearConfig::new(hidden_size, vocab_size).init(&device);

    let mut optim = AdamConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);

    // Secuencia objetivo determinista (ej. 1, 2, 3...)
    let mut target_indices = Vec::new();
    for i in 0..(batch_size * seq_len) {
        target_indices.push((i % vocab_size) as i64);
    }
    let targets = Tensor::<AdBackend, 1, burn::tensor::Int>::from_data(
        TensorData::new(target_indices.clone(), [batch_size * seq_len]),
        &device,
    );
    // Entradas One-Hot
    let eye = Tensor::<AdBackend, 2>::eye(vocab_size, &device);
    let inputs = eye.select(0, targets.clone()).reshape([batch_size, seq_len, vocab_size]);

    let mut loss_start = 0.0;
    let mut loss_end = 0.0;

    println!("Iniciando Overfit en 100 iteraciones...");
    for epoch in 0..100 {
        // Forward
        let (h_seq, _) = mlstm.forward(&inputs, None);
        let logits = final_proj.forward(h_seq).reshape([batch_size * seq_len, vocab_size]);
        
        let loss = loss_fn.forward(logits.clone(), targets.clone());

        let loss_val = loss.clone().into_data().as_slice::<f32>().unwrap()[0];
        
        if epoch == 0 { loss_start = loss_val; }
        if epoch == 99 { loss_end = loss_val; }

        // Backward
        let grads = loss.backward();
        let grads_params = burn::optim::GradientsParams::from_grads(grads, &mlstm);
        
        let lr = 0.01;
        
        // Optimizar manual
        mlstm = optim.step(lr, mlstm, grads_params); // LR alto para converger rápido en el test
    }

    println!("Loss Inicial: {:.4}", loss_start);
    println!("Loss Final:   {:.4}", loss_end);

    if loss_end < loss_start * 0.1 {
        println!("✅ TEST CONVERGENCIA PASADO: La red está aprendiendo y redujo el error dramáticamente.");
    } else {
        println!("❌ TEST CONVERGENCIA FALLIDO: La red no está aprendiendo a memorizar siquiera una secuencia estática.");
    }
}

fn run_convergence_recurrent_test() {
    use burn::optim::{AdamConfig, Optimizer};
    use burn::nn::loss::CrossEntropyLossConfig;
    use burn::tensor::TensorData;
    
    let device = Default::default();
    let batch_size = 2;
    let seq_len = 10;
    let vocab_size = 16;
    let hidden_size = 32;
    let num_heads = 4;

    type AdBackend = Autodiff<TestBackend>;

    let config = MLstmconfig::new(vocab_size, hidden_size, 1)
        .with_num_heads(num_heads)
        .with_dropout(0.0);
    
    let mut mlstm: MLstm<AdBackend> = config.init(&device);
    let final_proj = burn::nn::LinearConfig::new(hidden_size, vocab_size).init(&device);

    let mut optim = AdamConfig::new().init();
    let loss_fn = CrossEntropyLossConfig::new().init(&device);

    // Secuencia objetivo determinista
    let mut target_indices = Vec::new();
    for i in 0..(batch_size * seq_len) {
        target_indices.push((i % vocab_size) as i64);
    }
    let targets = Tensor::<AdBackend, 1, burn::tensor::Int>::from_data(
        TensorData::new(target_indices.clone(), [batch_size * seq_len]),
        &device,
    ).reshape([batch_size, seq_len]);

    // Entradas One-Hot completas
    let eye = Tensor::<AdBackend, 2>::eye(vocab_size, &device);
    let inputs_full = eye.select(0, targets.clone().reshape([batch_size * seq_len])).reshape([batch_size, seq_len, vocab_size]);

    let mut loss_start = 0.0;
    let mut loss_end = 0.0;

    println!("Iniciando Overfit RECURRENTE (Paso a paso con Estado) en 100 iteraciones...");
    for epoch in 0..100 {
        let mut current_state = None;
        let mut total_loss = Tensor::<AdBackend, 1>::zeros([1], &device);
        
        // Loop Recurrente: 1 token a la vez
        for t in 0..seq_len {
            let input_t = inputs_full.clone().slice([0..batch_size, t..t+1, 0..vocab_size]);
            let target_t = targets.clone().slice([0..batch_size, t..t+1]).reshape([batch_size * 1]);
            
            let (h_seq, next_state) = mlstm.forward(&input_t, current_state);
            current_state = Some(next_state);
            
            let logits = final_proj.forward(h_seq).reshape([batch_size * 1, vocab_size]);
            let step_loss = loss_fn.forward(logits, target_t);
            total_loss = total_loss + step_loss.reshape([1]);
        }
        
        let avg_loss = total_loss.clone() / (seq_len as f32);
        let loss_val = avg_loss.clone().into_data().as_slice::<f32>().unwrap()[0];
        
        if epoch == 0 { loss_start = loss_val; }
        if epoch == 99 { loss_end = loss_val; }

        let grads = avg_loss.backward();
        let grads_params = burn::optim::GradientsParams::from_grads(grads, &mlstm);
        mlstm = optim.step(0.01, mlstm, grads_params);
    }

    println!("Loss Inicial: {:.4}", loss_start);
    println!("Loss Final:   {:.4}", loss_end);

    if loss_end < loss_start * 0.1 {
        println!("✅ TEST RECURRENTE PASADO: El Cell paso a paso transporta correctamente la memoria y el gradiente fluye a través de los estados!");
    } else {
        println!("❌ TEST RECURRENTE FALLIDO: La propagación BPTT de estado a estado (forward_step) o la memoria C está matemáticamente rota.");
    }
}

fn main() {
    println!("--- Ejecutando Equivalencia Dual/Serial ---");
    run_equivalence();
    println!("\n--- Ejecutando Test de Gradientes mLSTM ---");
    run_grad_mlstm();
    println!("\n--- Ejecutando Test de Gradiente de Larga Distancia (512 tokens) ---");
   run_long_distance_grad();
    println!("\n--- Ejecutando Test de Convergencia Secuencial ---");
    run_convergence_test();
    println!("\n--- Ejecutando Test de Convergencia Recurrente (Estado Paso a Paso) ---");
    run_convergence_recurrent_test();
}

 // 