use std::io::{self, Read};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.first().map(String::as_str) == Some("--qnn-context") {
        if let Err(error) = build_qnn_context(&args[1..]) {
            eprintln!("{error}");
            std::process::exit(1);
        }
        return;
    }
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .expect("read request");
    let replay = args.iter().any(|arg| arg == "--decision-replay");
    if args.iter().any(|arg| arg == "--rim-check") {
        match bhe_runtime::rim_check::judge_video_json(&input) {
            Ok(output) => println!("{output}"),
            Err(error) => {
                eprintln!("{error}");
                std::process::exit(1);
            }
        }
        return;
    }
    let result = if replay {
        bhe_runtime::evaluate_decision_replay_json(&input)
    } else {
        bhe_runtime::analyze_json(&input)
    };
    match result {
        Ok(output) => println!("{output}"),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}

#[cfg(all(target_os = "android", feature = "android-ep"))]
fn build_qnn_context(args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    use ort::ep::QNN;
    let [model, output] = args else {
        return Err("usage: bhe-runtime --qnn-context <model.onnx> <context.onnx>".into());
    };
    ort::session::Session::builder()?
        .with_config_entry("ep.context_enable", "1")?
        .with_config_entry("ep.context_embed_mode", "1")?
        .with_config_entry("ep.context_file_path", output)?
        .with_execution_providers([QNN::default()
            .with_backend_path("libQnnHtp.so")
            .with_htp_fp16_precision(true)
            .build()
            .error_on_failure()])?
        .commit_from_file(model)?;
    println!("{output}");
    Ok(())
}

#[cfg(not(all(target_os = "android", feature = "android-ep")))]
fn build_qnn_context(_args: &[String]) -> Result<(), Box<dyn std::error::Error>> {
    Err("--qnn-context requires an Android build with the android-ep feature".into())
}
