use anyhow::Result;
use clap::{Parser, Subcommand};
use colored::*;
use std::path::PathBuf;
use std::time::Instant;
use toontalker_core::{
    ArtisticDomain, DenseMotionGenerator, MotionKeypoint, ToonTalkerConfig, ToonTalkerPipeline,
    WeightsManager,
};

#[derive(Parser, Debug)]
#[command(
    name = "toontalker",
    author = "ToonTalker Contributors",
    version = "0.1.0",
    about = "High-performance Rust CLI for ToonTalker: Cross-Domain Cartoon Talking Face Animation"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Animate cartoon portrait with cross-domain motion transfer
    Animate {
        /// Source cartoon portrait image
        #[arg(short = 's', long = "source", required = true)]
        source: PathBuf,

        /// Output animation destination path
        #[arg(short = 'o', long = "output", default_value = "cartoon_animation.mp4")]
        output: PathBuf,

        /// Target domain style (cartoon, anime, caricature, sketch, oilpainting, 3davatar, realportrait)
        #[arg(short = 'd', long = "domain", default_value = "cartoon")]
        domain: String,

        /// Canvas resolution (default: 256)
        #[arg(long = "size", default_value_t = 256)]
        size: u32,
    },

    /// List supported artistic domains and motion scale factors
    Domains,

    /// Inspect images or SafeTensors checkpoints
    Inspect {
        /// Target file
        #[arg(required = true)]
        file: PathBuf,
    },

    /// Benchmark motion field computation and transfer throughput
    Bench,
}

fn parse_domain(s: &str) -> ArtisticDomain {
    match s.to_lowercase().as_str() {
        "anime" => ArtisticDomain::Anime,
        "caricature" => ArtisticDomain::Caricature,
        "sketch" => ArtisticDomain::Sketch,
        "oilpainting" | "oil" => ArtisticDomain::OilPainting,
        "3davatar" | "3d" => ArtisticDomain::ThreeDAvatar,
        "realportrait" | "real" => ArtisticDomain::RealPortrait,
        _ => ArtisticDomain::Cartoon,
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Animate {
            source,
            output,
            domain,
            size,
        } => {
            let art_domain = parse_domain(&domain);
            println!("{}", "=== ToonTalker Cross-Domain Animation ===".bold().cyan());
            println!("  Source Image:  {}", source.display().to_string().yellow());
            println!("  Domain Style:  {:?}", art_domain);
            println!("  Resolution:    {}x{}", size, size);

            let config = ToonTalkerConfig {
                size,
                fps: 25.0,
                domain: art_domain,
                relative_motion: true,
                adapt_scale: true,
                checkpoint_dir: None,
            };

            let start = Instant::now();
            let pipeline = ToonTalkerPipeline::new(config);
            let out_file = pipeline.run(&source, &output)?;

            println!(
                "{} Animation motion transfer complete in {:.2?} -> {}",
                "✓".green(),
                start.elapsed(),
                out_file.display().to_string().bold()
            );
        }

        Commands::Domains => {
            println!("{}", "=== Supported Visual Domains in ToonTalker ===".bold().cyan());
            let domains = [
                (ArtisticDomain::Cartoon, "2D animated cartoons and stylized illustrations"),
                (ArtisticDomain::Anime, "Japanese animation characters with large eyes and distinct mouth shapes"),
                (ArtisticDomain::Caricature, "Exaggerated facial features and expressive deformation"),
                (ArtisticDomain::Sketch, "Pencil and monochrome line sketches"),
                (ArtisticDomain::OilPainting, "Classical fine-art oil portraits with texture preservation"),
                (ArtisticDomain::ThreeDAvatar, "Stylized 3D CGI characters and metaverse avatars"),
                (ArtisticDomain::RealPortrait, "Photorealistic human reference subjects"),
            ];

            for (d, desc) in domains {
                println!("  • {:<15} (Motion Scale: {:<4.2}) : {}",
                    format!("{:?}", d).yellow(),
                    d.motion_scale_factor(),
                    desc
                );
            }
        }

        Commands::Inspect { file } => {
            println!("{} {}", "Inspecting:".bold().cyan(), file.display());
            let ext = file.extension().and_then(|s| s.to_str()).unwrap_or("");

            match ext.to_lowercase().as_str() {
                "safetensors" => {
                    let tensors = WeightsManager::inspect_checkpoint(&file)?;
                    println!("  Type:          SafeTensors Checkpoint");
                    println!("  Tensors Count: {}", tensors.len());
                    for t in tensors.iter().take(10) {
                        println!("    • {:<35} {:?} ({})", t.name, t.shape, t.dtype);
                    }
                    if tensors.len() > 10 {
                        println!("    ... and {} more tensors", tensors.len() - 10);
                    }
                }
                "json" => {
                    let text = std::fs::read_to_string(&file)?;
                    println!("  Type:          JSON Motion Data");
                    println!("  Characters:    {}", text.len());
                }
                _ => {
                    println!("File extension .{} not supported for detailed inspection", ext);
                }
            }
        }

        Commands::Bench => {
            println!("{}", "=== Benchmarking ToonTalker Dense Motion Field ===".bold().cyan());
            let generator = DenseMotionGenerator::new(256);
            let src_kp = (0..10)
                .map(|i| MotionKeypoint {
                    x: 0.2 + (i as f32 * 0.05),
                    y: 0.2 + (i as f32 * 0.05),
                    jacobian: [1.0, 0.0, 0.0, 1.0],
                })
                .collect::<Vec<_>>();

            let tgt_kp = src_kp
                .iter()
                .map(|k| MotionKeypoint {
                    x: k.x + 0.05,
                    y: k.y + 0.03,
                    jacobian: k.jacobian,
                })
                .collect::<Vec<_>>();

            let start = Instant::now();
            let iterations = 100;
            for _ in 0..iterations {
                let _ = generator.compute_motion_field(&src_kp, &tgt_kp);
            }
            let dur = start.elapsed();
            let per_frame = dur / iterations;

            println!("  Dense Flow (256x256): {:.2?} per frame ({:.1} FPS throughput)",
                per_frame, 1.0 / per_frame.as_secs_f64());
        }
    }

    Ok(())
}
