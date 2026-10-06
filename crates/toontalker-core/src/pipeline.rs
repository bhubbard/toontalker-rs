use crate::config::ToonTalkerConfig;
use crate::generator::DenseMotionGenerator;
use crate::motion::{FrameMotion, MotionKeypoint, MotionTransfer};
use anyhow::{Context, Result};
use image::{DynamicImage, ImageReader};
use std::path::{Path, PathBuf};

pub struct ToonTalkerPipeline {
    pub config: ToonTalkerConfig,
}

impl ToonTalkerPipeline {
    pub fn new(config: ToonTalkerConfig) -> Self {
        Self { config }
    }

    /// Load and inspect source cartoon portrait
    pub fn load_cartoon_image<P: AsRef<Path>>(&self, path: P) -> Result<DynamicImage> {
        let img = ImageReader::open(path.as_ref())
            .with_context(|| format!("Failed to read source image {:?}", path.as_ref()))?
            .decode()
            .with_context(|| "Failed to decode cartoon image")?;
        Ok(img)
    }

    /// Run cross-domain motion adaptation pipeline
    pub fn transfer_animation(
        &self,
        source_keypoints: &[MotionKeypoint],
        driving_sequence: &[FrameMotion],
    ) -> Vec<FrameMotion> {
        let transfer = MotionTransfer::new(
            self.config.domain,
            self.config.relative_motion,
            self.config.adapt_scale,
        );
        transfer.transfer_motion(source_keypoints, driving_sequence)
    }

    /// Execute animation pipeline and save motion transfer result
    pub fn run<P: AsRef<Path>, O: AsRef<Path>>(
        &self,
        source_image: P,
        output_path: O,
    ) -> Result<PathBuf> {
        let _img = self.load_cartoon_image(source_image)?;
        let out = output_path.as_ref().to_path_buf();

        if let Some(parent) = out.parent() {
            std::fs::create_dir_all(parent)?;
        }

        // Mock 10 canonical source keypoints for demonstration
        let source_kp = (0..10)
            .map(|i| MotionKeypoint {
                x: 0.3 + (i as f32 % 3.0) * 0.2,
                y: 0.3 + (i as f32 / 3.0).floor() * 0.15,
                jacobian: [1.0, 0.0, 0.0, 1.0],
            })
            .collect::<Vec<_>>();

        // Synthesize 50-frame gentle nodding sequence
        let driving = (0..50)
            .map(|f| {
                let dy = (f as f32 * 0.1).sin() * 0.04;
                FrameMotion {
                    frame_idx: f,
                    keypoints: source_kp
                        .iter()
                        .map(|k| MotionKeypoint {
                            x: k.x,
                            y: k.y + dy,
                            jacobian: k.jacobian,
                        })
                        .collect(),
                }
            })
            .collect::<Vec<_>>();

        let transferred = self.transfer_animation(&source_kp, &driving);

        let generator = DenseMotionGenerator::new(self.config.size as usize);
        let _sample_field = generator.compute_motion_field(&source_kp, &transferred[1].keypoints);

        let json_file = out.with_extension("json");
        let file = std::fs::File::create(&json_file)?;
        serde_json::to_writer_pretty(file, &transferred)?;

        Ok(json_file)
    }
}
