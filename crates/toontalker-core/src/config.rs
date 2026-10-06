use crate::domain::ArtisticDomain;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToonTalkerConfig {
    /// Canvas resolution (standard: 256)
    pub size: u32,
    /// Video frame rate (default 25.0)
    pub fps: f32,
    /// Target cartoon visual domain
    pub domain: ArtisticDomain,
    /// Transfer relative motion rather than absolute coordinates
    pub relative_motion: bool,
    /// Adapt scale to source face proportions
    pub adapt_scale: bool,
    /// Checkpoint directory override
    pub checkpoint_dir: Option<PathBuf>,
}

impl Default for ToonTalkerConfig {
    fn default() -> Self {
        Self {
            size: 256,
            fps: 25.0,
            domain: ArtisticDomain::Cartoon,
            relative_motion: true,
            adapt_scale: true,
            checkpoint_dir: None,
        }
    }
}
