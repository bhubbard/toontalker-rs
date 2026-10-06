use serde::{Deserialize, Serialize};

/// Stylized face visual domains supported by ToonTalker
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ArtisticDomain {
    Cartoon,
    Anime,
    Caricature,
    Sketch,
    OilPainting,
    ThreeDAvatar,
    RealPortrait,
}

impl Default for ArtisticDomain {
    fn default() -> Self {
        Self::Cartoon
    }
}

impl ArtisticDomain {
    /// Domain-specific motion scaling factor (compensates for exaggerated cartoon mouth/eye proportions)
    pub fn motion_scale_factor(&self) -> f32 {
        match self {
            Self::Anime => 1.25,        // Exaggerated eye and mouth movements
            Self::Caricature => 1.40,   // High-deformation expressive motion
            Self::Cartoon => 1.15,
            Self::ThreeDAvatar => 1.05,
            Self::OilPainting => 0.90,  // Subtle, painterly texture preservation
            Self::Sketch => 1.0,
            Self::RealPortrait => 1.0,
        }
    }

    /// Default occluded area threshold
    pub fn occlusion_threshold(&self) -> f32 {
        match self {
            Self::Anime | Self::Cartoon => 0.45,
            _ => 0.50,
        }
    }
}
