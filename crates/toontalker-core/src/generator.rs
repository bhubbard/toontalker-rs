use crate::motion::MotionKeypoint;
use serde::{Deserialize, Serialize};

/// 2D dense optical flow vector field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DenseFlowField {
    pub width: usize,
    pub height: usize,
    /// Flow vectors (dx, dy) per pixel
    pub flow: Vec<(f32, f32)>,
    /// Occlusion confidence mask [0.0 = occluded, 1.0 = visible]
    pub occlusion_mask: Vec<f32>,
}

pub struct DenseMotionGenerator {
    pub width: usize,
    pub height: usize,
}

impl DenseMotionGenerator {
    pub fn new(size: usize) -> Self {
        Self {
            width: size,
            height: size,
        }
    }

    /// Compute approximation of dense motion flow field from sparse keypoints
    pub fn compute_motion_field(
        &self,
        source_kp: &[MotionKeypoint],
        target_kp: &[MotionKeypoint],
    ) -> DenseFlowField {
        let total_pixels = self.width * self.height;
        let mut flow = vec![(0.0f32, 0.0f32); total_pixels];
        let mut occlusion_mask = vec![1.0f32; total_pixels];

        if source_kp.is_empty() || target_kp.is_empty() {
            return DenseFlowField {
                width: self.width,
                height: self.height,
                flow,
                occlusion_mask,
            };
        }

        // Radial Basis Function (RBF) inverse-distance weighted interpolation
        for y in 0..self.height {
            let py = y as f32 / self.height as f32;
            for x in 0..self.width {
                let px = x as f32 / self.width as f32;

                let mut total_weight = 0.0f32;
                let mut sum_dx = 0.0f32;
                let mut sum_dy = 0.0f32;

                for (src, tgt) in source_kp.iter().zip(target_kp.iter()) {
                    let dist_sq = (px - src.x).powi(2) + (py - src.y).powi(2);
                    let weight = (-dist_sq / 0.08).exp();

                    let dx = tgt.x - src.x;
                    let dy = tgt.y - src.y;

                    sum_dx += dx * weight;
                    sum_dy += dy * weight;
                    total_weight += weight;
                }

                let idx = y * self.width + x;
                if total_weight > 1e-5 {
                    let flow_x = sum_dx / total_weight;
                    let flow_y = sum_dy / total_weight;
                    flow[idx] = (flow_x, flow_y);

                    // Occlusion estimation from motion gradient
                    let mag = (flow_x * flow_x + flow_y * flow_y).sqrt();
                    if mag > 0.25 {
                        occlusion_mask[idx] = (1.0 - (mag - 0.25) * 2.0).clamp(0.0, 1.0);
                    }
                }
            }
        }

        DenseFlowField {
            width: self.width,
            height: self.height,
            flow,
            occlusion_mask,
        }
    }
}
