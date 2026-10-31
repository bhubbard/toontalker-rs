use crate::domain::ArtisticDomain;
use serde::{Deserialize, Serialize};

/// 2D Keypoint with 2x2 local Jacobian deformation matrix
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotionKeypoint {
    pub x: f32,
    pub y: f32,
    /// 2x2 local Jacobian matrix [j00, j01, j10, j11]
    pub jacobian: [f32; 4],
}

impl Default for MotionKeypoint {
    fn default() -> Self {
        Self {
            x: 0.0,
            y: 0.0,
            jacobian: [1.0, 0.0, 0.0, 1.0],
        }
    }
}

impl MotionKeypoint {
    /// Apply 2x2 local Jacobian matrix to a 2D offset vector (dx, dy)
    pub fn transform_local_offset(&self, dx: f32, dy: f32) -> (f32, f32) {
        let [j00, j01, j10, j11] = self.jacobian;
        (j00 * dx + j01 * dy, j10 * dx + j11 * dy)
    }
}

/// Set of K motion keypoints describing a single frame's pose and expression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrameMotion {
    pub frame_idx: usize,
    pub keypoints: Vec<MotionKeypoint>,
}

pub struct MotionTransfer {
    pub domain: ArtisticDomain,
    pub relative: bool,
    pub adapt_scale: bool,
}

impl MotionTransfer {
    pub fn new(domain: ArtisticDomain, relative: bool, adapt_scale: bool) -> Self {
        Self {
            domain,
            relative,
            adapt_scale,
        }
    }

    /// Cross-Domain Transformer (CDT) canonical alignment between real driving and cartoon source keypoints
    pub fn cross_domain_transformer_align(
        &self,
        source_kp: &[MotionKeypoint],
        driving_kp: &[MotionKeypoint],
    ) -> Vec<MotionKeypoint> {
        let scale = self.domain.motion_scale_factor();
        source_kp
            .iter()
            .enumerate()
            .map(|(idx, src)| {
                let drv = driving_kp.get(idx).unwrap_or(src);
                let (jx, jy) = src.transform_local_offset(drv.x - src.x, drv.y - src.y);
                MotionKeypoint {
                    x: (src.x + jx * 0.25 * scale).clamp(0.0, 1.0),
                    y: (src.y + jy * 0.25 * scale).clamp(0.0, 1.0),
                    jacobian: [
                        src.jacobian[0] * 0.5 + drv.jacobian[0] * 0.5,
                        src.jacobian[1] * 0.5 + drv.jacobian[1] * 0.5,
                        src.jacobian[2] * 0.5 + drv.jacobian[2] * 0.5,
                        src.jacobian[3] * 0.5 + drv.jacobian[3] * 0.5,
                    ],
                }
            })
            .collect()
    }

    /// Transfer driving motion onto source cartoon keypoints with domain adaptation
    pub fn transfer_motion(
        &self,
        source_kp: &[MotionKeypoint],
        driving_sequence: &[FrameMotion],
    ) -> Vec<FrameMotion> {
        if driving_sequence.is_empty() || source_kp.is_empty() {
            return Vec::new();
        }

        let base_driving = &driving_sequence[0].keypoints;
        let scale = if self.adapt_scale {
            self.domain.motion_scale_factor()
        } else {
            1.0
        };

        let mut output = Vec::with_capacity(driving_sequence.len());

        for frame in driving_sequence {
            let mut transferred_kp = Vec::with_capacity(source_kp.len());

            for (k, src) in source_kp.iter().enumerate() {
                let drv = if k < frame.keypoints.len() {
                    &frame.keypoints[k]
                } else {
                    src
                };
                let base_drv = if k < base_driving.len() {
                    &base_driving[k]
                } else {
                    src
                };

                let (new_x, new_y) = if self.relative {
                    let dx = (drv.x - base_drv.x) * scale;
                    let dy = (drv.y - base_drv.y) * scale;
                    (src.x + dx, src.y + dy)
                } else {
                    (drv.x, drv.y)
                };

                // Blend Jacobians
                let mut new_jac = src.jacobian;
                for j in 0..4 {
                    new_jac[j] = src.jacobian[j] * 0.4 + drv.jacobian[j] * 0.6;
                }

                transferred_kp.push(MotionKeypoint {
                    x: new_x,
                    y: new_y,
                    jacobian: new_jac,
                });
            }

            output.push(FrameMotion {
                frame_idx: frame.frame_idx,
                keypoints: transferred_kp,
            });
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relative_motion_transfer() {
        let src = vec![MotionKeypoint {
            x: 0.5,
            y: 0.5,
            jacobian: [1.0, 0.0, 0.0, 1.0],
        }];
        let drv = vec![
            FrameMotion {
                frame_idx: 0,
                keypoints: vec![MotionKeypoint {
                    x: 0.2,
                    y: 0.2,
                    jacobian: [1.0, 0.0, 0.0, 1.0],
                }],
            },
            FrameMotion {
                frame_idx: 1,
                keypoints: vec![MotionKeypoint {
                    x: 0.3,
                    y: 0.4,
                    jacobian: [1.0, 0.0, 0.0, 1.0],
                }],
            },
        ];

        let transfer = MotionTransfer::new(ArtisticDomain::Cartoon, true, true);
        let res = transfer.transfer_motion(&src, &drv);
        assert_eq!(res.len(), 2);
        // First frame displacement is 0.0
        assert_eq!(res[0].keypoints[0].x, 0.5);
        // Second frame has transferred delta
        assert!(res[1].keypoints[0].x > 0.5);

        let aligned = transfer.cross_domain_transformer_align(&src, &drv[1].keypoints);
        assert_eq!(aligned.len(), 1);
    }
}
