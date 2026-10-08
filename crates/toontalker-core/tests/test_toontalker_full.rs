use toontalker_core::{
    ArtisticDomain, DenseMotionGenerator, FrameMotion, MotionKeypoint, MotionTransfer,
    ToonTalkerConfig, WeightsManager,
};

#[test]
fn test_artistic_domain_scaling() {
    assert_eq!(ArtisticDomain::default(), ArtisticDomain::Cartoon);
    assert_eq!(ArtisticDomain::Cartoon.motion_scale_factor(), 1.15);
    assert_eq!(ArtisticDomain::Anime.motion_scale_factor(), 1.25);
    assert_eq!(ArtisticDomain::Caricature.motion_scale_factor(), 1.40);
    assert_eq!(ArtisticDomain::RealPortrait.motion_scale_factor(), 1.0);
}

#[test]
fn test_motion_transfer_relative() {
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
    // Relative motion preserves origin on frame 0
    assert_eq!(res[0].keypoints[0].x, 0.5);
    assert_eq!(res[0].keypoints[0].y, 0.5);
    // Frame 1 has motion
    assert!(res[1].keypoints[0].x > 0.5);
    assert!(res[1].keypoints[0].y > 0.5);
}

#[test]
fn test_dense_motion_generator() {
    let gen = DenseMotionGenerator::new(64);
    assert_eq!(gen.width, 64);
    assert_eq!(gen.height, 64);

    let src = vec![MotionKeypoint {
        x: 0.5,
        y: 0.5,
        jacobian: [1.0, 0.0, 0.0, 1.0],
    }];
    let tgt = vec![MotionKeypoint {
        x: 0.6,
        y: 0.7,
        jacobian: [1.0, 0.0, 0.0, 1.0],
    }];

    let flow = gen.compute_motion_field(&src, &tgt);
    assert_eq!(flow.width, 64);
    assert_eq!(flow.height, 64);
    assert_eq!(flow.flow.len(), 64 * 64);
    assert_eq!(flow.occlusion_mask.len(), 64 * 64);
}

#[test]
fn test_config_defaults() {
    let config = ToonTalkerConfig::default();
    assert_eq!(config.domain, ArtisticDomain::Cartoon);
    assert!(config.relative_motion);
    assert!(config.adapt_scale);
    assert_eq!(config.size, 256);
}

#[test]
fn test_weights_manager_invalid_path() {
    let res = WeightsManager::inspect_checkpoint("/invalid/path/toontalker.safetensors");
    assert!(res.is_err());
}
