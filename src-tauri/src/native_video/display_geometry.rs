use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DisplayMode {
    Fit,
    Fill,
    Original,
}

impl Default for DisplayMode {
    fn default() -> Self {
        Self::Fit
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogicalRect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalRect {
    pub x: i32,
    pub y: i32,
    pub width: i32,
    pub height: i32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UvCrop {
    pub left: f32,
    pub top: f32,
    pub right: f32,
    pub bottom: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisplayGeometryInput {
    pub coded_width: u32,
    pub coded_height: u32,
    pub sample_aspect_ratio: f64,
    pub rotation_degrees: i32,
    pub viewport_width: f64,
    pub viewport_height: f64,
    pub scale_factor: f64,
    pub mode: DisplayMode,
    pub title_bar_safe_inset: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DisplayGeometry {
    pub physical_viewport: PhysicalRect,
    pub uv_crop: UvCrop,
    pub content_rect: LogicalRect,
    pub subtitle_safe_rect: LogicalRect,
    pub hud_hit_test_rect: LogicalRect,
    pub rotation_quadrants: u8,
}

fn normalize_rotation(rotation_degrees: i32) -> Result<u8, String> {
    match rotation_degrees.rem_euclid(360) {
        0 => Ok(0),
        90 => Ok(1),
        180 => Ok(2),
        270 => Ok(3),
        _ => Err("display rotation must be a multiple of 90 degrees".to_string()),
    }
}

fn physical_rect(rect: LogicalRect, scale_factor: f64, viewport_height: f64) -> PhysicalRect {
    PhysicalRect {
        x: (rect.x * scale_factor).round() as i32,
        y: ((viewport_height - rect.y - rect.height) * scale_factor).round() as i32,
        width: (rect.width * scale_factor).round().max(1.0) as i32,
        height: (rect.height * scale_factor).round().max(1.0) as i32,
    }
}

fn inset_rect(rect: LogicalRect, horizontal: f64, vertical: f64) -> LogicalRect {
    LogicalRect {
        x: rect.x + horizontal,
        y: rect.y + vertical,
        width: (rect.width - horizontal * 2.0).max(1.0),
        height: (rect.height - vertical * 2.0).max(1.0),
    }
}

pub fn compute_display_geometry(input: DisplayGeometryInput) -> Result<DisplayGeometry, String> {
    if input.coded_width == 0
        || input.coded_height == 0
        || !input.sample_aspect_ratio.is_finite()
        || input.sample_aspect_ratio <= 0.0
        || !input.viewport_width.is_finite()
        || !input.viewport_height.is_finite()
        || input.viewport_width <= 0.0
        || input.viewport_height <= 0.0
        || !input.scale_factor.is_finite()
        || input.scale_factor <= 0.0
        || !input.title_bar_safe_inset.is_finite()
        || input.title_bar_safe_inset < 0.0
        || input.title_bar_safe_inset >= input.viewport_height
    {
        return Err("display geometry input is invalid".to_string());
    }

    let rotation_quadrants = normalize_rotation(input.rotation_degrees)?;
    let unrotated_width = f64::from(input.coded_width) * input.sample_aspect_ratio;
    let unrotated_height = f64::from(input.coded_height);
    let (source_width, source_height) = if rotation_quadrants % 2 == 0 {
        (unrotated_width, unrotated_height)
    } else {
        (unrotated_height, unrotated_width)
    };
    let available = LogicalRect {
        x: 0.0,
        y: input.title_bar_safe_inset,
        width: input.viewport_width,
        height: input.viewport_height - input.title_bar_safe_inset,
    };
    let source_aspect = source_width / source_height;
    let viewport_aspect = available.width / available.height;

    let (physical_viewport, content_rect, uv_crop) = match input.mode {
        DisplayMode::Fit => {
            let scale = (available.width / source_width).min(available.height / source_height);
            let rect = LogicalRect {
                x: available.x + (available.width - source_width * scale) * 0.5,
                y: available.y + (available.height - source_height * scale) * 0.5,
                width: source_width * scale,
                height: source_height * scale,
            };
            (
                physical_rect(rect, input.scale_factor, input.viewport_height),
                rect,
                UvCrop {
                    left: 0.0,
                    top: 0.0,
                    right: 1.0,
                    bottom: 1.0,
                },
            )
        }
        DisplayMode::Fill => {
            let crop = if source_aspect > viewport_aspect {
                let visible = viewport_aspect / source_aspect;
                let margin = ((1.0 - visible) * 0.5) as f32;
                UvCrop {
                    left: margin,
                    top: 0.0,
                    right: 1.0 - margin,
                    bottom: 1.0,
                }
            } else {
                let visible = source_aspect / viewport_aspect;
                let margin = ((1.0 - visible) * 0.5) as f32;
                UvCrop {
                    left: 0.0,
                    top: margin,
                    right: 1.0,
                    bottom: 1.0 - margin,
                }
            };
            (
                physical_rect(available, input.scale_factor, input.viewport_height),
                available,
                crop,
            )
        }
        DisplayMode::Original => {
            let rect = LogicalRect {
                x: available.x + (available.width - source_width) * 0.5,
                y: available.y + (available.height - source_height) * 0.5,
                width: source_width,
                height: source_height,
            };
            (
                physical_rect(rect, input.scale_factor, input.viewport_height),
                rect,
                UvCrop {
                    left: 0.0,
                    top: 0.0,
                    right: 1.0,
                    bottom: 1.0,
                },
            )
        }
    };

    let visible_content = LogicalRect {
        x: content_rect.x.max(available.x),
        y: content_rect.y.max(available.y),
        width: (content_rect.x + content_rect.width).min(available.x + available.width)
            - content_rect.x.max(available.x),
        height: (content_rect.y + content_rect.height).min(available.y + available.height)
            - content_rect.y.max(available.y),
    };
    let subtitle_safe_rect = inset_rect(
        visible_content,
        visible_content.width * 0.05,
        visible_content.height * 0.05,
    );

    Ok(DisplayGeometry {
        physical_viewport,
        uv_crop,
        content_rect,
        subtitle_safe_rect,
        hud_hit_test_rect: available,
        rotation_quadrants,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input(width: u32, height: u32) -> DisplayGeometryInput {
        DisplayGeometryInput {
            coded_width: width,
            coded_height: height,
            sample_aspect_ratio: 1.0,
            rotation_degrees: 0,
            viewport_width: 800.0,
            viewport_height: 832.0,
            scale_factor: 1.0,
            mode: DisplayMode::Fit,
            title_bar_safe_inset: 32.0,
        }
    }

    #[test]
    fn fit_handles_landscape_portrait_square_anamorphic_and_rotation() {
        let cases = [
            (input(1920, 1080), (800, 450), 0),
            (input(1080, 1920), (450, 800), 0),
            (input(1000, 1000), (800, 800), 0),
            (
                DisplayGeometryInput {
                    sample_aspect_ratio: 2.0,
                    ..input(720, 576)
                },
                (800, 320),
                0,
            ),
            (
                DisplayGeometryInput {
                    rotation_degrees: 90,
                    ..input(1920, 1080)
                },
                (450, 800),
                1,
            ),
        ];
        for (case, expected_size, expected_rotation) in cases {
            let geometry = compute_display_geometry(case).unwrap();
            assert_eq!(
                (
                    geometry.physical_viewport.width,
                    geometry.physical_viewport.height
                ),
                expected_size
            );
            assert_eq!(geometry.rotation_quadrants, expected_rotation);
            assert!(geometry.content_rect.y >= 32.0);
        }
    }

    #[test]
    fn physical_output_tracks_fractional_platform_scale_factors() {
        for scale in [1.0, 1.25, 1.5, 2.0] {
            let geometry = compute_display_geometry(DisplayGeometryInput {
                scale_factor: scale,
                ..input(1920, 1080)
            })
            .unwrap();
            assert_eq!(geometry.physical_viewport.width, (800.0 * scale) as i32);
            assert_eq!(
                geometry.physical_viewport.height,
                (450.0 * scale).round() as i32
            );
            assert_eq!(geometry.physical_viewport.y, (175.0 * scale).round() as i32);
        }
    }

    #[test]
    fn fill_uses_one_canonical_uv_crop() {
        let geometry = compute_display_geometry(DisplayGeometryInput {
            mode: DisplayMode::Fill,
            ..input(1920, 1080)
        })
        .unwrap();
        assert_eq!(geometry.content_rect.width, 800.0);
        assert_eq!(geometry.content_rect.height, 800.0);
        assert!((geometry.uv_crop.left - 0.21875).abs() < 0.000_01);
        assert!((geometry.uv_crop.right - 0.78125).abs() < 0.000_01);
    }

    #[test]
    fn original_preserves_intrinsic_display_size_and_centering() {
        let geometry = compute_display_geometry(DisplayGeometryInput {
            viewport_width: 2560.0,
            viewport_height: 1472.0,
            scale_factor: 1.5,
            mode: DisplayMode::Original,
            ..input(1920, 1080)
        })
        .unwrap();
        assert_eq!(geometry.content_rect.width, 1920.0);
        assert_eq!(geometry.content_rect.height, 1080.0);
        assert_eq!(geometry.physical_viewport.width, 2880);
        assert_eq!(geometry.physical_viewport.height, 1620);
        assert_eq!(geometry.hud_hit_test_rect.y, 32.0);
    }

    #[test]
    fn invalid_rotation_and_geometry_are_rejected() {
        assert!(compute_display_geometry(DisplayGeometryInput {
            rotation_degrees: 45,
            ..input(1920, 1080)
        })
        .is_err());
        assert!(compute_display_geometry(DisplayGeometryInput {
            sample_aspect_ratio: 0.0,
            ..input(1920, 1080)
        })
        .is_err());
    }
}
