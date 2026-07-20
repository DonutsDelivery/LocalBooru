use serde::{Deserialize, Serialize};

pub const DEFAULT_SURFACE_POOL_SIZE: usize = 3;
pub const MAX_SURFACE_PLANES: usize = 4;
pub const MAX_SURFACE_DIMENSION: u32 = 16_384;
pub const DRM_FORMAT_ABGR8888: u32 = 0x3432_4241;
pub const DRM_FORMAT_YUV420: u32 = 0x3231_5559;

fn default_sample_aspect_ratio() -> f64 {
    1.0
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceHandleKind {
    DmaBuf,
    SharedMemory,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceColorSpace {
    Bt601,
    Bt709,
    Bt2020,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceColorRange {
    Narrow,
    Full,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceChromaLocation {
    Left,
    Center,
    TopLeft,
    Top,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SurfacePlane {
    pub stride: u32,
    pub offset: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmabufObject {
    pub size: u64,
    pub modifier: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmabufPlane {
    pub object_index: u32,
    pub stride: u32,
    pub offset: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmabufLayer {
    pub fourcc: u32,
    pub width: u32,
    pub height: u32,
    pub planes: Vec<DmabufPlane>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DmabufLayout {
    pub objects: Vec<DmabufObject>,
    pub layers: Vec<DmabufLayer>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SurfaceDescriptor {
    pub generation: u64,
    pub buffer_id: u32,
    pub width: u32,
    pub height: u32,
    #[serde(default = "default_sample_aspect_ratio")]
    pub sample_aspect_ratio: f64,
    #[serde(default)]
    pub rotation_degrees: i32,
    pub fourcc: u32,
    pub modifier: u64,
    pub handle_kind: SurfaceHandleKind,
    #[serde(default)]
    pub reusable_dmabuf: bool,
    pub producer_drm_node: Option<String>,
    #[serde(default)]
    pub color_space: Option<SurfaceColorSpace>,
    #[serde(default)]
    pub color_range: Option<SurfaceColorRange>,
    #[serde(default)]
    pub chroma_location: Option<SurfaceChromaLocation>,
    pub planes: Vec<SurfacePlane>,
    #[serde(default)]
    pub dmabuf: Option<DmabufLayout>,
}

impl SurfaceDescriptor {
    pub fn validate(&self, received_fd_count: usize) -> Result<(), String> {
        if self.width == 0
            || self.height == 0
            || self.width > MAX_SURFACE_DIMENSION
            || self.height > MAX_SURFACE_DIMENSION
        {
            return Err("surface dimensions are outside the supported range".to_string());
        }
        if !self.sample_aspect_ratio.is_finite()
            || self.sample_aspect_ratio <= 0.0
            || !matches!(self.rotation_degrees.rem_euclid(360), 0 | 90 | 180 | 270)
        {
            return Err("surface display geometry metadata is invalid".to_string());
        }
        if self.planes.is_empty() || self.planes.len() > MAX_SURFACE_PLANES {
            return Err("surface plane count is outside the supported range".to_string());
        }
        if self.planes.iter().any(|plane| plane.stride == 0) {
            return Err("surface plane stride must be non-zero".to_string());
        }
        if self.handle_kind == SurfaceHandleKind::SharedMemory {
            let valid_planes = (self.fourcc == DRM_FORMAT_ABGR8888 && self.planes.len() == 1)
                || (self.fourcc == DRM_FORMAT_YUV420 && self.planes.len() == 3);
            if !valid_planes {
                return Err("shared-memory surface format and planes are inconsistent".to_string());
            }
            if received_fd_count != 1 {
                return Err(format!(
                    "shared-memory descriptor expected one backing descriptor, received {received_fd_count}"
                ));
            }
            return Ok(());
        }
        if self.handle_kind == SurfaceHandleKind::DmaBuf {
            if let Some(layout) = &self.dmabuf {
                if layout.objects.is_empty() || layout.objects.len() > 5 {
                    return Err("DMA-BUF object count is outside the supported range".to_string());
                }
                if received_fd_count != layout.objects.len() {
                    return Err(format!(
                        "DMA-BUF descriptor expected {} object descriptors, received {received_fd_count}",
                        layout.objects.len()
                    ));
                }
                if layout.layers.is_empty()
                    || layout.layers.iter().any(|layer| {
                        layer.width == 0
                            || layer.height == 0
                            || layer.width > self.width
                            || layer.height > self.height
                            || layer.planes.is_empty()
                            || layer.planes.len() > MAX_SURFACE_PLANES
                            || layer.planes.iter().any(|plane| {
                                plane.stride == 0
                                    || plane.object_index as usize >= layout.objects.len()
                            })
                    })
                {
                    return Err("DMA-BUF layer metadata is invalid".to_string());
                }
                return Ok(());
            }
        }
        if received_fd_count != self.planes.len() {
            return Err(format!(
                "surface descriptor expected {} file descriptors, received {received_fd_count}",
                self.planes.len()
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SurfaceFrameReady {
    pub generation: u64,
    pub buffer_id: u32,
    pub sequence: u64,
    pub pts_seconds: f64,
    pub has_native_fence: bool,
}

impl SurfaceFrameReady {
    pub fn validate(&self, received_fd_count: usize) -> Result<(), String> {
        let expected = usize::from(self.has_native_fence);
        if received_fd_count != expected {
            return Err(format!(
                "frame-ready expected {expected} fence descriptors, received {received_fd_count}"
            ));
        }
        if !self.pts_seconds.is_finite() || self.pts_seconds < 0.0 {
            return Err("frame-ready timestamp must be finite and non-negative".to_string());
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SurfaceFrameRelease {
    pub generation: u64,
    pub buffer_id: u32,
    pub sequence: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn descriptor(kind: SurfaceHandleKind) -> SurfaceDescriptor {
        SurfaceDescriptor {
            generation: 7,
            buffer_id: 1,
            width: 1920,
            height: 1080,
            sample_aspect_ratio: 1.0,
            rotation_degrees: 0,
            fourcc: 0x3432_4241,
            modifier: 0,
            handle_kind: kind,
            reusable_dmabuf: false,
            producer_drm_node: Some("/dev/dri/renderD128".to_string()),
            color_space: None,
            color_range: None,
            chroma_location: None,
            planes: vec![SurfacePlane {
                stride: 1920 * 4,
                offset: 0,
            }],
            dmabuf: None,
        }
    }

    #[test]
    fn descriptor_requires_one_ancillary_fd_per_plane() {
        let descriptor = descriptor(SurfaceHandleKind::DmaBuf);
        assert!(descriptor.validate(1).is_ok());
        assert!(descriptor.validate(0).is_err());
        assert!(descriptor.validate(2).is_err());
    }

    #[test]
    fn shared_memory_rejects_multiplane_descriptors() {
        let mut descriptor = descriptor(SurfaceHandleKind::SharedMemory);
        descriptor.planes.push(SurfacePlane {
            stride: 960 * 4,
            offset: 1920 * 1080 * 4,
        });
        assert!(descriptor.validate(2).is_err());
    }

    #[test]
    fn shared_memory_accepts_planar_yuv420_layout() {
        let mut descriptor = descriptor(SurfaceHandleKind::SharedMemory);
        descriptor.fourcc = DRM_FORMAT_YUV420;
        descriptor.producer_drm_node = None;
        descriptor.planes = vec![
            SurfacePlane {
                stride: 1920,
                offset: 0,
            },
            SurfacePlane {
                stride: 960,
                offset: 1920 * 1080,
            },
            SurfacePlane {
                stride: 960,
                offset: 1920 * 1080 + 960 * 540,
            },
        ];
        assert!(descriptor.validate(1).is_ok());
        assert!(descriptor.validate(3).is_err());
        descriptor.planes[2].stride = 0;
        assert!(descriptor.validate(1).is_err());
    }

    #[test]
    fn dmabuf_layout_validates_object_and_layer_mapping() {
        let mut descriptor = descriptor(SurfaceHandleKind::DmaBuf);
        descriptor.dmabuf = Some(DmabufLayout {
            objects: vec![
                DmabufObject {
                    size: 4096,
                    modifier: 17,
                },
                DmabufObject {
                    size: 2048,
                    modifier: 17,
                },
            ],
            layers: vec![
                DmabufLayer {
                    fourcc: u32::from_le_bytes(*b"R8  "),
                    width: 1920,
                    height: 1080,
                    planes: vec![DmabufPlane {
                        object_index: 0,
                        stride: 1920,
                        offset: 0,
                    }],
                },
                DmabufLayer {
                    fourcc: u32::from_le_bytes(*b"RG88"),
                    width: 960,
                    height: 540,
                    planes: vec![DmabufPlane {
                        object_index: 1,
                        stride: 1920,
                        offset: 0,
                    }],
                },
            ],
        });
        assert!(descriptor.validate(2).is_ok());
        assert!(descriptor.validate(1).is_err());
        descriptor.dmabuf.as_mut().unwrap().layers[1].planes[0].object_index = 2;
        assert!(descriptor.validate(2).is_err());
    }

    #[test]
    fn native_fence_presence_controls_frame_ready_fd_count() {
        let frame = SurfaceFrameReady {
            generation: 7,
            buffer_id: 1,
            sequence: 42,
            pts_seconds: 3.5,
            has_native_fence: true,
        };
        assert!(frame.validate(1).is_ok());
        assert!(frame.validate(0).is_err());
    }

    #[test]
    fn release_round_trips_without_pixel_payloads() {
        let release = SurfaceFrameRelease {
            generation: 8,
            buffer_id: 2,
            sequence: 99,
        };
        let encoded = serde_json::to_string(&release).unwrap();
        assert!(!encoded.contains("pixels"));
        assert_eq!(
            serde_json::from_str::<SurfaceFrameRelease>(&encoded).unwrap(),
            release
        );
    }
}
