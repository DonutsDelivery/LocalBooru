use std::collections::HashMap;
use std::os::fd::{AsRawFd, OwnedFd, RawFd};

use super::surface_channel::{ReceivedSurfaceMessage, SurfaceChannelMessage};
use super::surface_protocol::{
    SurfaceDescriptor, SurfaceFrameReady, SurfaceFrameRelease, SurfaceHandleKind,
};

#[derive(Default)]
pub struct DmabufSurfaceConsumer {
    generation: Option<u64>,
    surfaces: HashMap<u32, RegisteredSurface>,
}

struct RegisteredSurface {
    descriptor: SurfaceDescriptor,
    object_fds: Vec<OwnedFd>,
    lease: Option<ActiveLease>,
}

struct ActiveLease {
    sequence: u64,
    fence_fd: Option<OwnedFd>,
}

pub struct DmabufFrameView<'a> {
    pub descriptor: &'a SurfaceDescriptor,
    pub frame: SurfaceFrameReady,
    pub object_fds: Vec<RawFd>,
    pub fence_fd: Option<RawFd>,
}

impl DmabufSurfaceConsumer {
    pub fn contains_surface(&self, generation: u64, buffer_id: u32) -> bool {
        self.generation == Some(generation) && self.surfaces.contains_key(&buffer_id)
    }

    pub fn register(&mut self, received: ReceivedSurfaceMessage) -> Result<(), String> {
        let SurfaceChannelMessage::SurfaceCreated { descriptor } = received.message else {
            return Err("expected a surface-created message".to_string());
        };
        if descriptor.handle_kind != SurfaceHandleKind::DmaBuf {
            return Err("DMA-BUF consumer received a non-DMA-BUF surface".to_string());
        }
        if descriptor.dmabuf.is_none() {
            return Err("DMA-BUF descriptor is missing object/layer metadata".to_string());
        }
        if descriptor
            .producer_drm_node
            .as_deref()
            .unwrap_or("")
            .is_empty()
        {
            return Err("DMA-BUF descriptor is missing its producer DRM node".to_string());
        }
        descriptor.validate(received.fds.len())?;
        match self.generation {
            Some(current) if descriptor.generation < current => {
                return Err("stale DMA-BUF surface generation".to_string());
            }
            Some(current) if descriptor.generation > current => {
                self.surfaces.clear();
                self.generation = Some(descriptor.generation);
            }
            None => self.generation = Some(descriptor.generation),
            _ => {}
        }
        if self
            .surfaces
            .get(&descriptor.buffer_id)
            .and_then(|surface| surface.lease.as_ref())
            .is_some()
        {
            return Err("cannot replace a consumer-owned DMA-BUF surface".to_string());
        }
        self.surfaces.insert(
            descriptor.buffer_id,
            RegisteredSurface {
                descriptor,
                object_fds: received.fds,
                lease: None,
            },
        );
        Ok(())
    }

    pub fn begin_frame(
        &mut self,
        received: ReceivedSurfaceMessage,
    ) -> Result<DmabufFrameView<'_>, String> {
        let SurfaceChannelMessage::FrameReady { frame } = received.message else {
            return Err("expected a frame-ready message".to_string());
        };
        frame.validate(received.fds.len())?;
        if self.generation != Some(frame.generation) {
            return Err("stale DMA-BUF frame generation".to_string());
        }
        let surface = self
            .surfaces
            .get_mut(&frame.buffer_id)
            .ok_or_else(|| "DMA-BUF frame references an unknown surface".to_string())?;
        if surface.descriptor.generation != frame.generation {
            return Err("DMA-BUF frame and surface generations differ".to_string());
        }
        if surface.lease.is_some() {
            return Err("DMA-BUF surface already has a consumer lease".to_string());
        }
        let mut fds = received.fds;
        let fence_fd = if frame.has_native_fence {
            fds.pop()
        } else {
            None
        };
        surface.lease = Some(ActiveLease {
            sequence: frame.sequence,
            fence_fd,
        });
        let lease = surface.lease.as_ref().expect("lease was just installed");
        Ok(DmabufFrameView {
            descriptor: &surface.descriptor,
            frame,
            object_fds: surface.object_fds.iter().map(AsRawFd::as_raw_fd).collect(),
            fence_fd: lease.fence_fd.as_ref().map(AsRawFd::as_raw_fd),
        })
    }

    pub fn complete_frame(&mut self, release: SurfaceFrameRelease) -> Result<(), String> {
        if self.generation != Some(release.generation) {
            return Err("stale DMA-BUF release generation".to_string());
        }
        let surface = self
            .surfaces
            .get_mut(&release.buffer_id)
            .ok_or_else(|| "DMA-BUF release references an unknown surface".to_string())?;
        let lease = surface
            .lease
            .as_ref()
            .ok_or_else(|| "DMA-BUF surface is not consumer-owned".to_string())?;
        if lease.sequence != release.sequence {
            return Err("DMA-BUF release sequence does not match the active lease".to_string());
        }
        surface.lease = None;
        Ok(())
    }

    pub fn reset(&mut self) {
        self.surfaces.clear();
        self.generation = None;
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use super::*;
    use crate::native_video::surface_protocol::{
        DmabufLayer, DmabufLayout, DmabufObject, DmabufPlane, SurfacePlane,
    };

    fn descriptor(generation: u64, buffer_id: u32) -> SurfaceDescriptor {
        SurfaceDescriptor {
            generation,
            buffer_id,
            width: 1280,
            height: 720,
            sample_aspect_ratio: 1.0,
            rotation_degrees: 0,
            fourcc: u32::from_le_bytes(*b"R8  "),
            modifier: 17,
            handle_kind: SurfaceHandleKind::DmaBuf,
            reusable_dmabuf: false,
            producer_drm_node: Some("/dev/dri/renderD128".to_string()),
            color_space: None,
            color_range: None,
            chroma_location: None,
            planes: vec![
                SurfacePlane {
                    stride: 1280,
                    offset: 0,
                },
                SurfacePlane {
                    stride: 1280,
                    offset: 0,
                },
            ],
            dmabuf: Some(DmabufLayout {
                objects: vec![
                    DmabufObject {
                        size: 1280 * 720,
                        modifier: 17,
                    },
                    DmabufObject {
                        size: 1280 * 360,
                        modifier: 17,
                    },
                ],
                layers: vec![
                    DmabufLayer {
                        fourcc: u32::from_le_bytes(*b"R8  "),
                        width: 1280,
                        height: 720,
                        planes: vec![DmabufPlane {
                            object_index: 0,
                            stride: 1280,
                            offset: 0,
                        }],
                    },
                    DmabufLayer {
                        fourcc: u32::from_le_bytes(*b"RG88"),
                        width: 640,
                        height: 360,
                        planes: vec![DmabufPlane {
                            object_index: 1,
                            stride: 1280,
                            offset: 0,
                        }],
                    },
                ],
            }),
        }
    }

    fn objects() -> Vec<OwnedFd> {
        vec![
            File::open("/dev/null").unwrap().into(),
            File::open("/dev/null").unwrap().into(),
        ]
    }

    #[test]
    fn retains_objects_until_exact_frame_completion() {
        let mut consumer = DmabufSurfaceConsumer::default();
        consumer
            .register(ReceivedSurfaceMessage {
                message: SurfaceChannelMessage::SurfaceCreated {
                    descriptor: descriptor(7, 1),
                },
                fds: objects(),
            })
            .unwrap();
        let ready = SurfaceFrameReady {
            generation: 7,
            buffer_id: 1,
            sequence: 19,
            pts_seconds: 1.5,
            has_native_fence: false,
        };
        {
            let view = consumer
                .begin_frame(ReceivedSurfaceMessage {
                    message: SurfaceChannelMessage::FrameReady { frame: ready },
                    fds: vec![],
                })
                .unwrap();
            assert_eq!(view.object_fds.len(), 2);
            assert_eq!(view.descriptor.dmabuf.as_ref().unwrap().layers.len(), 2);
        }
        assert!(consumer
            .complete_frame(SurfaceFrameRelease {
                generation: 7,
                buffer_id: 1,
                sequence: 20,
            })
            .is_err());
        consumer
            .complete_frame(SurfaceFrameRelease {
                generation: 7,
                buffer_id: 1,
                sequence: 19,
            })
            .unwrap();
    }

    #[test]
    fn refuses_replacement_while_consumer_owns_surface() {
        let mut consumer = DmabufSurfaceConsumer::default();
        consumer
            .register(ReceivedSurfaceMessage {
                message: SurfaceChannelMessage::SurfaceCreated {
                    descriptor: descriptor(4, 0),
                },
                fds: objects(),
            })
            .unwrap();
        let _view = consumer
            .begin_frame(ReceivedSurfaceMessage {
                message: SurfaceChannelMessage::FrameReady {
                    frame: SurfaceFrameReady {
                        generation: 4,
                        buffer_id: 0,
                        sequence: 1,
                        pts_seconds: 0.0,
                        has_native_fence: false,
                    },
                },
                fds: vec![],
            })
            .unwrap();
        assert!(consumer
            .register(ReceivedSurfaceMessage {
                message: SurfaceChannelMessage::SurfaceCreated {
                    descriptor: descriptor(4, 0),
                },
                fds: objects(),
            })
            .is_err());
    }
}
