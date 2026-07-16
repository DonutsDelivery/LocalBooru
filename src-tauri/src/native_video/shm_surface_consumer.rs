use std::collections::HashMap;
use std::io;
use std::os::fd::{AsRawFd, OwnedFd};
use std::ptr::NonNull;
use std::rc::Rc;
use std::slice;

use super::surface_channel::{ReceivedSurfaceMessage, SurfaceChannelMessage};
use super::surface_protocol::{
    SurfaceDescriptor, SurfaceFrameReady, SurfaceFrameRelease, SurfaceHandleKind, DRM_FORMAT_YUV420,
};

struct MappedSurface {
    descriptor: SurfaceDescriptor,
    _fd: OwnedFd,
    mapping: NonNull<u8>,
    mapping_len: usize,
}

impl Drop for MappedSurface {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.mapping.as_ptr().cast(), self.mapping_len);
        }
    }
}

pub struct ShmFrameView {
    surface: Rc<MappedSurface>,
    pub frame: SurfaceFrameReady,
}

impl ShmFrameView {
    pub fn width(&self) -> u32 {
        self.surface.descriptor.width
    }

    pub fn height(&self) -> u32 {
        self.surface.descriptor.height
    }

    pub fn stride(&self) -> u32 {
        self.surface.descriptor.planes[0].stride
    }

    pub fn bytes(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.surface.mapping.as_ptr(), self.surface.mapping_len) }
    }

    pub fn descriptor(&self) -> &SurfaceDescriptor {
        &self.surface.descriptor
    }

    pub fn release(&self) -> SurfaceFrameRelease {
        SurfaceFrameRelease {
            generation: self.frame.generation,
            buffer_id: self.frame.buffer_id,
            sequence: self.frame.sequence,
        }
    }
}

#[derive(Default)]
pub struct ShmSurfaceConsumer {
    generation: Option<u64>,
    surfaces: HashMap<u32, Rc<MappedSurface>>,
}

impl ShmSurfaceConsumer {
    pub fn register(&mut self, received: ReceivedSurfaceMessage) -> Result<(), String> {
        let SurfaceChannelMessage::SurfaceCreated { descriptor } = received.message else {
            return Err("expected surface_created message".to_string());
        };
        descriptor.validate(received.fds.len())?;
        if descriptor.handle_kind != SurfaceHandleKind::SharedMemory {
            return Err("DMA-BUF descriptors require the GPU importer".to_string());
        }
        let [fd]: [OwnedFd; 1] = received
            .fds
            .try_into()
            .map_err(|_| "shared-memory surface requires exactly one fd".to_string())?;
        let mapping_len = descriptor
            .planes
            .iter()
            .enumerate()
            .filter_map(|(index, plane)| {
                let plane_height = if descriptor.fourcc == DRM_FORMAT_YUV420 && index > 0 {
                    descriptor.height / 2
                } else {
                    descriptor.height
                };
                usize::try_from(plane.offset).ok()?.checked_add(
                    usize::try_from(plane.stride)
                        .ok()?
                        .checked_mul(usize::try_from(plane_height).ok()?)?,
                )
            })
            .max()
            .ok_or_else(|| "shared-memory surface size overflow".to_string())?;

        let mut stat = std::mem::MaybeUninit::<libc::stat>::uninit();
        if unsafe { libc::fstat(fd.as_raw_fd(), stat.as_mut_ptr()) } != 0 {
            return Err(format!(
                "failed to inspect shared-memory surface: {}",
                io::Error::last_os_error()
            ));
        }
        let file_len = unsafe { stat.assume_init().st_size };
        if file_len < 0 || usize::try_from(file_len).unwrap_or(0) < mapping_len {
            return Err("shared-memory fd is smaller than its descriptor".to_string());
        }
        let mapping = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                mapping_len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd.as_raw_fd(),
                0,
            )
        };
        if mapping == libc::MAP_FAILED {
            return Err(format!(
                "failed to map shared-memory surface: {}",
                io::Error::last_os_error()
            ));
        }
        let mapping = NonNull::new(mapping.cast::<u8>())
            .ok_or_else(|| "shared-memory mapping returned null".to_string())?;

        if self.generation != Some(descriptor.generation) {
            self.surfaces.clear();
            self.generation = Some(descriptor.generation);
        }
        self.surfaces.insert(
            descriptor.buffer_id,
            Rc::new(MappedSurface {
                descriptor,
                _fd: fd,
                mapping,
                mapping_len,
            }),
        );
        Ok(())
    }

    pub fn frame(&self, received: ReceivedSurfaceMessage) -> Result<ShmFrameView, String> {
        let SurfaceChannelMessage::FrameReady { frame } = received.message else {
            return Err("expected frame_ready message".to_string());
        };
        frame.validate(received.fds.len())?;
        if frame.has_native_fence {
            return Err("native fences require the DMA-BUF importer".to_string());
        }
        if self.generation != Some(frame.generation) {
            return Err("stale shared-memory frame generation".to_string());
        }
        let surface = self
            .surfaces
            .get(&frame.buffer_id)
            .ok_or_else(|| "frame references an unknown shared-memory buffer".to_string())?;
        Ok(ShmFrameView {
            surface: Rc::clone(surface),
            frame,
        })
    }

    pub fn reset(&mut self) {
        self.surfaces.clear();
        self.generation = None;
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::CString;
    use std::os::fd::{FromRawFd, OwnedFd};

    use super::*;
    use crate::native_video::surface_protocol::SurfacePlane;

    fn memfd(bytes: &[u8]) -> OwnedFd {
        let name = CString::new("localbooru-shm-consumer-test").unwrap();
        let raw = unsafe { libc::memfd_create(name.as_ptr(), libc::MFD_CLOEXEC) };
        assert!(raw >= 0);
        assert_eq!(
            unsafe { libc::ftruncate(raw, bytes.len() as libc::off_t) },
            0
        );
        assert_eq!(
            unsafe { libc::pwrite(raw, bytes.as_ptr().cast(), bytes.len(), 0) },
            bytes.len() as isize
        );
        unsafe { OwnedFd::from_raw_fd(raw) }
    }

    #[test]
    fn maps_announced_surface_and_rejects_stale_frames() {
        let fd = memfd(&[1, 2, 3, 4, 5, 6, 7, 8]);
        let descriptor = SurfaceDescriptor {
            generation: 8,
            buffer_id: 2,
            width: 2,
            height: 1,
            sample_aspect_ratio: 1.0,
            rotation_degrees: 0,
            fourcc: 0x3432_4241,
            modifier: 0,
            handle_kind: SurfaceHandleKind::SharedMemory,
            reusable_dmabuf: false,
            producer_drm_node: None,
            color_space: None,
            color_range: None,
            chroma_location: None,
            planes: vec![SurfacePlane {
                stride: 8,
                offset: 0,
            }],
            dmabuf: None,
        };
        let mut consumer = ShmSurfaceConsumer::default();
        consumer
            .register(ReceivedSurfaceMessage {
                message: SurfaceChannelMessage::SurfaceCreated { descriptor },
                fds: vec![fd],
            })
            .unwrap();

        let view = consumer
            .frame(ReceivedSurfaceMessage {
                message: SurfaceChannelMessage::FrameReady {
                    frame: SurfaceFrameReady {
                        generation: 8,
                        buffer_id: 2,
                        sequence: 4,
                        pts_seconds: 1.5,
                        has_native_fence: false,
                    },
                },
                fds: vec![],
            })
            .unwrap();
        assert_eq!(view.bytes(), &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(view.release().sequence, 4);
        consumer.reset();
        assert_eq!(view.bytes(), &[1, 2, 3, 4, 5, 6, 7, 8]);
        drop(view);

        let error = consumer
            .frame(ReceivedSurfaceMessage {
                message: SurfaceChannelMessage::FrameReady {
                    frame: SurfaceFrameReady {
                        generation: 7,
                        buffer_id: 2,
                        sequence: 5,
                        pts_seconds: 2.0,
                        has_native_fence: false,
                    },
                },
                fds: vec![],
            })
            .err()
            .unwrap();
        assert!(error.contains("stale"));
    }

    #[test]
    fn rejects_descriptor_larger_than_backing_fd() {
        let fd = memfd(&[0; 4]);
        let mut consumer = ShmSurfaceConsumer::default();
        let error = consumer
            .register(ReceivedSurfaceMessage {
                message: SurfaceChannelMessage::SurfaceCreated {
                    descriptor: SurfaceDescriptor {
                        generation: 1,
                        buffer_id: 0,
                        width: 2,
                        height: 1,
                        sample_aspect_ratio: 1.0,
                        rotation_degrees: 0,
                        fourcc: 0x3432_4241,
                        modifier: 0,
                        handle_kind: SurfaceHandleKind::SharedMemory,
                        reusable_dmabuf: false,
                        producer_drm_node: None,
                        color_space: None,
                        color_range: None,
                        chroma_location: None,
                        planes: vec![SurfacePlane {
                            stride: 8,
                            offset: 0,
                        }],
                        dmabuf: None,
                    },
                },
                fds: vec![fd],
            })
            .unwrap_err();
        assert!(error.contains("smaller"));
    }
}
