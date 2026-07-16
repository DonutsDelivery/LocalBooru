use std::os::fd::RawFd;

use super::surface_protocol::{
    SurfaceChromaLocation, SurfaceColorRange, SurfaceColorSpace, SurfaceDescriptor,
};

const EGL_NONE: i32 = 0x3038;
const EGL_WIDTH: i32 = 0x3057;
const EGL_HEIGHT: i32 = 0x3056;
const EGL_LINUX_DRM_FOURCC_EXT: i32 = 0x3271;
const EGL_DMA_BUF_PLANE_FD_EXT: [i32; 4] = [0x3272, 0x3275, 0x3278, 0x3440];
const EGL_DMA_BUF_PLANE_OFFSET_EXT: [i32; 4] = [0x3273, 0x3276, 0x3279, 0x3441];
const EGL_DMA_BUF_PLANE_PITCH_EXT: [i32; 4] = [0x3274, 0x3277, 0x327a, 0x3442];
const EGL_DMA_BUF_PLANE_MODIFIER_LO_EXT: [i32; 4] = [0x3443, 0x3445, 0x3447, 0x3449];
const EGL_DMA_BUF_PLANE_MODIFIER_HI_EXT: [i32; 4] = [0x3444, 0x3446, 0x3448, 0x344a];
const DRM_FORMAT_MOD_INVALID: u64 = u64::MAX;
const DRM_FORMAT_R8: u32 = u32::from_le_bytes(*b"R8  ");
const DRM_FORMAT_RG88: u32 = u32::from_le_bytes(*b"RG88");
const DRM_FORMAT_NV12: u32 = u32::from_le_bytes(*b"NV12");
const EGL_YUV_COLOR_SPACE_HINT_EXT: i32 = 0x327b;
const EGL_SAMPLE_RANGE_HINT_EXT: i32 = 0x327c;
const EGL_YUV_CHROMA_HORIZONTAL_SITING_HINT_EXT: i32 = 0x327d;
const EGL_YUV_CHROMA_VERTICAL_SITING_HINT_EXT: i32 = 0x327e;
const EGL_ITU_REC709_EXT: i32 = 0x3280;
const EGL_ITU_REC601_EXT: i32 = 0x327f;
const EGL_ITU_REC2020_EXT: i32 = 0x3281;
const EGL_YUV_FULL_RANGE_EXT: i32 = 0x3282;
const EGL_YUV_NARROW_RANGE_EXT: i32 = 0x3283;
const EGL_YUV_CHROMA_SITING_0_EXT: i32 = 0x3284;
const EGL_YUV_CHROMA_SITING_0_5_EXT: i32 = 0x3285;

pub struct EglDmabufLayerAttributes {
    pub width: u32,
    pub height: u32,
    pub fourcc: u32,
    pub attributes: Vec<i32>,
}

pub fn validate_egl_extensions(
    extensions: &str,
    has_explicit_modifier: bool,
) -> Result<(), String> {
    let has = |name: &str| extensions.split_ascii_whitespace().any(|item| item == name);
    if !has("EGL_EXT_image_dma_buf_import") {
        return Err("EGL_EXT_image_dma_buf_import is unavailable".to_string());
    }
    if has_explicit_modifier && !has("EGL_EXT_image_dma_buf_import_modifiers") {
        return Err("EGL DMA-BUF modifier import is unavailable".to_string());
    }
    Ok(())
}

pub fn build_layer_attributes(
    descriptor: &SurfaceDescriptor,
    layer_index: usize,
    object_fds: &[RawFd],
) -> Result<EglDmabufLayerAttributes, String> {
    descriptor.validate(object_fds.len())?;
    let layout = descriptor
        .dmabuf
        .as_ref()
        .ok_or_else(|| "DMA-BUF layout is missing".to_string())?;
    let layer = layout
        .layers
        .get(layer_index)
        .ok_or_else(|| "DMA-BUF layer index is out of range".to_string())?;
    if layer.width > i32::MAX as u32 || layer.height > i32::MAX as u32 {
        return Err("DMA-BUF layer dimensions do not fit EGLint".to_string());
    }

    let mut attributes = vec![
        EGL_WIDTH,
        layer.width as i32,
        EGL_HEIGHT,
        layer.height as i32,
        EGL_LINUX_DRM_FOURCC_EXT,
        layer.fourcc as i32,
    ];
    for (plane_index, plane) in layer.planes.iter().enumerate() {
        let object_index = plane.object_index as usize;
        let object = layout
            .objects
            .get(object_index)
            .ok_or_else(|| "DMA-BUF plane references an unknown object".to_string())?;
        let fd = *object_fds
            .get(object_index)
            .ok_or_else(|| "DMA-BUF object descriptor is missing".to_string())?;
        if fd < 0 || plane.offset > i32::MAX as u32 || plane.stride > i32::MAX as u32 {
            return Err("DMA-BUF plane values do not fit EGLint".to_string());
        }
        attributes.extend_from_slice(&[
            EGL_DMA_BUF_PLANE_FD_EXT[plane_index],
            fd,
            EGL_DMA_BUF_PLANE_OFFSET_EXT[plane_index],
            plane.offset as i32,
            EGL_DMA_BUF_PLANE_PITCH_EXT[plane_index],
            plane.stride as i32,
        ]);
        if object.modifier != DRM_FORMAT_MOD_INVALID {
            attributes.extend_from_slice(&[
                EGL_DMA_BUF_PLANE_MODIFIER_LO_EXT[plane_index],
                object.modifier as u32 as i32,
                EGL_DMA_BUF_PLANE_MODIFIER_HI_EXT[plane_index],
                (object.modifier >> 32) as u32 as i32,
            ]);
        }
    }
    attributes.push(EGL_NONE);
    Ok(EglDmabufLayerAttributes {
        width: layer.width,
        height: layer.height,
        fourcc: layer.fourcc,
        attributes,
    })
}

pub fn build_nv12_attributes(
    descriptor: &SurfaceDescriptor,
    object_fds: &[RawFd],
) -> Result<EglDmabufLayerAttributes, String> {
    descriptor.validate(object_fds.len())?;
    let layout = descriptor
        .dmabuf
        .as_ref()
        .ok_or_else(|| "DMA-BUF layout is missing".to_string())?;
    if layout.layers.len() != 2
        || layout.layers[0].fourcc != DRM_FORMAT_R8
        || layout.layers[1].fourcc != DRM_FORMAT_RG88
        || layout.layers[0].planes.len() != 1
        || layout.layers[1].planes.len() != 1
    {
        return Err("DMA-BUF layout is not the supported two-plane NV12 form".to_string());
    }
    if descriptor.width > i32::MAX as u32 || descriptor.height > i32::MAX as u32 {
        return Err("DMA-BUF dimensions do not fit EGLint".to_string());
    }
    let mut attributes = vec![
        EGL_WIDTH,
        descriptor.width as i32,
        EGL_HEIGHT,
        descriptor.height as i32,
        EGL_LINUX_DRM_FOURCC_EXT,
        DRM_FORMAT_NV12 as i32,
    ];
    for (plane_index, layer) in layout.layers.iter().enumerate() {
        let plane = &layer.planes[0];
        let object_index = plane.object_index as usize;
        let object = layout
            .objects
            .get(object_index)
            .ok_or_else(|| "DMA-BUF plane references an unknown object".to_string())?;
        let fd = *object_fds
            .get(object_index)
            .ok_or_else(|| "DMA-BUF object descriptor is missing".to_string())?;
        if fd < 0 || plane.offset > i32::MAX as u32 || plane.stride > i32::MAX as u32 {
            return Err("DMA-BUF plane values do not fit EGLint".to_string());
        }
        attributes.extend_from_slice(&[
            EGL_DMA_BUF_PLANE_FD_EXT[plane_index],
            fd,
            EGL_DMA_BUF_PLANE_OFFSET_EXT[plane_index],
            plane.offset as i32,
            EGL_DMA_BUF_PLANE_PITCH_EXT[plane_index],
            plane.stride as i32,
        ]);
        if object.modifier != DRM_FORMAT_MOD_INVALID {
            attributes.extend_from_slice(&[
                EGL_DMA_BUF_PLANE_MODIFIER_LO_EXT[plane_index],
                object.modifier as u32 as i32,
                EGL_DMA_BUF_PLANE_MODIFIER_HI_EXT[plane_index],
                (object.modifier >> 32) as u32 as i32,
            ]);
        }
    }
    let color_space = match descriptor.color_space.unwrap_or_else(|| {
        if descriptor.width >= 1280 || descriptor.height > 576 {
            SurfaceColorSpace::Bt709
        } else {
            SurfaceColorSpace::Bt601
        }
    }) {
        SurfaceColorSpace::Bt601 => EGL_ITU_REC601_EXT,
        SurfaceColorSpace::Bt709 => EGL_ITU_REC709_EXT,
        SurfaceColorSpace::Bt2020 => EGL_ITU_REC2020_EXT,
    };
    let color_range = match descriptor.color_range.unwrap_or(SurfaceColorRange::Narrow) {
        SurfaceColorRange::Narrow => EGL_YUV_NARROW_RANGE_EXT,
        SurfaceColorRange::Full => EGL_YUV_FULL_RANGE_EXT,
    };
    let (chroma_horizontal, chroma_vertical) = match descriptor
        .chroma_location
        .unwrap_or(SurfaceChromaLocation::Center)
    {
        SurfaceChromaLocation::Left => (EGL_YUV_CHROMA_SITING_0_EXT, EGL_YUV_CHROMA_SITING_0_5_EXT),
        SurfaceChromaLocation::TopLeft => {
            (EGL_YUV_CHROMA_SITING_0_EXT, EGL_YUV_CHROMA_SITING_0_EXT)
        }
        SurfaceChromaLocation::Top => (EGL_YUV_CHROMA_SITING_0_5_EXT, EGL_YUV_CHROMA_SITING_0_EXT),
        SurfaceChromaLocation::Center => {
            (EGL_YUV_CHROMA_SITING_0_5_EXT, EGL_YUV_CHROMA_SITING_0_5_EXT)
        }
    };
    attributes.extend_from_slice(&[
        EGL_YUV_COLOR_SPACE_HINT_EXT,
        color_space,
        EGL_SAMPLE_RANGE_HINT_EXT,
        color_range,
        EGL_YUV_CHROMA_HORIZONTAL_SITING_HINT_EXT,
        chroma_horizontal,
        EGL_YUV_CHROMA_VERTICAL_SITING_HINT_EXT,
        chroma_vertical,
        EGL_NONE,
    ]);
    Ok(EglDmabufLayerAttributes {
        width: descriptor.width,
        height: descriptor.height,
        fourcc: DRM_FORMAT_NV12,
        attributes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_video::surface_protocol::{
        DmabufLayer, DmabufLayout, DmabufObject, DmabufPlane, SurfaceHandleKind, SurfacePlane,
    };

    fn descriptor() -> SurfaceDescriptor {
        SurfaceDescriptor {
            generation: 2,
            buffer_id: 0,
            width: 1280,
            height: 720,
            sample_aspect_ratio: 1.0,
            rotation_degrees: 0,
            fourcc: u32::from_le_bytes(*b"R8  "),
            modifier: 0x0300_0000_0060_6014,
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
                        size: 983_040,
                        modifier: 0x0300_0000_0060_6014,
                    },
                    DmabufObject {
                        size: 524_288,
                        modifier: 0x0300_0000_0060_6014,
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

    #[test]
    fn builds_modifier_aware_chroma_layer_attributes() {
        let attributes = build_layer_attributes(&descriptor(), 1, &[31, 32]).unwrap();
        assert_eq!(attributes.width, 640);
        assert_eq!(attributes.height, 360);
        assert_eq!(attributes.fourcc, u32::from_le_bytes(*b"RG88"));
        assert!(attributes
            .attributes
            .windows(2)
            .any(|pair| pair == [EGL_DMA_BUF_PLANE_FD_EXT[0], 32]));
        assert!(attributes
            .attributes
            .windows(2)
            .any(|pair| pair == [EGL_DMA_BUF_PLANE_MODIFIER_LO_EXT[0], 0x0060_6014]));
        assert_eq!(attributes.attributes.last(), Some(&EGL_NONE));
    }

    #[test]
    fn builds_composite_nv12_external_image_attributes() {
        let attributes = build_nv12_attributes(&descriptor(), &[31, 32]).unwrap();
        assert_eq!(attributes.width, 1280);
        assert_eq!(attributes.height, 720);
        assert_eq!(attributes.fourcc, DRM_FORMAT_NV12);
        assert!(attributes
            .attributes
            .windows(2)
            .any(|pair| pair == [EGL_DMA_BUF_PLANE_FD_EXT[0], 31]));
        assert!(attributes
            .attributes
            .windows(2)
            .any(|pair| pair == [EGL_DMA_BUF_PLANE_FD_EXT[1], 32]));
        assert!(attributes
            .attributes
            .windows(2)
            .any(|pair| pair == [EGL_YUV_COLOR_SPACE_HINT_EXT, EGL_ITU_REC709_EXT]));
        assert_eq!(attributes.attributes.last(), Some(&EGL_NONE));
    }

    #[test]
    fn maps_frame_color_metadata_to_egl_hints() {
        let mut descriptor = descriptor();
        descriptor.color_space = Some(SurfaceColorSpace::Bt2020);
        descriptor.color_range = Some(SurfaceColorRange::Full);
        descriptor.chroma_location = Some(SurfaceChromaLocation::TopLeft);
        let attributes = build_nv12_attributes(&descriptor, &[31, 32]).unwrap();
        for expected in [
            [EGL_YUV_COLOR_SPACE_HINT_EXT, EGL_ITU_REC2020_EXT],
            [EGL_SAMPLE_RANGE_HINT_EXT, EGL_YUV_FULL_RANGE_EXT],
            [
                EGL_YUV_CHROMA_HORIZONTAL_SITING_HINT_EXT,
                EGL_YUV_CHROMA_SITING_0_EXT,
            ],
            [
                EGL_YUV_CHROMA_VERTICAL_SITING_HINT_EXT,
                EGL_YUV_CHROMA_SITING_0_EXT,
            ],
        ] {
            assert!(attributes
                .attributes
                .windows(2)
                .any(|pair| pair == expected));
        }
    }

    #[test]
    fn requires_modifier_extension_only_for_explicit_modifiers() {
        assert!(validate_egl_extensions("EGL_EXT_image_dma_buf_import", false).is_ok());
        assert!(validate_egl_extensions("EGL_EXT_image_dma_buf_import", true).is_err());
        assert!(validate_egl_extensions(
            "EGL_EXT_image_dma_buf_import EGL_EXT_image_dma_buf_import_modifiers",
            true
        )
        .is_ok());
    }
}
