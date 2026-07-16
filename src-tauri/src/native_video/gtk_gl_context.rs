use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::os::fd::RawFd;

use super::egl_dmabuf_import::{build_nv12_attributes, validate_egl_extensions};
use super::surface_protocol::{SurfaceColorRange, SurfaceColorSpace, SurfaceDescriptor};

const EGL_EXTENSIONS: i32 = 0x3055;
const EGL_VENDOR: i32 = 0x3053;
const EGL_VERSION: i32 = 0x3054;
const EGL_LINUX_DMA_BUF_EXT: i32 = 0x3270;
const GL_COLOR_BUFFER_BIT: u32 = 0x0000_4000;
const GL_NO_ERROR: u32 = 0;
const GL_VERSION: u32 = 0x1f02;
const GL_VERTEX_SHADER: u32 = 0x8b31;
const GL_FRAGMENT_SHADER: u32 = 0x8b30;
const GL_COMPILE_STATUS: u32 = 0x8b81;
const GL_LINK_STATUS: u32 = 0x8b82;
const GL_INFO_LOG_LENGTH: u32 = 0x8b84;
const GL_ARRAY_BUFFER: u32 = 0x8892;
const GL_STATIC_DRAW: u32 = 0x88e4;
const GL_FLOAT: u32 = 0x1406;
const GL_TEXTURE0: u32 = 0x84c0;
const GL_TEXTURE_EXTERNAL_OES: u32 = 0x8d65;
const GL_TEXTURE_2D: u32 = 0x0de1;
const GL_RGBA: u32 = 0x1908;
const GL_LUMINANCE: u32 = 0x1909;
const GL_UNSIGNED_BYTE: u32 = 0x1401;
const GL_TEXTURE_MIN_FILTER: u32 = 0x2801;
const GL_TEXTURE_MAG_FILTER: u32 = 0x2800;
const GL_TEXTURE_WRAP_S: u32 = 0x2802;
const GL_TEXTURE_WRAP_T: u32 = 0x2803;
const GL_LINEAR: i32 = 0x2601;
const GL_CLAMP_TO_EDGE: i32 = 0x812f;
const GL_TRIANGLE_STRIP: u32 = 0x0005;
const GL_SYNC_GPU_COMMANDS_COMPLETE: u32 = 0x9117;
const GL_ALREADY_SIGNALED: u32 = 0x911a;
const GL_CONDITION_SATISFIED: u32 = 0x911c;
const GL_WAIT_FAILED: u32 = 0x911d;

#[link(name = "EGL")]
extern "C" {
    fn eglGetCurrentDisplay() -> *mut c_void;
    fn eglGetCurrentContext() -> *mut c_void;
    fn eglQueryString(display: *mut c_void, name: i32) -> *const c_char;
    fn eglGetError() -> u32;
    fn eglGetProcAddress(name: *const c_char) -> *const c_void;
}

#[link(name = "GLESv2")]
extern "C" {
    fn glClearColor(red: f32, green: f32, blue: f32, alpha: f32);
    fn glClear(mask: u32);
    fn glFlush();
    fn glFenceSync(condition: u32, flags: u32) -> *mut c_void;
    fn glClientWaitSync(sync: *mut c_void, flags: u32, timeout: u64) -> u32;
    fn glDeleteSync(sync: *mut c_void);
    fn glGetError() -> u32;
    fn glGetString(name: u32) -> *const u8;
    fn glCreateShader(shader_type: u32) -> u32;
    fn glShaderSource(shader: u32, count: i32, source: *const *const c_char, length: *const i32);
    fn glCompileShader(shader: u32);
    fn glGetShaderiv(shader: u32, pname: u32, value: *mut i32);
    fn glGetShaderInfoLog(shader: u32, size: i32, length: *mut i32, log: *mut c_char);
    fn glDeleteShader(shader: u32);
    fn glCreateProgram() -> u32;
    fn glAttachShader(program: u32, shader: u32);
    fn glLinkProgram(program: u32);
    fn glGetProgramiv(program: u32, pname: u32, value: *mut i32);
    fn glGetProgramInfoLog(program: u32, size: i32, length: *mut i32, log: *mut c_char);
    fn glDeleteProgram(program: u32);
    fn glGenBuffers(count: i32, buffers: *mut u32);
    fn glDeleteBuffers(count: i32, buffers: *const u32);

    fn glBindBuffer(target: u32, buffer: u32);
    fn glBufferData(target: u32, size: isize, data: *const c_void, usage: u32);
    fn glViewport(x: i32, y: i32, width: i32, height: i32);
    fn glUseProgram(program: u32);
    fn glGetUniformLocation(program: u32, name: *const c_char) -> i32;
    fn glUniform1i(location: i32, value: i32);
    fn glUniform4f(location: i32, x: f32, y: f32, z: f32, w: f32);
    fn glGetAttribLocation(program: u32, name: *const c_char) -> i32;
    fn glEnableVertexAttribArray(index: u32);
    fn glVertexAttribPointer(
        index: u32,
        size: i32,
        kind: u32,
        normalized: u8,
        stride: i32,
        pointer: *const c_void,
    );
    fn glGenTextures(count: i32, textures: *mut u32);
    fn glActiveTexture(texture: u32);
    fn glBindTexture(target: u32, texture: u32);
    fn glTexParameteri(target: u32, name: u32, value: i32);
    fn glTexImage2D(
        target: u32,
        level: i32,
        internal_format: i32,
        width: i32,
        height: i32,
        border: i32,
        format: u32,
        kind: u32,
        pixels: *const c_void,
    );
    fn glTexSubImage2D(
        target: u32,
        level: i32,
        xoffset: i32,
        yoffset: i32,
        width: i32,
        height: i32,
        format: u32,
        kind: u32,
        pixels: *const c_void,
    );
    fn glDrawArrays(mode: u32, first: i32, count: i32);
    fn glDeleteTextures(count: i32, textures: *const u32);
}

type EglCreateImageKhr =
    unsafe extern "C" fn(*mut c_void, *mut c_void, i32, *mut c_void, *const i32) -> *mut c_void;
type EglDestroyImageKhr = unsafe extern "C" fn(*mut c_void, *mut c_void) -> u32;
type GlEglImageTargetTexture2dOes = unsafe extern "C" fn(u32, *mut c_void);

unsafe fn egl_proc(name: &str) -> Result<*const c_void, String> {
    let name = CString::new(name).map_err(|error| error.to_string())?;
    let address = eglGetProcAddress(name.as_ptr());
    if address.is_null() {
        return Err(format!("{name:?} is unavailable in the GTK EGL context"));
    }
    Ok(address)
}

pub struct GpuFence {
    sync: *mut c_void,
    textures: Vec<u32>,
    egl_image: Option<(*mut c_void, *mut c_void, EglDestroyImageKhr)>,
}

impl GpuFence {
    pub fn is_signaled(&self) -> Result<bool, String> {
        let status = unsafe { glClientWaitSync(self.sync, 0, 0) };
        match status {
            GL_ALREADY_SIGNALED | GL_CONDITION_SATISFIED => Ok(true),
            GL_WAIT_FAILED => Err(format!(
                "GPU fence wait failed with GL error 0x{:x}",
                unsafe { glGetError() }
            )),
            _ => Ok(false),
        }
    }

    fn retain_egl_image(
        mut self,
        display: *mut c_void,
        image: *mut c_void,
        destroy: EglDestroyImageKhr,
    ) -> Self {
        self.egl_image = Some((display, image, destroy));
        self
    }
}

impl Drop for GpuFence {
    fn drop(&mut self) {
        unsafe {
            glDeleteSync(self.sync);
            if !self.textures.is_empty() {
                glDeleteTextures(self.textures.len() as i32, self.textures.as_ptr());
            }
            if let Some((display, image, destroy)) = self.egl_image.take() {
                destroy(display, image);
            }
        }
    }
}

struct ExternalImageRenderer {
    program: u32,
    vertex_buffer: u32,
    position: u32,
    image_target: GlEglImageTargetTexture2dOes,
}

impl Drop for ExternalImageRenderer {
    fn drop(&mut self) {
        unsafe {
            glDeleteBuffers(1, &self.vertex_buffer);
            glDeleteProgram(self.program);
        }
    }
}

struct CachedExternalImage {
    display: *mut c_void,
    image: *mut c_void,
    texture: u32,
    destroy: EglDestroyImageKhr,
}

unsafe fn destroy_cached_external_image(entry: CachedExternalImage) {
    glDeleteTextures(1, &entry.texture);
    (entry.destroy)(entry.display, entry.image);
}

thread_local! {
    static EXTERNAL_RENDERER: RefCell<Option<ExternalImageRenderer>> = const { RefCell::new(None) };
    static EXTERNAL_CONTEXT: Cell<usize> = const { Cell::new(0) };
    static EXTERNAL_IMAGE_CACHE: RefCell<HashMap<(usize, usize, u64, u32, Vec<(u64, u64)>), CachedExternalImage>> = RefCell::new(HashMap::new());
    static RGBA_RENDERER: RefCell<Option<RgbaRenderer>> = const { RefCell::new(None) };
    static YUV_RENDERER: RefCell<Option<YuvRenderer>> = const { RefCell::new(None) };
}

fn dmabuf_object_identity(object_fds: &[RawFd]) -> Result<Vec<(u64, u64)>, String> {
    object_fds
        .iter()
        .map(|fd| {
            let mut metadata = std::mem::MaybeUninit::<libc::stat>::zeroed();
            if unsafe { libc::fstat(*fd, metadata.as_mut_ptr()) } != 0 {
                return Err(format!(
                    "failed to identify reusable DMA-BUF object: {}",
                    std::io::Error::last_os_error()
                ));
            }
            let metadata = unsafe { metadata.assume_init() };
            Ok((metadata.st_dev as u64, metadata.st_ino as u64))
        })
        .collect()
}

unsafe fn apply_uv_transform(program: u32, crop: [f32; 4], rotation_quadrants: u8) {
    let crop_name = CString::new("uv_crop").unwrap();
    glUniform4f(
        glGetUniformLocation(program, crop_name.as_ptr()),
        crop[0],
        crop[1],
        crop[2],
        crop[3],
    );
    let rotation_name = CString::new("uv_rotation").unwrap();
    glUniform1i(
        glGetUniformLocation(program, rotation_name.as_ptr()),
        i32::from(rotation_quadrants % 4),
    );
}

struct RgbaRenderer {
    program: u32,
    vertex_buffer: u32,
    position: u32,
}

impl RgbaRenderer {
    unsafe fn create() -> Result<Self, String> {
        let vertex = compile_shader(
            GL_VERTEX_SHADER,
            "#ifdef GL_ES\nprecision mediump float;\n#endif\nattribute vec2 position; varying vec2 texture_coordinate; void main() { gl_Position = vec4(position, 0.0, 1.0); texture_coordinate = vec2((position.x + 1.0) * 0.5, 1.0 - (position.y + 1.0) * 0.5); }",
        )?;
        let fragment = compile_shader(
            GL_FRAGMENT_SHADER,
            "#ifdef GL_ES\nprecision mediump float;\n#endif\nuniform sampler2D video_texture; uniform vec4 uv_crop; uniform int uv_rotation; varying vec2 texture_coordinate; vec2 source_uv(vec2 p) { vec2 q = mix(uv_crop.xy, uv_crop.zw, p); if (uv_rotation == 1) return vec2(q.y, 1.0-q.x); if (uv_rotation == 2) return vec2(1.0-q.x, 1.0-q.y); if (uv_rotation == 3) return vec2(1.0-q.y, q.x); return q; } void main() { gl_FragColor = texture2D(video_texture, source_uv(texture_coordinate)); }",
        )?;
        let program = glCreateProgram();
        glAttachShader(program, vertex);
        glAttachShader(program, fragment);
        glLinkProgram(program);
        glDeleteShader(vertex);
        glDeleteShader(fragment);
        let mut linked = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &mut linked);
        if linked == 0 {
            glDeleteProgram(program);
            return Err("RGBA texture shader link failed".to_string());
        }
        let vertices: [f32; 8] = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let mut vertex_buffer = 0;
        glGenBuffers(1, &mut vertex_buffer);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
        glBufferData(
            GL_ARRAY_BUFFER,
            std::mem::size_of_val(&vertices) as isize,
            vertices.as_ptr().cast(),
            GL_STATIC_DRAW,
        );
        glUseProgram(program);
        let sampler = CString::new("video_texture").unwrap();
        glUniform1i(glGetUniformLocation(program, sampler.as_ptr()), 0);
        let position_name = CString::new("position").unwrap();
        let position = glGetAttribLocation(program, position_name.as_ptr());
        if position < 0 {
            glDeleteProgram(program);
            return Err("RGBA texture shader position input is missing".to_string());
        }
        Ok(Self {
            program,
            vertex_buffer,
            position: position as u32,
        })
    }

    unsafe fn draw(
        &self,
        rgba: &[u8],
        source_width: i32,
        source_height: i32,
        viewport_x: i32,
        viewport_y: i32,
        viewport_width: i32,
        viewport_height: i32,
        uv_crop: [f32; 4],
        rotation_quadrants: u8,
    ) -> Result<GpuFence, String> {
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);
        glViewport(
            viewport_x,
            viewport_y,
            viewport_width.max(1),
            viewport_height.max(1),
        );
        glUseProgram(self.program);
        apply_uv_transform(self.program, uv_crop, rotation_quadrants);
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer);
        glEnableVertexAttribArray(self.position);
        glVertexAttribPointer(self.position, 2, GL_FLOAT, 0, 0, std::ptr::null());
        let mut texture = 0;
        glGenTextures(1, &mut texture);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA as i32,
            source_width,
            source_height,
            0,
            GL_RGBA,
            GL_UNSIGNED_BYTE,
            rgba.as_ptr().cast(),
        );
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        let fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        if fence.is_null() {
            glDeleteTextures(1, &texture);
            return Err(format!(
                "RGBA glFenceSync failed with GL error 0x{:x}",
                glGetError()
            ));
        }
        glFlush();
        let error = glGetError();
        if error != GL_NO_ERROR {
            glDeleteSync(fence);
            glDeleteTextures(1, &texture);
            return Err(format!(
                "RGBA texture draw failed with GL error 0x{error:x}"
            ));
        }
        Ok(GpuFence {
            sync: fence,
            textures: vec![texture],
            egl_image: None,
        })
    }
}

struct YuvRenderer {
    program: u32,
    vertex_buffer: u32,
    position: u32,
    texture_sets: RefCell<HashMap<u32, ([u32; 3], i32, i32)>>,
}

impl YuvRenderer {
    unsafe fn create() -> Result<Self, String> {
        let vertex = compile_shader(
            GL_VERTEX_SHADER,
            "#ifdef GL_ES\nprecision mediump float;\n#endif\nattribute vec2 position; varying vec2 texture_coordinate; void main() { gl_Position = vec4(position, 0.0, 1.0); texture_coordinate = vec2((position.x + 1.0) * 0.5, 1.0 - (position.y + 1.0) * 0.5); }",
        )?;
        let fragment = compile_shader(
            GL_FRAGMENT_SHADER,
            "#ifdef GL_ES\nprecision mediump float;\n#endif\nuniform sampler2D y_texture; uniform sampler2D u_texture; uniform sampler2D v_texture; uniform vec4 uv_crop; uniform int uv_rotation; uniform int yuv_matrix; uniform int yuv_full_range; varying vec2 texture_coordinate; vec2 source_uv(vec2 p) { vec2 q = mix(uv_crop.xy, uv_crop.zw, p); if (uv_rotation == 1) return vec2(q.y, 1.0-q.x); if (uv_rotation == 2) return vec2(1.0-q.x, 1.0-q.y); if (uv_rotation == 3) return vec2(1.0-q.y, q.x); return q; } void main() { vec2 uv = source_uv(texture_coordinate); float raw_y = texture2D(y_texture, uv).r; float y = yuv_full_range == 1 ? raw_y : 1.16438356 * (raw_y - 0.06274510); float u = texture2D(u_texture, uv).r - 0.5; float v = texture2D(v_texture, uv).r - 0.5; float rv; float gu; float gv; float bu; if (yuv_matrix == 0) { rv = yuv_full_range == 1 ? 1.402 : 1.59602715; gu = yuv_full_range == 1 ? -0.344136 : -0.39176160; gv = yuv_full_range == 1 ? -0.714136 : -0.81296765; bu = yuv_full_range == 1 ? 1.772 : 2.01723214; } else if (yuv_matrix == 2) { rv = yuv_full_range == 1 ? 1.474600 : 1.67867411; gu = yuv_full_range == 1 ? -0.164553 : -0.18732610; gv = yuv_full_range == 1 ? -0.571353 : -0.65042432; bu = yuv_full_range == 1 ? 1.881400 : 2.14177232; } else { rv = yuv_full_range == 1 ? 1.574800 : 1.79274107; gu = yuv_full_range == 1 ? -0.187324 : -0.21324861; gv = yuv_full_range == 1 ? -0.468124 : -0.53290933; bu = yuv_full_range == 1 ? 1.855600 : 2.11240179; } gl_FragColor = vec4(y + rv*v, y + gu*u + gv*v, y + bu*u, 1.0); }",
        )?;
        let program = glCreateProgram();
        glAttachShader(program, vertex);
        glAttachShader(program, fragment);
        glLinkProgram(program);
        glDeleteShader(vertex);
        glDeleteShader(fragment);
        let mut linked = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &mut linked);
        if linked == 0 {
            glDeleteProgram(program);
            return Err("YUV texture shader link failed".to_string());
        }
        let vertices: [f32; 8] = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let mut vertex_buffer = 0;
        glGenBuffers(1, &mut vertex_buffer);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
        glBufferData(
            GL_ARRAY_BUFFER,
            std::mem::size_of_val(&vertices) as isize,
            vertices.as_ptr().cast(),
            GL_STATIC_DRAW,
        );
        glUseProgram(program);
        for (name, unit) in [("y_texture", 0), ("u_texture", 1), ("v_texture", 2)] {
            let name = CString::new(name).unwrap();
            glUniform1i(glGetUniformLocation(program, name.as_ptr()), unit);
        }
        let position_name = CString::new("position").unwrap();
        let position = glGetAttribLocation(program, position_name.as_ptr());
        if position < 0 {
            glDeleteProgram(program);
            return Err("YUV texture shader position input is missing".to_string());
        }
        Ok(Self {
            program,
            vertex_buffer,
            position: position as u32,
            texture_sets: RefCell::new(HashMap::new()),
        })
    }

    unsafe fn draw(
        &self,
        buffer_id: u32,
        yuv: &[u8],
        width: i32,
        height: i32,
        offsets: [usize; 3],
        viewport_x: i32,
        viewport_y: i32,
        viewport_width: i32,
        viewport_height: i32,
        uv_crop: [f32; 4],
        rotation_quadrants: u8,
        color_space: Option<SurfaceColorSpace>,
        color_range: Option<SurfaceColorRange>,
    ) -> Result<GpuFence, String> {
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);
        glViewport(
            viewport_x,
            viewport_y,
            viewport_width.max(1),
            viewport_height.max(1),
        );
        glUseProgram(self.program);
        apply_uv_transform(self.program, uv_crop, rotation_quadrants);
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer);
        glEnableVertexAttribArray(self.position);
        glVertexAttribPointer(self.position, 2, GL_FLOAT, 0, 0, std::ptr::null());
        let mut texture_sets = self.texture_sets.borrow_mut();
        let matrix = match color_space.unwrap_or(SurfaceColorSpace::Bt709) {
            SurfaceColorSpace::Bt601 => 0,
            SurfaceColorSpace::Bt709 => 1,
            SurfaceColorSpace::Bt2020 => 2,
        };
        glUniform1i(
            glGetUniformLocation(self.program, b"yuv_matrix\0".as_ptr().cast()),
            matrix,
        );
        glUniform1i(
            glGetUniformLocation(self.program, b"yuv_full_range\0".as_ptr().cast()),
            i32::from(matches!(color_range, Some(SurfaceColorRange::Full))),
        );
        let needs_allocation = texture_sets
            .get(&buffer_id)
            .map(|(_, texture_width, texture_height)| {
                *texture_width != width || *texture_height != height
            })
            .unwrap_or(true);
        if needs_allocation {
            if let Some((textures, _, _)) = texture_sets.remove(&buffer_id) {
                glDeleteTextures(3, textures.as_ptr());
            }
            let mut textures = [0_u32; 3];
            glGenTextures(3, textures.as_mut_ptr());
            for index in 0..3 {
                let plane_width = if index == 0 { width } else { width / 2 };
                let plane_height = if index == 0 { height } else { height / 2 };
                glActiveTexture(GL_TEXTURE0 + index as u32);
                glBindTexture(GL_TEXTURE_2D, textures[index]);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
                glTexImage2D(
                    GL_TEXTURE_2D,
                    0,
                    GL_LUMINANCE as i32,
                    plane_width,
                    plane_height,
                    0,
                    GL_LUMINANCE,
                    GL_UNSIGNED_BYTE,
                    std::ptr::null(),
                );
            }
            texture_sets.insert(buffer_id, (textures, width, height));
        }
        let textures = texture_sets
            .get(&buffer_id)
            .expect("YUV texture set was initialized")
            .0;
        for index in 0..3 {
            let plane_width = if index == 0 { width } else { width / 2 };
            let plane_height = if index == 0 { height } else { height / 2 };
            glActiveTexture(GL_TEXTURE0 + index as u32);
            glBindTexture(GL_TEXTURE_2D, textures[index]);

            glTexSubImage2D(
                GL_TEXTURE_2D,
                0,
                0,
                0,
                plane_width,
                plane_height,
                GL_LUMINANCE,
                GL_UNSIGNED_BYTE,
                yuv.as_ptr().add(offsets[index]).cast(),
            );
        }
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        let fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        if fence.is_null() {
            return Err(format!(
                "YUV glFenceSync failed with GL error 0x{:x}",
                glGetError()
            ));
        }
        glFlush();
        let error = glGetError();
        if error != GL_NO_ERROR {
            glDeleteSync(fence);
            return Err(format!("YUV texture draw failed with GL error 0x{error:x}"));
        }
        Ok(GpuFence {
            sync: fence,
            textures: Vec::new(),
            egl_image: None,
        })
    }
}

unsafe fn compile_shader(kind: u32, source: &str) -> Result<u32, String> {
    let source = CString::new(source).map_err(|error| error.to_string())?;
    let shader = glCreateShader(kind);
    let pointer = source.as_ptr();
    glShaderSource(shader, 1, &pointer, std::ptr::null());
    glCompileShader(shader);
    let mut compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &mut compiled);
    if compiled != 0 {
        return Ok(shader);
    }
    let mut length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &mut length);
    let mut log = vec![0_u8; length.max(1) as usize];
    glGetShaderInfoLog(
        shader,
        length,
        std::ptr::null_mut(),
        log.as_mut_ptr().cast(),
    );
    glDeleteShader(shader);
    Err(format!(
        "external-image shader compilation failed: {}",
        CStr::from_ptr(log.as_ptr().cast()).to_string_lossy()
    ))
}

impl ExternalImageRenderer {
    unsafe fn create() -> Result<Self, String> {
        let vertex = compile_shader(
            GL_VERTEX_SHADER,
            "#ifdef GL_ES\nprecision mediump float;\n#endif\nattribute vec2 position; varying vec2 texture_coordinate; void main() { gl_Position = vec4(position, 0.0, 1.0); texture_coordinate = vec2((position.x + 1.0) * 0.5, 1.0 - (position.y + 1.0) * 0.5); }",
        )?;
        let fragment = compile_shader(
            GL_FRAGMENT_SHADER,
            "#extension GL_OES_EGL_image_external : require\n#ifdef GL_ES\nprecision mediump float;\n#endif\nuniform samplerExternalOES video_texture; uniform vec4 uv_crop; uniform int uv_rotation; varying vec2 texture_coordinate; vec2 source_uv(vec2 p) { vec2 q = mix(uv_crop.xy, uv_crop.zw, p); if (uv_rotation == 1) return vec2(q.y, 1.0-q.x); if (uv_rotation == 2) return vec2(1.0-q.x, 1.0-q.y); if (uv_rotation == 3) return vec2(1.0-q.y, q.x); return q; } void main() { gl_FragColor = texture2D(video_texture, source_uv(texture_coordinate)); }",
        )?;
        let program = glCreateProgram();
        glAttachShader(program, vertex);
        glAttachShader(program, fragment);
        glLinkProgram(program);
        glDeleteShader(vertex);
        glDeleteShader(fragment);
        let mut linked = 0;
        glGetProgramiv(program, GL_LINK_STATUS, &mut linked);
        if linked == 0 {
            let mut length = 0;
            glGetProgramiv(program, GL_INFO_LOG_LENGTH, &mut length);
            let mut log = vec![0_u8; length.max(1) as usize];
            glGetProgramInfoLog(
                program,
                length,
                std::ptr::null_mut(),
                log.as_mut_ptr().cast(),
            );
            glDeleteProgram(program);
            return Err(format!(
                "external-image shader link failed: {}",
                CStr::from_ptr(log.as_ptr().cast()).to_string_lossy()
            ));
        }
        let vertices: [f32; 8] = [-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let mut vertex_buffer = 0;
        glGenBuffers(1, &mut vertex_buffer);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
        glBufferData(
            GL_ARRAY_BUFFER,
            std::mem::size_of_val(&vertices) as isize,
            vertices.as_ptr().cast(),
            GL_STATIC_DRAW,
        );
        glUseProgram(program);
        let sampler = CString::new("video_texture").unwrap();
        glUniform1i(glGetUniformLocation(program, sampler.as_ptr()), 0);
        let position_name = CString::new("position").unwrap();
        let position = glGetAttribLocation(program, position_name.as_ptr());
        if position < 0 {
            glDeleteProgram(program);
            return Err("external-image shader position input is missing".to_string());
        }
        glEnableVertexAttribArray(position as u32);
        glVertexAttribPointer(position as u32, 2, GL_FLOAT, 0, 0, std::ptr::null());
        let image_target: GlEglImageTargetTexture2dOes =
            std::mem::transmute(egl_proc("glEGLImageTargetTexture2DOES")?);
        Ok(Self {
            program,
            vertex_buffer,
            position: position as u32,
            image_target,
        })
    }

    unsafe fn bind_image(&self, image: *mut c_void) -> Result<u32, String> {
        let mut texture = 0;
        glGenTextures(1, &mut texture);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_EXTERNAL_OES, texture);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        (self.image_target)(GL_TEXTURE_EXTERNAL_OES, image);
        let error = glGetError();
        if error != GL_NO_ERROR {
            glDeleteTextures(1, &texture);
            return Err(format!(
                "external NV12 texture binding failed with GL error 0x{error:x}"
            ));
        }
        Ok(texture)
    }

    unsafe fn draw(
        &self,
        image: *mut c_void,
        viewport_x: i32,
        viewport_y: i32,
        width: i32,
        height: i32,
        uv_crop: [f32; 4],
        rotation_quadrants: u8,
    ) -> Result<GpuFence, String> {
        let width = width.max(1);
        let height = height.max(1);
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);
        glViewport(viewport_x, viewport_y, width, height);
        glUseProgram(self.program);
        apply_uv_transform(self.program, uv_crop, rotation_quadrants);
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer);
        glEnableVertexAttribArray(self.position);
        glVertexAttribPointer(self.position, 2, GL_FLOAT, 0, 0, std::ptr::null());
        let mut texture = 0;
        glGenTextures(1, &mut texture);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_EXTERNAL_OES, texture);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        (self.image_target)(GL_TEXTURE_EXTERNAL_OES, image);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        let fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        if fence.is_null() {
            glDeleteTextures(1, &texture);
            return Err(format!(
                "glFenceSync failed with GL error 0x{:x}",
                glGetError()
            ));
        }
        glFlush();
        let error = glGetError();
        if error != GL_NO_ERROR {
            glDeleteSync(fence);
            glDeleteTextures(1, &texture);
            return Err(format!(
                "external NV12 shader draw failed with GL error 0x{error:x}"
            ));
        }
        Ok(GpuFence {
            sync: fence,
            textures: vec![texture],
            egl_image: None,
        })
    }

    unsafe fn draw_cached(
        &self,
        texture: u32,
        viewport_x: i32,
        viewport_y: i32,
        width: i32,
        height: i32,
        uv_crop: [f32; 4],
        rotation_quadrants: u8,
    ) -> Result<GpuFence, String> {
        glClearColor(0.0, 0.0, 0.0, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);
        glViewport(viewport_x, viewport_y, width.max(1), height.max(1));
        glUseProgram(self.program);
        apply_uv_transform(self.program, uv_crop, rotation_quadrants);
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer);
        glEnableVertexAttribArray(self.position);
        glVertexAttribPointer(self.position, 2, GL_FLOAT, 0, 0, std::ptr::null());
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_EXTERNAL_OES, texture);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
        let fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
        if fence.is_null() {
            return Err(format!(
                "glFenceSync failed with GL error 0x{:x}",
                glGetError()
            ));
        }
        glFlush();
        let error = glGetError();
        if error != GL_NO_ERROR {
            glDeleteSync(fence);
            return Err(format!(
                "cached external NV12 shader draw failed with GL error 0x{error:x}"
            ));
        }
        Ok(GpuFence {
            sync: fence,
            textures: Vec::new(),
            egl_image: None,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GtkGlCapabilities {
    pub egl_vendor: String,
    pub egl_version: String,
    pub gl_version: String,
    pub dmabuf_import: bool,
    pub dmabuf_modifiers: bool,
}

unsafe fn query_string(display: *mut c_void, name: i32, label: &str) -> Result<String, String> {
    let value = eglQueryString(display, name);
    if value.is_null() {
        return Err(format!(
            "eglQueryString({label}) failed with EGL error 0x{:x}",
            eglGetError()
        ));
    }
    Ok(CStr::from_ptr(value).to_string_lossy().into_owned())
}

pub fn current_capabilities() -> Result<GtkGlCapabilities, String> {
    unsafe {
        let display = eglGetCurrentDisplay();
        if display.is_null() {
            return Err("GTK GLArea has no current EGL display".to_string());
        }
        let extensions = query_string(display, EGL_EXTENSIONS, "extensions")?;
        let egl_vendor = query_string(display, EGL_VENDOR, "vendor")?;
        let egl_version = query_string(display, EGL_VERSION, "version")?;
        let gl_version_ptr = glGetString(GL_VERSION);
        if gl_version_ptr.is_null() {
            return Err("GTK GLArea has no current OpenGL version".to_string());
        }
        let gl_version = CStr::from_ptr(gl_version_ptr.cast())
            .to_string_lossy()
            .into_owned();
        let dmabuf_import = validate_egl_extensions(&extensions, false).is_ok();
        let dmabuf_modifiers = validate_egl_extensions(&extensions, true).is_ok();
        Ok(GtkGlCapabilities {
            egl_vendor,
            egl_version,
            gl_version,
            dmabuf_import,
            dmabuf_modifiers,
        })
    }
}

pub fn clear_probe() -> Result<(), String> {
    unsafe {
        glClearColor(0.02, 0.08, 0.16, 1.0);
        glClear(GL_COLOR_BUFFER_BIT);
        glFlush();
        let error = glGetError();
        if error != GL_NO_ERROR {
            return Err(format!("GTK GLArea clear failed with GL error 0x{error:x}"));
        }
    }
    Ok(())
}

pub fn clear_external_image_cache() {
    unsafe {
        EXTERNAL_IMAGE_CACHE.with(|cache| {
            for (_, entry) in cache.borrow_mut().drain() {
                destroy_cached_external_image(entry);
            }
        });
        EXTERNAL_RENDERER.with(|renderer| *renderer.borrow_mut() = None);
        EXTERNAL_CONTEXT.with(|context| context.set(0));
    }
}

pub fn upload_yuv420_frame(
    buffer_id: u32,
    yuv: &[u8],
    source_width: i32,
    source_height: i32,
    strides: [i32; 3],
    offsets: [usize; 3],
    viewport_x: i32,
    viewport_y: i32,
    viewport_width: i32,
    viewport_height: i32,
    uv_crop: [f32; 4],
    rotation_quadrants: u8,
    color_space: Option<SurfaceColorSpace>,
    color_range: Option<SurfaceColorRange>,
) -> Result<GpuFence, String> {
    let expected_strides = [source_width, source_width / 2, source_width / 2];
    let end = offsets[2]
        .checked_add((strides[2] as usize).saturating_mul((source_height / 2) as usize))
        .ok_or_else(|| "YUV texture size overflow".to_string())?;
    if source_width <= 0 || source_height <= 0 || strides != expected_strides || yuv.len() < end {
        return Err("YUV420 texture upload requires tightly packed valid planes".to_string());
    }
    unsafe {
        YUV_RENDERER.with(|slot| {
            if slot.borrow().is_none() {
                *slot.borrow_mut() = Some(YuvRenderer::create()?);
            }
            slot.borrow()
                .as_ref()
                .expect("YUV renderer was initialized")
                .draw(
                    buffer_id,
                    yuv,
                    source_width,
                    source_height,
                    offsets,
                    viewport_x,
                    viewport_y,
                    viewport_width,
                    viewport_height,
                    uv_crop,
                    rotation_quadrants,
                    color_space,
                    color_range,
                )
        })
    }
}

pub fn upload_rgba_frame(
    rgba: &[u8],
    source_width: i32,
    source_height: i32,
    stride: i32,
    viewport_x: i32,
    viewport_y: i32,
    viewport_width: i32,
    viewport_height: i32,
    uv_crop: [f32; 4],
    rotation_quadrants: u8,
) -> Result<GpuFence, String> {
    if source_width <= 0
        || source_height <= 0
        || stride != source_width.saturating_mul(4)
        || rgba.len() < (stride as usize).saturating_mul(source_height as usize)
    {
        return Err("RGBA texture upload requires tightly packed valid pixels".to_string());
    }
    unsafe {
        RGBA_RENDERER.with(|slot| {
            if slot.borrow().is_none() {
                *slot.borrow_mut() = Some(RgbaRenderer::create()?);
            }
            slot.borrow()
                .as_ref()
                .expect("RGBA renderer was initialized")
                .draw(
                    rgba,
                    source_width,
                    source_height,
                    viewport_x,
                    viewport_y,
                    viewport_width,
                    viewport_height,
                    uv_crop,
                    rotation_quadrants,
                )
        })
    }
}

pub fn import_nv12_frame(
    descriptor: &SurfaceDescriptor,
    object_fds: &[RawFd],
    viewport_x: i32,
    viewport_y: i32,
    viewport_width: i32,
    viewport_height: i32,
    uv_crop: [f32; 4],
    rotation_quadrants: u8,
) -> Result<GpuFence, String> {
    let import = build_nv12_attributes(descriptor, object_fds)?;
    unsafe {
        let display = eglGetCurrentDisplay();
        if display.is_null() {
            return Err("GTK GLArea has no current EGL display".to_string());
        }
        let context = eglGetCurrentContext();
        if context.is_null() {
            return Err("GTK GLArea has no current EGL context".to_string());
        }
        let create: EglCreateImageKhr = std::mem::transmute(egl_proc("eglCreateImageKHR")?);
        let destroy: EglDestroyImageKhr = std::mem::transmute(egl_proc("eglDestroyImageKHR")?);
        EXTERNAL_CONTEXT.with(|active_context| {
            if active_context.get() != context as usize {
                EXTERNAL_IMAGE_CACHE.with(|cache| {
                    for (_, entry) in cache.borrow_mut().drain() {
                        (entry.destroy)(entry.display, entry.image);
                    }
                });
                EXTERNAL_RENDERER.with(|renderer| {
                    if let Some(renderer) = renderer.borrow_mut().take() {
                        // The previous context owns these GL names and will reclaim them
                        // when it is destroyed; deleting them in the new context is invalid.
                        std::mem::forget(renderer);
                    }
                });
                active_context.set(context as usize);
            }
        });
        let create_image = || {
            let image = create(
                display,
                std::ptr::null_mut(),
                EGL_LINUX_DMA_BUF_EXT,
                std::ptr::null_mut(),
                import.attributes.as_ptr(),
            );
            if image.is_null() {
                return Err(format!(
                    "eglCreateImageKHR(NV12) failed with EGL error 0x{:x}",
                    eglGetError()
                ));
            }
            Ok(image)
        };
        EXTERNAL_RENDERER.with(|slot| {
            if slot.borrow().is_none() {
                *slot.borrow_mut() = Some(ExternalImageRenderer::create()?);
            }
            Ok::<(), String>(())
        })?;
        let object_identity = if descriptor.reusable_dmabuf {
            Some(dmabuf_object_identity(object_fds)?)
        } else {
            None
        };
        let (image, cached_texture) = if descriptor.reusable_dmabuf {
            EXTERNAL_IMAGE_CACHE.with(|cache| -> Result<(*mut c_void, Option<u32>), String> {
                let mut cache = cache.borrow_mut();
                let key = (
                    display as usize,
                    context as usize,
                    descriptor.generation,
                    descriptor.buffer_id,
                    object_identity
                        .clone()
                        .expect("reusable DMA-BUF identity was initialized"),
                );
                if let Some(entry) = cache.get(&key) {
                    return Ok((entry.image, Some(entry.texture)));
                }
                if cache.len() >= 6 {
                    for (_, entry) in cache.drain() {
                        destroy_cached_external_image(entry);
                    }
                }
                let image = create_image()?;
                let texture = EXTERNAL_RENDERER.with(|slot| {
                    slot.borrow()
                        .as_ref()
                        .expect("external renderer was initialized")
                        .bind_image(image)
                })?;
                cache.insert(
                    key,
                    CachedExternalImage {
                        display,
                        image,
                        texture,
                        destroy,
                    },
                );
                Ok((image, Some(texture)))
            })?
        } else {
            (create_image()?, None)
        };
        let draw_result = EXTERNAL_RENDERER.with(|slot| {
            let renderer_ref = slot.borrow();
            let renderer = renderer_ref
                .as_ref()
                .expect("external renderer was initialized");
            if let Some(texture) = cached_texture {
                renderer.draw_cached(
                    texture,
                    viewport_x,
                    viewport_y,
                    viewport_width,
                    viewport_height,
                    uv_crop,
                    rotation_quadrants,
                )
            } else {
                renderer.draw(
                    image,
                    viewport_x,
                    viewport_y,
                    viewport_width,
                    viewport_height,
                    uv_crop,
                    rotation_quadrants,
                )
            }
        });
        match draw_result {
            Ok(fence) if descriptor.reusable_dmabuf => Ok(fence),
            Ok(fence) => Ok(fence.retain_egl_image(display, image, destroy)),
            Err(error) => {
                if descriptor.reusable_dmabuf || destroy(display, image) != 0 {
                    return Err(error);
                } else {
                    return Err(format!(
                        "{error}; eglDestroyImageKHR also failed with EGL error 0x{:x}",
                        eglGetError()
                    ));
                }
            }
        }
    }
}
