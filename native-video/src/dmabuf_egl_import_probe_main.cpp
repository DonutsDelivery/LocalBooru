// SPDX-License-Identifier: MIT
#include "dmabuf_frame_decoder.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES2/gl2.h>
#include <GLES2/gl2ext.h>
#include <drm/drm_fourcc.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace {

class EglSession {
 public:
  EglSession() {
    display_ = eglGetPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA,
                                     EGL_DEFAULT_DISPLAY, nullptr);
    if (display_ == EGL_NO_DISPLAY || !eglInitialize(display_, nullptr, nullptr)) {
      throw std::runtime_error("failed to initialize a surfaceless EGL display");
    }
    if (!eglBindAPI(EGL_OPENGL_ES_API)) {
      throw std::runtime_error("failed to bind the OpenGL ES API");
    }
    const EGLint config_attributes[] = {
        EGL_SURFACE_TYPE, EGL_PBUFFER_BIT, EGL_RENDERABLE_TYPE,
        EGL_OPENGL_ES2_BIT, EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8,
        EGL_BLUE_SIZE, 8, EGL_NONE};
    EGLConfig config = nullptr;
    EGLint count = 0;
    if (!eglChooseConfig(display_, config_attributes, &config, 1, &count) ||
        count != 1) {
      throw std::runtime_error("failed to choose an EGL configuration");
    }
    const EGLint surface_attributes[] = {EGL_WIDTH, 1, EGL_HEIGHT, 1, EGL_NONE};
    surface_ = eglCreatePbufferSurface(display_, config, surface_attributes);
    if (surface_ == EGL_NO_SURFACE) {
      throw std::runtime_error("failed to create an EGL pbuffer");
    }
    const EGLint context_attributes[] = {EGL_CONTEXT_CLIENT_VERSION, 2, EGL_NONE};
    context_ = eglCreateContext(display_, config, EGL_NO_CONTEXT,
                                context_attributes);
    if (context_ == EGL_NO_CONTEXT ||
        !eglMakeCurrent(display_, surface_, surface_, context_)) {
      throw std::runtime_error("failed to create a current EGL context");
    }
  }

  ~EglSession() {
    if (display_ == EGL_NO_DISPLAY) return;
    eglMakeCurrent(display_, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (context_ != EGL_NO_CONTEXT) eglDestroyContext(display_, context_);
    if (surface_ != EGL_NO_SURFACE) eglDestroySurface(display_, surface_);
    eglTerminate(display_);
  }

  EGLDisplay display() const { return display_; }

 private:
  EGLDisplay display_ = EGL_NO_DISPLAY;
  EGLSurface surface_ = EGL_NO_SURFACE;
  EGLContext context_ = EGL_NO_CONTEXT;
};

bool has_extension(const char* extensions, const std::string& requested) {
  if (!extensions) return false;
  const std::string list(extensions);
  std::size_t start = 0;
  while (start < list.size()) {
    const std::size_t end = list.find(' ', start);
    const std::string item = list.substr(start, end - start);
    if (item == requested) return true;
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return false;
}

std::vector<EGLAttrib> nv12_attributes(
    const localbooru::native_video::DmabufVideoFrame& frame) {
  if (frame.layers().size() != 2 || frame.layers()[0].planes.size() != 1 ||
      frame.layers()[1].planes.size() != 1) {
    throw std::runtime_error("DRM PRIME frame is not a two-plane NV12 layout");
  }
  const auto& luma = frame.layers()[0].planes[0];
  const auto& chroma = frame.layers()[1].planes[0];
  const auto& luma_object = frame.objects().at(luma.object_index);
  const auto& chroma_object = frame.objects().at(chroma.object_index);
  return {
      EGL_WIDTH,
      frame.width(),
      EGL_HEIGHT,
      frame.height(),
      EGL_LINUX_DRM_FOURCC_EXT,
      DRM_FORMAT_NV12,
      EGL_DMA_BUF_PLANE0_FD_EXT,
      luma_object.fd,
      EGL_DMA_BUF_PLANE0_OFFSET_EXT,
      luma.offset,
      EGL_DMA_BUF_PLANE0_PITCH_EXT,
      luma.pitch,
      EGL_DMA_BUF_PLANE0_MODIFIER_LO_EXT,
      static_cast<EGLAttrib>(luma_object.modifier & 0xffffffffU),
      EGL_DMA_BUF_PLANE0_MODIFIER_HI_EXT,
      static_cast<EGLAttrib>(luma_object.modifier >> 32U),
      EGL_DMA_BUF_PLANE1_FD_EXT,
      chroma_object.fd,
      EGL_DMA_BUF_PLANE1_OFFSET_EXT,
      chroma.offset,
      EGL_DMA_BUF_PLANE1_PITCH_EXT,
      chroma.pitch,
      EGL_DMA_BUF_PLANE1_MODIFIER_LO_EXT,
      static_cast<EGLAttrib>(chroma_object.modifier & 0xffffffffU),
      EGL_DMA_BUF_PLANE1_MODIFIER_HI_EXT,
      static_cast<EGLAttrib>(chroma_object.modifier >> 32U),
      EGL_YUV_COLOR_SPACE_HINT_EXT,
      EGL_ITU_REC709_EXT,
      EGL_SAMPLE_RANGE_HINT_EXT,
      EGL_YUV_FULL_RANGE_EXT,
      EGL_YUV_CHROMA_HORIZONTAL_SITING_HINT_EXT,
      EGL_YUV_CHROMA_SITING_0_5_EXT,
      EGL_YUV_CHROMA_VERTICAL_SITING_HINT_EXT,
      EGL_YUV_CHROMA_SITING_0_5_EXT,
      EGL_NONE};
}

GLuint compile_shader(GLenum type, const char* source) {
  const GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &source, nullptr);
  glCompileShader(shader);
  GLint compiled = GL_FALSE;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
  if (compiled == GL_TRUE) return shader;
  GLint length = 0;
  glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &length);
  std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
  glGetShaderInfoLog(shader, length, nullptr, log.data());
  glDeleteShader(shader);
  throw std::runtime_error("failed to compile DMA-BUF probe shader: " + log);
}

GLuint create_nv12_program() {
  static constexpr char kVertex[] = R"(
#ifdef GL_ES
precision mediump float;
#endif
attribute vec2 position;
varying vec2 texture_coordinate;
void main() {
  gl_Position = vec4(position, 0.0, 1.0);
  texture_coordinate = vec2((position.x + 1.0) * 0.5,
                            1.0 - (position.y + 1.0) * 0.5);
}
)";
  static constexpr char kFragment[] = R"(
#extension GL_OES_EGL_image_external : require
#ifdef GL_ES
precision mediump float;
#endif
uniform samplerExternalOES video_texture;
varying vec2 texture_coordinate;
void main() {
  gl_FragColor = texture2D(video_texture, vec2(0.251, 0.251));
}
)";
  const GLuint vertex = compile_shader(GL_VERTEX_SHADER, kVertex);
  const GLuint fragment = compile_shader(GL_FRAGMENT_SHADER, kFragment);
  const GLuint program = glCreateProgram();
  glAttachShader(program, vertex);
  glAttachShader(program, fragment);
  glLinkProgram(program);
  glDeleteShader(vertex);
  glDeleteShader(fragment);
  GLint linked = GL_FALSE;
  glGetProgramiv(program, GL_LINK_STATUS, &linked);
  if (linked == GL_TRUE) return program;
  GLint length = 0;
  glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);
  std::string log(static_cast<std::size_t>(std::max(length, 1)), '\0');
  glGetProgramInfoLog(program, length, nullptr, log.data());
  glDeleteProgram(program);
  throw std::runtime_error("failed to link DMA-BUF probe shader: " + log);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 4) {
    std::cerr << "usage: native-video-dmabuf-egl-import-probe MEDIA [DRM_NODE] "
                 "[FRAME_COUNT]\n";
    return 2;
  }
  try {
    const std::string drm_node = argc >= 3 ? argv[2] : "/dev/dri/renderD128";
    const std::uint64_t requested_frames =
        argc == 4 ? std::stoull(argv[3]) : 1;
    if (requested_frames == 0 || requested_frames > 100000) {
      throw std::runtime_error("frame count must be between 1 and 100000");
    }
    localbooru::native_video::DmabufFrameDecoder decoder(argv[1], drm_node);
    EglSession egl;
    const char* extensions = eglQueryString(egl.display(), EGL_EXTENSIONS);
    if (!has_extension(extensions, "EGL_EXT_image_dma_buf_import") ||
        !has_extension(extensions,
                       "EGL_EXT_image_dma_buf_import_modifiers")) {
      throw std::runtime_error("EGL display lacks DMA-BUF modifier import");
    }
    const auto image_target = reinterpret_cast<PFNGLEGLIMAGETARGETTEXTURE2DOESPROC>(
        eglGetProcAddress("glEGLImageTargetTexture2DOES"));
    if (!image_target) {
      throw std::runtime_error("glEGLImageTargetTexture2DOES is unavailable");
    }

    const GLuint program = create_nv12_program();
    static constexpr GLfloat vertices[] = {
        -1.0F, -1.0F, 1.0F, -1.0F, -1.0F, 1.0F, 1.0F, 1.0F};
    GLuint vertex_buffer = 0;
    glGenBuffers(1, &vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
    glViewport(0, 0, 1, 1);
    glUseProgram(program);
    glUniform1i(glGetUniformLocation(program, "video_texture"), 0);
    const GLint position = glGetAttribLocation(program, "position");
    if (position < 0) throw std::runtime_error("NV12 shader position input is missing");
    glEnableVertexAttribArray(static_cast<GLuint>(position));
    glVertexAttribPointer(static_cast<GLuint>(position), 2, GL_FLOAT, GL_FALSE,
                          0, nullptr);
    std::array<std::uint8_t, 4> sample{};
    std::uint64_t rendered_frames = 0;
    int width = 0;
    int height = 0;
    const auto started = std::chrono::steady_clock::now();
    while (rendered_frames < requested_frames) {
      auto frame = decoder.next_frame();
      if (!frame) break;
      if (rendered_frames == 0) {
        width = frame->width();
        height = frame->height();
      }
      auto attributes = nv12_attributes(*frame);
      EGLImageKHR image = eglCreateImage(
          egl.display(), EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, nullptr,
          attributes.data());
      if (image == EGL_NO_IMAGE_KHR) {
        throw std::runtime_error(
            "eglCreateImage rejected composite NV12 DMA-BUF with EGL error " +
            std::to_string(eglGetError()));
      }
      GLuint texture = 0;
      glGenTextures(1, &texture);
      glActiveTexture(GL_TEXTURE0);
      glBindTexture(GL_TEXTURE_EXTERNAL_OES, texture);
      glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
      glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_S,
                      GL_CLAMP_TO_EDGE);
      glTexParameteri(GL_TEXTURE_EXTERNAL_OES, GL_TEXTURE_WRAP_T,
                      GL_CLAMP_TO_EDGE);
      image_target(GL_TEXTURE_EXTERNAL_OES, image);
      glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
      glFinish();
      if (glGetError() != GL_NO_ERROR) {
        throw std::runtime_error("NV12 DMA-BUF import or shader draw failed");
      }
      if (rendered_frames == 0 || rendered_frames + 1 == requested_frames) {
        glReadPixels(0, 0, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, sample.data());
        if (glGetError() != GL_NO_ERROR || sample[3] != 255) {
          throw std::runtime_error("NV12 shader output readback failed");
        }
      }
      glDeleteTextures(1, &texture);
      eglDestroyImage(egl.display(), image);
      ++rendered_frames;
    }
    const double elapsed_seconds = std::chrono::duration<double>(
                                       std::chrono::steady_clock::now() - started)
                                       .count();
    if (rendered_frames == 0) {
      throw std::runtime_error("media produced no hardware frame");
    }

    nlohmann::json output{{"imported", true},
                          {"drm_node", drm_node},
                          {"width", width},
                          {"height", height},
                          {"requested_frames", requested_frames},
                          {"rendered_frames", rendered_frames},
                          {"elapsed_seconds", elapsed_seconds},
                          {"rendered_fps", rendered_frames / elapsed_seconds},
                          {"egl_images_per_frame", 1},
                          {"shader", "nv12_external_oes"},
                          {"sample_rgba", sample},
                          {"gpu_synchronized", true}};
    std::cout << output.dump() << '\n';
    glDisableVertexAttribArray(static_cast<GLuint>(position));
    glDeleteBuffers(1, &vertex_buffer);
    glDeleteProgram(program);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
