// SPDX-License-Identifier: MIT
#include "dmabuf_frame_uploader.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>

extern "C" {
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/hwcontext.h>
#include <libavutil/pixfmt.h>
}

namespace localbooru::native_video {
namespace {

std::string av_error(int code) {
  char buffer[AV_ERROR_MAX_STRING_SIZE]{};
  av_strerror(code, buffer, sizeof(buffer));
  return buffer;
}

AVColorSpace color_space_from_name(const std::string &name) {
  if (name == "bt601")
    return AVCOL_SPC_SMPTE170M;
  if (name == "bt2020")
    return AVCOL_SPC_BT2020_NCL;
  return AVCOL_SPC_BT709;
}

AVChromaLocation chroma_location_from_name(const std::string &name) {
  if (name == "center")
    return AVCHROMA_LOC_CENTER;
  if (name == "top_left")
    return AVCHROMA_LOC_TOPLEFT;
  return AVCHROMA_LOC_LEFT;
}

} // namespace

struct DmabufFrameUploader::Impl {
  AVBufferRef *hardware_device = nullptr;
  AVBufferRef *hardware_frames = nullptr;
  AVFrame *software_frame = nullptr;
  std::vector<AVFrame *> hardware_frames_by_buffer;
  int width = 0;
  int height = 0;
  double sample_aspect_ratio = 1.0;
  int rotation_degrees = 0;
  std::string color_space;
  std::string color_range;
  std::string chroma_location;

  ~Impl() {
    for (auto *&frame : hardware_frames_by_buffer)
      av_frame_free(&frame);
    av_frame_free(&software_frame);
    av_buffer_unref(&hardware_frames);
    av_buffer_unref(&hardware_device);
  }
};

DmabufFrameUploader::DmabufFrameUploader(
    const std::string &render_node, int width, int height,
    double sample_aspect_ratio, int rotation_degrees, std::string color_space,
    std::string color_range, std::string chroma_location)
    : impl_(std::make_unique<Impl>()) {
  if (width <= 0 || height <= 0 || width % 2 != 0 || height % 2 != 0)
    throw std::runtime_error("DMA-BUF upload requires even positive dimensions");
  impl_->width = width;
  impl_->height = height;
  impl_->sample_aspect_ratio = sample_aspect_ratio;
  impl_->rotation_degrees = rotation_degrees;
  impl_->color_space = std::move(color_space);
  impl_->color_range = std::move(color_range);
  impl_->chroma_location = std::move(chroma_location);

  int status = av_hwdevice_ctx_create(&impl_->hardware_device,
                                      AV_HWDEVICE_TYPE_VAAPI,
                                      render_node.c_str(), nullptr, 0);
  if (status < 0)
    throw std::runtime_error("VA-API upload device creation failed: " +
                             av_error(status));
  impl_->hardware_frames = av_hwframe_ctx_alloc(impl_->hardware_device);
  if (!impl_->hardware_frames)
    throw std::runtime_error("VA-API upload frame context allocation failed");
  auto *frames = reinterpret_cast<AVHWFramesContext *>(
      impl_->hardware_frames->data);
  frames->format = AV_PIX_FMT_VAAPI;
  frames->sw_format = AV_PIX_FMT_NV12;
  frames->width = width;
  frames->height = height;
  frames->initial_pool_size = 3;
  status = av_hwframe_ctx_init(impl_->hardware_frames);
  if (status < 0)
    throw std::runtime_error("VA-API upload frame context failed: " +
                             av_error(status));

  impl_->software_frame = av_frame_alloc();
  if (!impl_->software_frame)
    throw std::runtime_error("VA-API upload frame allocation failed");
  impl_->software_frame->format = AV_PIX_FMT_NV12;
  impl_->software_frame->width = width;
  impl_->software_frame->height = height;
  status = av_frame_get_buffer(impl_->software_frame, 32);
  if (status < 0)
    throw std::runtime_error("NV12 staging frame allocation failed: " +
                             av_error(status));
  for (std::size_t index = 0; index < 3; ++index) {
    auto *hardware_frame = av_frame_alloc();
    if (!hardware_frame)
      throw std::runtime_error("VA-API upload surface allocation failed");
    status = av_hwframe_get_buffer(impl_->hardware_frames, hardware_frame, 0);
    if (status < 0) {
      av_frame_free(&hardware_frame);
      throw std::runtime_error("VA-API upload surface allocation failed: " +
                               av_error(status));
    }
    impl_->hardware_frames_by_buffer.push_back(hardware_frame);
  }
}

DmabufFrameUploader::~DmabufFrameUploader() = default;

DmabufVideoFrame
DmabufFrameUploader::upload(const DecodedVideoFrame &frame,
                            std::size_t buffer_id) {
  if (buffer_id >= impl_->hardware_frames_by_buffer.size())
    throw std::runtime_error("DMA-BUF upload buffer ID is out of range");
  if (!frame.yuv420p || frame.width != impl_->width ||
      frame.height != impl_->height) {
    throw std::runtime_error("DMA-BUF upload received an incompatible frame");
  }
  const std::size_t y_size =
      static_cast<std::size_t>(impl_->width) * impl_->height;
  const std::size_t chroma_width = static_cast<std::size_t>(impl_->width / 2);
  const std::size_t chroma_height = static_cast<std::size_t>(impl_->height / 2);
  const std::size_t chroma_size = chroma_width * chroma_height;
  if (frame.rgba.size() < y_size + 2 * chroma_size)
    throw std::runtime_error("DMA-BUF upload received a truncated YUV frame");
  int status = av_frame_make_writable(impl_->software_frame);
  if (status < 0)
    throw std::runtime_error("NV12 staging frame is not writable: " +
                             av_error(status));

  for (int row = 0; row < impl_->height; ++row) {
    std::memcpy(impl_->software_frame->data[0] +
                    row * impl_->software_frame->linesize[0],
                frame.rgba.data() + static_cast<std::size_t>(row) * impl_->width,
                static_cast<std::size_t>(impl_->width));
  }
  const auto *u = frame.rgba.data() + y_size;
  const auto *v = u + chroma_size;
  for (std::size_t row = 0; row < chroma_height; ++row) {
    auto *uv = impl_->software_frame->data[1] +
               row * impl_->software_frame->linesize[1];
    for (std::size_t column = 0; column < chroma_width; ++column) {
      const std::size_t source = row * chroma_width + column;
      uv[column * 2] = u[source];
      uv[column * 2 + 1] = v[source];
    }
  }

  auto *hardware_frame = impl_->hardware_frames_by_buffer[buffer_id];
  status = av_hwframe_transfer_data(hardware_frame, impl_->software_frame, 0);
  if (status < 0) {
    throw std::runtime_error("VA-API frame upload failed: " + av_error(status));
  }
  hardware_frame->colorspace = color_space_from_name(impl_->color_space);
  hardware_frame->color_range =
      impl_->color_range == "full" ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;
  hardware_frame->chroma_location =
      chroma_location_from_name(impl_->chroma_location);
  return DmabufFrameDecoder::export_frame(
      hardware_frame, impl_->hardware_device, frame.pts_seconds,
      impl_->sample_aspect_ratio, impl_->rotation_degrees, impl_->color_space,
      impl_->color_range, impl_->chroma_location);
}

} // namespace localbooru::native_video
