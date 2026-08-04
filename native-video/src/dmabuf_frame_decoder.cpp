// SPDX-License-Identifier: MIT
#include "dmabuf_frame_decoder.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/display.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_drm.h>
#include <libavutil/hwcontext_vaapi.h>
#include <va/va.h>
}

namespace localbooru::native_video {
namespace {

std::string av_error(int code) {
  char buffer[AV_ERROR_MAX_STRING_SIZE]{};
  av_strerror(code, buffer, sizeof(buffer));
  return buffer;
}

AVPixelFormat select_vaapi_format(AVCodecContext *,
                                  const AVPixelFormat *formats) {
  for (const AVPixelFormat *format = formats; *format != AV_PIX_FMT_NONE;
       ++format) {
    if (*format == AV_PIX_FMT_VAAPI)
      return *format;
  }
  return AV_PIX_FMT_NONE;
}

bool decoder_supports_vaapi(const AVCodec *decoder) {
  for (int index = 0;; ++index) {
    const AVCodecHWConfig *config = avcodec_get_hw_config(decoder, index);
    if (!config)
      return false;
    if (config->device_type == AV_HWDEVICE_TYPE_VAAPI &&
        (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) != 0) {
      return true;
    }
  }
}

std::string frame_color_space(const AVFrame &frame) {
  switch (frame.colorspace) {
  case AVCOL_SPC_BT709:
    return "bt709";
  case AVCOL_SPC_BT2020_NCL:
  case AVCOL_SPC_BT2020_CL:
    return "bt2020";
  case AVCOL_SPC_BT470BG:
  case AVCOL_SPC_SMPTE170M:
    return "bt601";
  default:
    return frame.width >= 1280 || frame.height > 576 ? "bt709" : "bt601";
  }
}

std::string frame_chroma_location(const AVFrame &frame) {
  switch (frame.chroma_location) {
  case AVCHROMA_LOC_LEFT:
    return "left";
  case AVCHROMA_LOC_TOPLEFT:
    return "top_left";
  case AVCHROMA_LOC_TOP:
    return "top";
  case AVCHROMA_LOC_CENTER:
  default:
    return "center";
  }
}

int rotation_for_stream(const AVStream &stream) {
  const auto *parameters = stream.codecpar;
  const auto *side_data = av_packet_side_data_get(
      parameters->coded_side_data, parameters->nb_coded_side_data,
      AV_PKT_DATA_DISPLAYMATRIX);
  if (!side_data || side_data->size < 9 * sizeof(std::int32_t))
    return 0;
  const double clockwise_rotation = -av_display_rotation_get(
      reinterpret_cast<const std::int32_t *>(side_data->data));
  const int rotation =
      static_cast<int>(std::llround(clockwise_rotation / 90.0)) * 90;
  return (rotation % 360 + 360) % 360;
}

} // namespace

struct DmabufVideoFrame::Impl {
  AVFrame *frame = nullptr;
  int width = 0;
  int height = 0;
  double sample_aspect_ratio = 1.0;
  int rotation_degrees = 0;
  double pts_seconds = 0.0;
  std::string color_space;
  std::string color_range;
  std::string chroma_location;
  std::vector<DmabufFrameObject> objects;
  std::vector<DmabufFrameLayer> layers;

  ~Impl() {
    if (frame)
      av_frame_free(&frame);
  }
};

DmabufVideoFrame::DmabufVideoFrame(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
DmabufVideoFrame::~DmabufVideoFrame() = default;
DmabufVideoFrame::DmabufVideoFrame(DmabufVideoFrame &&) noexcept = default;
DmabufVideoFrame &
DmabufVideoFrame::operator=(DmabufVideoFrame &&) noexcept = default;
int DmabufVideoFrame::width() const { return impl_->width; }
int DmabufVideoFrame::height() const { return impl_->height; }
double DmabufVideoFrame::sample_aspect_ratio() const {
  return impl_->sample_aspect_ratio;
}
int DmabufVideoFrame::rotation_degrees() const {
  return impl_->rotation_degrees;
}
double DmabufVideoFrame::pts_seconds() const { return impl_->pts_seconds; }
const std::string &DmabufVideoFrame::color_space() const {
  return impl_->color_space;
}
const std::string &DmabufVideoFrame::color_range() const {
  return impl_->color_range;
}
const std::string &DmabufVideoFrame::chroma_location() const {
  return impl_->chroma_location;
}
const std::vector<DmabufFrameObject> &DmabufVideoFrame::objects() const {
  return impl_->objects;
}
const std::vector<DmabufFrameLayer> &DmabufVideoFrame::layers() const {
  return impl_->layers;
}

struct DmabufFrameDecoder::Impl {
  AVFormatContext *format = nullptr;
  AVCodecContext *codec = nullptr;
  AVBufferRef *hardware_device = nullptr;
  AVPacket *packet = nullptr;
  AVFrame *hardware_frame = nullptr;
  int stream_index = -1;
  double sample_aspect_ratio = 1.0;
  int rotation_degrees = 0;
  bool input_eof = false;
  bool flush_sent = false;
  double seek_target = 0.0;

  ~Impl() {
    if (hardware_frame)
      av_frame_free(&hardware_frame);
    if (packet)
      av_packet_free(&packet);
    if (codec)
      avcodec_free_context(&codec);
    if (hardware_device)
      av_buffer_unref(&hardware_device);
    if (format)
      avformat_close_input(&format);
  }

  DmabufVideoFrame export_frame() {
    const auto *frames_context = reinterpret_cast<const AVHWFramesContext *>(
        hardware_frame->hw_frames_ctx ? hardware_frame->hw_frames_ctx->data
                                      : nullptr);
    const auto *device_context =
        frames_context ? reinterpret_cast<const AVHWDeviceContext *>(
                             frames_context->device_ref->data)
                       : nullptr;
    const auto *vaapi_context =
        device_context ? reinterpret_cast<const AVVAAPIDeviceContext *>(
                             device_context->hwctx)
                       : nullptr;
    const VASurfaceID surface = static_cast<VASurfaceID>(
        reinterpret_cast<std::uintptr_t>(hardware_frame->data[3]));
    if (!vaapi_context || surface == VA_INVALID_SURFACE ||
        vaSyncSurface(vaapi_context->display, surface) != VA_STATUS_SUCCESS) {
      throw std::runtime_error("failed to synchronize VA-API decode surface");
    }
    auto exported = std::make_unique<DmabufVideoFrame::Impl>();
    exported->frame = av_frame_alloc();
    if (!exported->frame)
      throw std::runtime_error("DMA-BUF frame allocation failed");
    exported->frame->format = AV_PIX_FMT_DRM_PRIME;
    const int status =
        av_hwframe_map(exported->frame, hardware_frame, AV_HWFRAME_MAP_READ);
    if (status < 0) {
      throw std::runtime_error("DRM PRIME map failed: " + av_error(status));
    }
    if (exported->frame->format != AV_PIX_FMT_DRM_PRIME ||
        !exported->frame->data[0]) {
      throw std::runtime_error("DRM PRIME map returned no descriptor");
    }

    const auto *descriptor = reinterpret_cast<const AVDRMFrameDescriptor *>(
        exported->frame->data[0]);
    if (descriptor->nb_objects <= 0 || descriptor->nb_layers <= 0) {
      throw std::runtime_error("DRM PRIME descriptor is empty");
    }
    exported->width = hardware_frame->width;
    exported->height = hardware_frame->height;
    exported->sample_aspect_ratio = sample_aspect_ratio;
    exported->rotation_degrees = rotation_degrees;
    exported->color_space = frame_color_space(*hardware_frame);
    exported->color_range =
        hardware_frame->color_range == AVCOL_RANGE_JPEG ? "full" : "narrow";
    exported->chroma_location = frame_chroma_location(*hardware_frame);
    if (hardware_frame->best_effort_timestamp != AV_NOPTS_VALUE) {
      exported->pts_seconds =
          static_cast<double>(hardware_frame->best_effort_timestamp) *
          av_q2d(format->streams[stream_index]->time_base);
    }
    for (int index = 0; index < descriptor->nb_objects; ++index) {
      const auto &object = descriptor->objects[index];
      if (object.fd < 0) {
        throw std::runtime_error("DRM PRIME object has an invalid fd");
      }
      exported->objects.push_back(DmabufFrameObject{
          .fd = object.fd,
          .size = static_cast<std::uint64_t>(object.size),
          .modifier = object.format_modifier,
      });
    }
    for (int layer_index = 0; layer_index < descriptor->nb_layers;
         ++layer_index) {
      const auto &source = descriptor->layers[layer_index];
      DmabufFrameLayer layer;
      layer.format = source.format;
      layer.width =
          layer_index == 0 ? exported->width : (exported->width + 1) / 2;
      layer.height =
          layer_index == 0 ? exported->height : (exported->height + 1) / 2;
      for (int plane_index = 0; plane_index < source.nb_planes; ++plane_index) {
        const auto &plane = source.planes[plane_index];
        if (plane.object_index < 0 ||
            plane.object_index >= descriptor->nb_objects) {
          throw std::runtime_error(
              "DRM PRIME plane references an invalid object");
        }
        layer.planes.push_back(DmabufFramePlane{
            .object_index = plane.object_index,
            .offset = static_cast<std::uint32_t>(plane.offset),
            .pitch = static_cast<std::uint32_t>(plane.pitch),
        });
      }
      exported->layers.push_back(std::move(layer));
    }
    return DmabufVideoFrame(std::move(exported));
  }
};

DmabufFrameDecoder::DmabufFrameDecoder(const std::string &media_path,
                                       const std::string &render_node)
    : impl_(std::make_unique<Impl>()) {
  int status =
      avformat_open_input(&impl_->format, media_path.c_str(), nullptr, nullptr);
  if (status < 0) {
    throw std::runtime_error("failed to open media: " + av_error(status));
  }
  status = avformat_find_stream_info(impl_->format, nullptr);
  if (status < 0) {
    throw std::runtime_error("failed to read media streams: " +
                             av_error(status));
  }
  const AVCodec *decoder = nullptr;
  impl_->stream_index = av_find_best_stream(impl_->format, AVMEDIA_TYPE_VIDEO,
                                            -1, -1, &decoder, 0);
  if (impl_->stream_index < 0 || !decoder) {
    throw std::runtime_error("media has no decodable video stream");
  }
  auto *video_stream = impl_->format->streams[impl_->stream_index];
  const AVRational sample_aspect_ratio =
      av_guess_sample_aspect_ratio(impl_->format, video_stream, nullptr);
  if (sample_aspect_ratio.num > 0 && sample_aspect_ratio.den > 0) {
    impl_->sample_aspect_ratio = av_q2d(sample_aspect_ratio);
  }
  impl_->rotation_degrees = rotation_for_stream(*video_stream);
  if (!decoder_supports_vaapi(decoder)) {
    throw std::runtime_error(
        "decoder does not expose a VA-API hardware configuration");
  }
  status =
      av_hwdevice_ctx_create(&impl_->hardware_device, AV_HWDEVICE_TYPE_VAAPI,
                             render_node.c_str(), nullptr, 0);
  if (status < 0) {
    throw std::runtime_error("VA-API device creation failed: " +
                             av_error(status));
  }
  impl_->codec = avcodec_alloc_context3(decoder);
  if (!impl_->codec)
    throw std::runtime_error("failed to allocate video decoder");
  status = avcodec_parameters_to_context(
      impl_->codec, impl_->format->streams[impl_->stream_index]->codecpar);
  if (status < 0) {
    throw std::runtime_error("failed to configure video decoder: " +
                             av_error(status));
  }
  impl_->codec->get_format = select_vaapi_format;
  impl_->codec->hw_device_ctx = av_buffer_ref(impl_->hardware_device);
  if (!impl_->codec->hw_device_ctx) {
    throw std::runtime_error("failed to retain VA-API device");
  }
  status = avcodec_open2(impl_->codec, decoder, nullptr);
  if (status < 0) {
    throw std::runtime_error("failed to open hardware video decoder: " +
                             av_error(status));
  }
  impl_->packet = av_packet_alloc();
  impl_->hardware_frame = av_frame_alloc();
  if (!impl_->packet || !impl_->hardware_frame) {
    throw std::runtime_error("failed to allocate decoder frames");
  }
}

DmabufFrameDecoder::~DmabufFrameDecoder() = default;

std::optional<DmabufVideoFrame> DmabufFrameDecoder::next_frame() {
  for (;;) {
    const int receive =
        avcodec_receive_frame(impl_->codec, impl_->hardware_frame);
    if (receive == 0) {
      auto frame = impl_->export_frame();
      av_frame_unref(impl_->hardware_frame);
      if (frame.pts_seconds() + 1e-6 < impl_->seek_target)
        continue;
      impl_->seek_target = 0.0;
      return frame;
    }
    if (receive == AVERROR_EOF)
      return std::nullopt;
    if (receive != AVERROR(EAGAIN)) {
      throw std::runtime_error("failed while decoding hardware frame: " +
                               av_error(receive));
    }
    if (impl_->input_eof) {
      if (impl_->flush_sent)
        return std::nullopt;
      const int send = avcodec_send_packet(impl_->codec, nullptr);
      if (send < 0 && send != AVERROR_EOF) {
        throw std::runtime_error("failed to flush hardware decoder: " +
                                 av_error(send));
      }
      impl_->flush_sent = true;
      continue;
    }
    int read = 0;
    do {
      read = av_read_frame(impl_->format, impl_->packet);
      if (read < 0) {
        impl_->input_eof = true;
        break;
      }
      if (impl_->packet->stream_index != impl_->stream_index) {
        av_packet_unref(impl_->packet);
        continue;
      }
      const int send = avcodec_send_packet(impl_->codec, impl_->packet);
      av_packet_unref(impl_->packet);
      if (send < 0 && send != AVERROR(EAGAIN)) {
        throw std::runtime_error("failed to submit video packet: " +
                                 av_error(send));
      }
      break;
    } while (read >= 0);
  }
}

DmabufVideoFrame DmabufFrameDecoder::export_frame(
    AVFrame *hardware_frame, AVBufferRef *hardware_device, double pts_seconds,
    double sample_aspect_ratio, int rotation_degrees,
    const std::string &color_space, const std::string &color_range,
    const std::string &chroma_location) {
  if (!hardware_frame || hardware_frame->format != AV_PIX_FMT_VAAPI)
    throw std::runtime_error("DMA-BUF export requires a VA-API frame");
  const auto *frames_context = reinterpret_cast<const AVHWFramesContext *>(
      hardware_frame->hw_frames_ctx ? hardware_frame->hw_frames_ctx->data
                                    : nullptr);
  const auto *device_context =
      frames_context ? reinterpret_cast<const AVHWDeviceContext *>(
                           frames_context->device_ctx)
                     : (hardware_device
                            ? reinterpret_cast<const AVHWDeviceContext *>(
                                  hardware_device->data)
                            : nullptr);
  const auto *vaapi_context =
      device_context ? reinterpret_cast<const AVVAAPIDeviceContext *>(
                           device_context->hwctx)
                     : nullptr;
  const VASurfaceID surface = static_cast<VASurfaceID>(
      reinterpret_cast<std::uintptr_t>(hardware_frame->data[3]));
  if (!vaapi_context || surface == VA_INVALID_SURFACE ||
      vaSyncSurface(vaapi_context->display, surface) != VA_STATUS_SUCCESS)
    throw std::runtime_error("failed to synchronize VA-API upload surface");

  auto exported = std::make_unique<DmabufVideoFrame::Impl>();
  exported->frame = av_frame_alloc();
  if (!exported->frame)
    throw std::runtime_error("DMA-BUF upload frame allocation failed");
  exported->frame->format = AV_PIX_FMT_DRM_PRIME;
  const int status =
      av_hwframe_map(exported->frame, hardware_frame, AV_HWFRAME_MAP_READ);
  if (status < 0)
    throw std::runtime_error("uploaded DRM PRIME map failed: " +
                             av_error(status));
  if (exported->frame->format != AV_PIX_FMT_DRM_PRIME ||
      !exported->frame->data[0])
    throw std::runtime_error("uploaded DRM PRIME map returned no descriptor");

  const auto *descriptor = reinterpret_cast<const AVDRMFrameDescriptor *>(
      exported->frame->data[0]);
  if (descriptor->nb_objects <= 0 || descriptor->nb_layers <= 0)
    throw std::runtime_error("uploaded DRM PRIME descriptor is empty");
  exported->width = hardware_frame->width;
  exported->height = hardware_frame->height;
  exported->sample_aspect_ratio = sample_aspect_ratio;
  exported->rotation_degrees = rotation_degrees;
  exported->pts_seconds = pts_seconds;
  exported->color_space = color_space;
  exported->color_range = color_range;
  exported->chroma_location = chroma_location;
  for (int index = 0; index < descriptor->nb_objects; ++index) {
    const auto &object = descriptor->objects[index];
    if (object.fd < 0)
      throw std::runtime_error("uploaded DRM PRIME object has an invalid fd");
    exported->objects.push_back(DmabufFrameObject{
        .fd = object.fd,
        .size = static_cast<std::uint64_t>(object.size),
        .modifier = object.format_modifier,
    });
  }
  for (int layer_index = 0; layer_index < descriptor->nb_layers;
       ++layer_index) {
    const auto &source = descriptor->layers[layer_index];
    DmabufFrameLayer layer;
    layer.format = source.format;
    layer.width =
        layer_index == 0 ? exported->width : (exported->width + 1) / 2;
    layer.height =
        layer_index == 0 ? exported->height : (exported->height + 1) / 2;
    for (int plane_index = 0; plane_index < source.nb_planes; ++plane_index) {
      const auto &plane = source.planes[plane_index];
      if (plane.object_index < 0 ||
          plane.object_index >= descriptor->nb_objects)
        throw std::runtime_error(
            "uploaded DRM PRIME plane references an invalid object");
      layer.planes.push_back(DmabufFramePlane{
          .object_index = plane.object_index,
          .offset = static_cast<std::uint32_t>(plane.offset),
          .pitch = static_cast<std::uint32_t>(plane.pitch),
      });
    }
    exported->layers.push_back(std::move(layer));
  }
  return DmabufVideoFrame(std::move(exported));
}

void DmabufFrameDecoder::seek(double position_seconds) {
  const double target =
      std::isfinite(position_seconds) ? std::max(0.0, position_seconds) : 0.0;
  const auto timestamp = static_cast<std::int64_t>(target * AV_TIME_BASE);
  const int status =
      av_seek_frame(impl_->format, -1, timestamp, AVSEEK_FLAG_BACKWARD);
  if (status < 0) {
    throw std::runtime_error("failed to seek hardware video: " +
                             av_error(status));
  }
  avcodec_flush_buffers(impl_->codec);
  av_packet_unref(impl_->packet);
  av_frame_unref(impl_->hardware_frame);
  impl_->input_eof = false;
  impl_->flush_sent = false;
  impl_->seek_target = target;
}

} // namespace localbooru::native_video
