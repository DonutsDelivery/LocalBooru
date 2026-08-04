// SPDX-License-Identifier: MIT
#include "dmabuf_probe.h"

#include <cerrno>
#include <cstring>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_drm.h>
#include <libavutil/pixfmt.h>
}

namespace localbooru::native_video {
namespace {

std::string av_error(int code) {
  char buffer[AV_ERROR_MAX_STRING_SIZE]{};
  av_strerror(code, buffer, sizeof(buffer));
  return buffer;
}

AVPixelFormat select_vaapi_format(AVCodecContext*,
                                  const AVPixelFormat* formats) {
  for (const AVPixelFormat* format = formats; *format != AV_PIX_FMT_NONE;
       ++format) {
    if (*format == AV_PIX_FMT_VAAPI) return *format;
  }
  return AV_PIX_FMT_NONE;
}

bool decoder_supports_vaapi(const AVCodec* decoder) {
  for (int index = 0;; ++index) {
    const AVCodecHWConfig* config = avcodec_get_hw_config(decoder, index);
    if (!config) return false;
    if (config->device_type == AV_HWDEVICE_TYPE_VAAPI &&
        (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) != 0) {
      return true;
    }
  }
}

}  // namespace

DmabufProbeResult probe_dmabuf_export(const std::string& media_path,
                                      const std::string& render_node) {
  DmabufProbeResult result;
  result.device = render_node;
  AVFormatContext* format = nullptr;
  AVCodecContext* codec = nullptr;
  AVBufferRef* hardware_device = nullptr;
  AVPacket* packet = nullptr;
  AVFrame* hardware_frame = nullptr;
  AVFrame* drm_frame = nullptr;

  const auto finish = [&](std::string reason) {
    result.reason = std::move(reason);
    if (drm_frame) av_frame_free(&drm_frame);
    if (hardware_frame) av_frame_free(&hardware_frame);
    if (packet) av_packet_free(&packet);
    if (codec) avcodec_free_context(&codec);
    if (hardware_device) av_buffer_unref(&hardware_device);
    if (format) avformat_close_input(&format);
    return result;
  };

  int status = avformat_open_input(&format, media_path.c_str(), nullptr, nullptr);
  if (status < 0) return finish("open failed: " + av_error(status));
  status = avformat_find_stream_info(format, nullptr);
  if (status < 0) return finish("stream probe failed: " + av_error(status));

  const AVCodec* decoder = nullptr;
  const int stream_index =
      av_find_best_stream(format, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
  if (stream_index < 0 || !decoder) return finish("no video decoder");
  if (!decoder_supports_vaapi(decoder)) {
    return finish("decoder does not expose a VA-API hardware configuration");
  }

  status = av_hwdevice_ctx_create(&hardware_device, AV_HWDEVICE_TYPE_VAAPI,
                                  render_node.c_str(), nullptr, 0);
  if (status < 0) {
    return finish("VA-API device creation failed: " + av_error(status));
  }

  codec = avcodec_alloc_context3(decoder);
  if (!codec) return finish("codec allocation failed");
  status = avcodec_parameters_to_context(codec,
                                         format->streams[stream_index]->codecpar);
  if (status < 0) return finish("codec setup failed: " + av_error(status));
  codec->get_format = select_vaapi_format;
  codec->hw_device_ctx = av_buffer_ref(hardware_device);
  if (!codec->hw_device_ctx) return finish("hardware device reference failed");
  status = avcodec_open2(codec, decoder, nullptr);
  if (status < 0) return finish("hardware decoder open failed: " + av_error(status));

  packet = av_packet_alloc();
  hardware_frame = av_frame_alloc();
  drm_frame = av_frame_alloc();
  if (!packet || !hardware_frame || !drm_frame) {
    return finish("frame allocation failed");
  }

  bool submitted_eof = false;
  for (;;) {
    status = avcodec_receive_frame(codec, hardware_frame);
    if (status == 0) break;
    if (status != AVERROR(EAGAIN)) {
      return finish("hardware decode failed: " + av_error(status));
    }
    if (submitted_eof) return finish("media ended before a frame was decoded");

    status = av_read_frame(format, packet);
    if (status < 0) {
      status = avcodec_send_packet(codec, nullptr);
      submitted_eof = true;
    } else if (packet->stream_index == stream_index) {
      status = avcodec_send_packet(codec, packet);
    } else {
      status = 0;
    }
    av_packet_unref(packet);
    if (status < 0 && status != AVERROR(EAGAIN) && status != AVERROR_EOF) {
      return finish("packet submission failed: " + av_error(status));
    }
  }

  if (hardware_frame->format != AV_PIX_FMT_VAAPI) {
    return finish("decoder returned a non-VA-API frame");
  }
  drm_frame->format = AV_PIX_FMT_DRM_PRIME;
  status = av_hwframe_map(drm_frame, hardware_frame, AV_HWFRAME_MAP_READ);
  if (status < 0) {
    return finish("DRM PRIME map failed: " + av_error(status));
  }
  if (drm_frame->format != AV_PIX_FMT_DRM_PRIME || !drm_frame->data[0]) {
    return finish("DRM PRIME map returned no descriptor");
  }

  const auto* descriptor =
      reinterpret_cast<const AVDRMFrameDescriptor*>(drm_frame->data[0]);
  if (descriptor->nb_objects <= 0 || descriptor->nb_layers <= 0) {
    return finish("DRM PRIME descriptor is empty");
  }
  result.width = hardware_frame->width;
  result.height = hardware_frame->height;
  for (int index = 0; index < descriptor->nb_objects; ++index) {
    const auto& object = descriptor->objects[index];
    if (object.fd < 0) return finish("DRM PRIME object has an invalid fd");
    result.objects.push_back(DmabufObjectProbe{
        .size = static_cast<std::uint64_t>(object.size),
        .modifier = object.format_modifier,
    });
  }
  for (int layer_index = 0; layer_index < descriptor->nb_layers;
       ++layer_index) {
    const auto& source = descriptor->layers[layer_index];
    DmabufLayerProbe layer;
    layer.format = source.format;
    for (int plane_index = 0; plane_index < source.nb_planes; ++plane_index) {
      const auto& plane = source.planes[plane_index];
      if (plane.object_index < 0 ||
          plane.object_index >= descriptor->nb_objects) {
        return finish("DRM PRIME plane references an invalid object");
      }
      layer.planes.push_back(DmabufPlaneProbe{
          .object_index = plane.object_index,
          .offset = static_cast<std::uint32_t>(plane.offset),
          .pitch = static_cast<std::uint32_t>(plane.pitch),
      });
    }
    result.layers.push_back(std::move(layer));
  }
  result.available = true;
  return finish("");
}

}  // namespace localbooru::native_video
