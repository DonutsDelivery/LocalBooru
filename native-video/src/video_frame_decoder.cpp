// SPDX-License-Identifier: MIT
#include "decoder.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/hwcontext.h>
#include <libswscale/swscale.h>
}

namespace localbooru::native_video {
namespace {
std::string av_error(int code) {
  char buffer[AV_ERROR_MAX_STRING_SIZE]{};
  av_strerror(code, buffer, sizeof(buffer));
  return buffer;
}

AVPixelFormat select_vaapi_format(AVCodecContext*, const AVPixelFormat* formats) {
  for (const AVPixelFormat* format = formats; *format != AV_PIX_FMT_NONE;
       ++format) {
    if (*format == AV_PIX_FMT_VAAPI) return *format;
  }
  return formats[0];
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

struct VideoFrameDecoder::Impl {
  AVFormatContext* format = nullptr;
  AVCodecContext* codec = nullptr;
  AVPacket* packet = nullptr;
  AVFrame* frame = nullptr;
  AVFrame* software_frame = nullptr;
  AVBufferRef* hardware_device = nullptr;
  SwsContext* scaler = nullptr;
  int stream_index = -1;
  bool input_eof = false;
  bool flush_sent = false;
  double seek_target = 0.0;

  ~Impl() {
    if (scaler) sws_freeContext(scaler);
    if (software_frame) av_frame_free(&software_frame);
    if (frame) av_frame_free(&frame);
    if (packet) av_packet_free(&packet);
    if (codec) avcodec_free_context(&codec);
    if (hardware_device) av_buffer_unref(&hardware_device);
    if (format) avformat_close_input(&format);
  }

  DecodedVideoFrame convert() {
    AVFrame* source = frame;
    if (frame->format == AV_PIX_FMT_VAAPI) {
      av_frame_unref(software_frame);
      const int transferred = av_hwframe_transfer_data(software_frame, frame, 0);
      if (transferred < 0) {
        throw std::runtime_error("failed to download hardware frame: " +
                                 av_error(transferred));
      }
      av_frame_copy_props(software_frame, frame);
      source = software_frame;
    }
    DecodedVideoFrame decoded;
    decoded.width = source->width;
    decoded.height = source->height;
    decoded.yuv420p = true;
    const std::size_t luma_size =
        static_cast<std::size_t>(source->width) * source->height;
    decoded.rgba.resize(luma_size + luma_size / 2U);
    std::uint8_t* destination[] = {
        decoded.rgba.data(), decoded.rgba.data() + luma_size,
        decoded.rgba.data() + luma_size + luma_size / 4U, nullptr};
    int destination_stride[] = {source->width, source->width / 2,
                                source->width / 2, 0};
    const auto source_format = static_cast<AVPixelFormat>(source->format);
    int converted = source->height;
    if (source_format == AV_PIX_FMT_YUV420P ||
        source_format == AV_PIX_FMT_YUVJ420P) {
      const auto copy_plane = [](std::uint8_t* output, int output_stride,
                                 const std::uint8_t* input, int input_stride,
                                 int row_bytes, int rows) {
        for (int row = 0; row < rows; ++row) {
          std::copy_n(input + row * input_stride, row_bytes,
                      output + row * output_stride);
        }
      };
      copy_plane(destination[0], destination_stride[0], source->data[0],
                 source->linesize[0], source->width, source->height);
      copy_plane(destination[1], destination_stride[1], source->data[1],
                 source->linesize[1], source->width / 2, source->height / 2);
      copy_plane(destination[2], destination_stride[2], source->data[2],
                 source->linesize[2], source->width / 2, source->height / 2);
    } else {
      scaler = sws_getCachedContext(
          scaler, source->width, source->height, source_format, source->width,
          source->height, AV_PIX_FMT_YUV420P, SWS_BILINEAR, nullptr, nullptr,
          nullptr);
      if (!scaler)
        throw std::runtime_error("failed to create YUV420 frame converter");
      converted =
          sws_scale(scaler, source->data, source->linesize, 0, source->height,
                    destination, destination_stride);
    }
    if (converted != source->height) {
      throw std::runtime_error("failed to convert decoded YUV420 frame");
    }

    if (source->best_effort_timestamp != AV_NOPTS_VALUE) {
      decoded.pts_seconds =
          static_cast<double>(source->best_effort_timestamp) *
          av_q2d(format->streams[stream_index]->time_base);
    }
    return decoded;
  }
};

VideoFrameDecoder::VideoFrameDecoder(const std::string& path)
    : impl_(std::make_unique<Impl>()) {
  int result = avformat_open_input(&impl_->format, path.c_str(), nullptr, nullptr);
  if (result < 0) {
    throw std::runtime_error("failed to open media: " + av_error(result));
  }
  result = avformat_find_stream_info(impl_->format, nullptr);
  if (result < 0) {
    throw std::runtime_error("failed to read media stream information: " +
                             av_error(result));
  }

  const AVCodec* decoder = nullptr;
  impl_->stream_index = av_find_best_stream(
      impl_->format, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
  if (impl_->stream_index < 0 || !decoder) {
    throw std::runtime_error("media has no decodable video stream");
  }

  impl_->codec = avcodec_alloc_context3(decoder);
  if (!impl_->codec) throw std::runtime_error("failed to allocate video decoder");
  result = avcodec_parameters_to_context(
      impl_->codec, impl_->format->streams[impl_->stream_index]->codecpar);
  if (result < 0) {
    throw std::runtime_error("failed to configure video decoder: " +
                             av_error(result));
  }
  if (std::getenv("LOCALBOORU_NATIVE_VIDEO_HW_DOWNLOAD") != nullptr &&
      decoder_supports_vaapi(decoder) &&
      av_hwdevice_ctx_create(&impl_->hardware_device, AV_HWDEVICE_TYPE_VAAPI,
                             nullptr, nullptr, 0) == 0) {
    impl_->codec->get_format = select_vaapi_format;
    impl_->codec->hw_device_ctx = av_buffer_ref(impl_->hardware_device);
  }
  result = avcodec_open2(impl_->codec, decoder, nullptr);
  if (result < 0) {
    throw std::runtime_error("failed to open video decoder: " + av_error(result));
  }

  impl_->packet = av_packet_alloc();
  impl_->frame = av_frame_alloc();
  impl_->software_frame = av_frame_alloc();
  if (!impl_->packet || !impl_->frame || !impl_->software_frame) {
    throw std::runtime_error("failed to allocate decoder frames");
  }
}

VideoFrameDecoder::~VideoFrameDecoder() = default;

std::optional<DecodedVideoFrame> VideoFrameDecoder::next_frame() {
  for (;;) {
    const int receive = avcodec_receive_frame(impl_->codec, impl_->frame);
    if (receive == 0) {
      auto decoded = impl_->convert();
      if (decoded.pts_seconds + 1e-6 < impl_->seek_target) continue;
      impl_->seek_target = 0.0;
      return decoded;
    }
    if (receive == AVERROR_EOF) return std::nullopt;
    if (receive != AVERROR(EAGAIN)) {
      throw std::runtime_error("failed while decoding video frame: " +
                               av_error(receive));
    }

    if (impl_->input_eof) {
      if (impl_->flush_sent) return std::nullopt;
      const int send = avcodec_send_packet(impl_->codec, nullptr);
      if (send < 0 && send != AVERROR_EOF) {
        throw std::runtime_error("failed to flush video decoder: " +
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

void VideoFrameDecoder::seek(double position_seconds) {
  const double target =
      std::isfinite(position_seconds) ? std::max(0.0, position_seconds) : 0.0;
  const auto timestamp = static_cast<std::int64_t>(target * AV_TIME_BASE);
  const int result =
      av_seek_frame(impl_->format, -1, timestamp, AVSEEK_FLAG_BACKWARD);
  if (result < 0) {
    throw std::runtime_error("failed to seek video: " + av_error(result));
  }
  avcodec_flush_buffers(impl_->codec);
  av_packet_unref(impl_->packet);
  av_frame_unref(impl_->frame);
  impl_->input_eof = false;
  impl_->flush_sent = false;
  impl_->seek_target = target;
}

}  // namespace localbooru::native_video
