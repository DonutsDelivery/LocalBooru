// SPDX-License-Identifier: MIT
#include "decoder.h"

#include <memory>
#include <optional>
#include <stdexcept>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/dict.h>
#include <libavutil/display.h>
#include <libswscale/swscale.h>
}

namespace localbooru::native_video {
namespace {
struct FormatCloser {
  void operator()(AVFormatContext *context) const {
    if (context)
      avformat_close_input(&context);
  }
};

std::string dictionary_value(AVDictionary *metadata, const char *key) {
  const auto *entry = av_dict_get(metadata, key, nullptr, 0);
  return entry && entry->value ? entry->value : "";
}

std::string color_space_for(const AVCodecParameters &parameters) {
  switch (parameters.color_space) {
  case AVCOL_SPC_BT709:
    return "bt709";
  case AVCOL_SPC_BT2020_NCL:
  case AVCOL_SPC_BT2020_CL:
    return "bt2020";
  case AVCOL_SPC_BT470BG:
  case AVCOL_SPC_SMPTE170M:
    return "bt601";
  default:
    return parameters.width >= 1280 || parameters.height > 576 ? "bt709"
                                                               : "bt601";
  }
}

std::string color_range_for(const AVCodecParameters &parameters) {
  return parameters.color_range == AVCOL_RANGE_JPEG ? "full" : "narrow";
}

std::string chroma_location_for(const AVCodecParameters &parameters) {
  switch (parameters.chroma_location) {
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
} // namespace

MediaMetadata probe_media(const std::string &path) {
  AVFormatContext *raw_context = nullptr;
  const int open_result =
      avformat_open_input(&raw_context, path.c_str(), nullptr, nullptr);
  if (open_result < 0) {
    char error[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(open_result, error, sizeof(error));
    throw std::runtime_error("failed to open media: " + std::string(error));
  }
  std::unique_ptr<AVFormatContext, FormatCloser> context(raw_context);
  const int info_result = avformat_find_stream_info(context.get(), nullptr);
  if (info_result < 0)
    throw std::runtime_error("failed to read media stream information");

  MediaMetadata result;
  if (context->duration != AV_NOPTS_VALUE) {
    result.duration_seconds =
        static_cast<double>(context->duration) / AV_TIME_BASE;
  }

  for (unsigned int index = 0; index < context->nb_streams; ++index) {
    auto *stream = context->streams[index];
    const auto *parameters = stream->codecpar;
    TrackMetadata track;
    track.index = static_cast<int>(index);
    track.language = dictionary_value(stream->metadata, "language");
    track.label = dictionary_value(stream->metadata, "title");

    switch (parameters->codec_type) {
    case AVMEDIA_TYPE_VIDEO: {
      track.kind = "video";
      if (result.width == 0) {
        result.width = parameters->width;
        result.height = parameters->height;
        result.color_space = color_space_for(*parameters);
        result.color_range = color_range_for(*parameters);
        result.chroma_location = chroma_location_for(*parameters);
        const AVRational sample_aspect_ratio =
            av_guess_sample_aspect_ratio(context.get(), stream, nullptr);
        if (sample_aspect_ratio.num > 0 && sample_aspect_ratio.den > 0) {
          result.sample_aspect_ratio = av_q2d(sample_aspect_ratio);
        }
        const auto *display_matrix_side_data = av_packet_side_data_get(
            parameters->coded_side_data, parameters->nb_coded_side_data,
            AV_PKT_DATA_DISPLAYMATRIX);
        if (display_matrix_side_data &&
            display_matrix_side_data->size >= 9 * sizeof(std::int32_t)) {
          const double clockwise_rotation =
              -av_display_rotation_get(reinterpret_cast<const std::int32_t *>(
                  display_matrix_side_data->data));
          result.rotation_degrees =
              static_cast<int>(std::llround(clockwise_rotation / 90.0)) * 90;
          result.rotation_degrees = (result.rotation_degrees % 360 + 360) % 360;
        }
        const AVRational rate =
            av_guess_frame_rate(context.get(), stream, nullptr);
        if (rate.num > 0 && rate.den > 0) {
          result.frame_rate = av_q2d(rate);
          result.frame_rate_numerator = rate.num;
          result.frame_rate_denominator = rate.den;
        }
      }
      break;
    }
    case AVMEDIA_TYPE_AUDIO:
      track.kind = "audio";
      break;
    case AVMEDIA_TYPE_SUBTITLE:
      track.kind = "subtitle";
      break;
    default:
      continue;
    }
    if (track.label.empty())
      track.label = track.kind + " " + std::to_string(index);
    result.tracks.push_back(std::move(track));
  }
  if (result.width == 0)
    throw std::runtime_error("media has no video stream");
  return result;
}

DecodedVideoFrame decode_first_video_frame(const std::string &path) {
  AVFormatContext *raw_format = nullptr;
  const int open_result =
      avformat_open_input(&raw_format, path.c_str(), nullptr, nullptr);
  if (open_result < 0) {
    char error[AV_ERROR_MAX_STRING_SIZE]{};
    av_strerror(open_result, error, sizeof(error));
    throw std::runtime_error("failed to open media: " + std::string(error));
  }
  std::unique_ptr<AVFormatContext, FormatCloser> format(raw_format);
  if (avformat_find_stream_info(format.get(), nullptr) < 0) {
    throw std::runtime_error("failed to read media stream information");
  }

  const AVCodec *codec = nullptr;
  const int stream_index =
      av_find_best_stream(format.get(), AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
  if (stream_index < 0 || codec == nullptr) {
    throw std::runtime_error("media has no decodable video stream");
  }

  AVCodecContext *codec_context = avcodec_alloc_context3(codec);
  if (codec_context == nullptr) {
    throw std::runtime_error("failed to allocate video decoder");
  }
  const auto codec_cleanup = [&] { avcodec_free_context(&codec_context); };
  try {
    if (avcodec_parameters_to_context(
            codec_context, format->streams[stream_index]->codecpar) < 0 ||
        avcodec_open2(codec_context, codec, nullptr) < 0) {
      throw std::runtime_error("failed to open video decoder");
    }

    AVPacket *packet = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    if (packet == nullptr || frame == nullptr) {
      if (packet)
        av_packet_free(&packet);
      if (frame)
        av_frame_free(&frame);
      throw std::runtime_error("failed to allocate decoder frames");
    }

    const auto convert = [&]() {
      SwsContext *scaler =
          sws_getContext(frame->width, frame->height,
                         static_cast<AVPixelFormat>(frame->format),
                         frame->width, frame->height, AV_PIX_FMT_RGBA,
                         SWS_BILINEAR, nullptr, nullptr, nullptr);
      if (scaler == nullptr) {
        throw std::runtime_error("failed to create RGBA frame converter");
      }
      DecodedVideoFrame decoded;
      decoded.width = frame->width;
      decoded.height = frame->height;
      decoded.rgba.resize(static_cast<std::size_t>(frame->width) *
                          static_cast<std::size_t>(frame->height) * 4U);
      std::uint8_t *destination[] = {decoded.rgba.data(), nullptr, nullptr,
                                     nullptr};
      int destination_stride[] = {frame->width * 4, 0, 0, 0};
      const int converted =
          sws_scale(scaler, frame->data, frame->linesize, 0, frame->height,
                    destination, destination_stride);
      sws_freeContext(scaler);
      if (converted != frame->height) {
        throw std::runtime_error("failed to convert decoded video frame");
      }
      const std::int64_t pts = frame->best_effort_timestamp;
      if (pts != AV_NOPTS_VALUE) {
        decoded.pts_seconds = static_cast<double>(pts) *
                              av_q2d(format->streams[stream_index]->time_base);
      }
      return decoded;
    };

    auto receive_frame = [&]() -> std::optional<DecodedVideoFrame> {
      const int result = avcodec_receive_frame(codec_context, frame);
      if (result == 0)
        return convert();
      if (result == AVERROR(EAGAIN) || result == AVERROR_EOF)
        return std::nullopt;
      throw std::runtime_error("failed while decoding video frame");
    };

    while (av_read_frame(format.get(), packet) >= 0) {
      if (packet->stream_index == stream_index) {
        const int send_result = avcodec_send_packet(codec_context, packet);
        av_packet_unref(packet);
        if (send_result < 0 && send_result != AVERROR(EAGAIN)) {
          throw std::runtime_error("failed to submit video packet to decoder");
        }
        if (auto decoded = receive_frame()) {
          av_frame_free(&frame);
          av_packet_free(&packet);
          codec_cleanup();
          return *decoded;
        }
      } else {
        av_packet_unref(packet);
      }
    }

    avcodec_send_packet(codec_context, nullptr);
    if (auto decoded = receive_frame()) {
      av_frame_free(&frame);
      av_packet_free(&packet);
      codec_cleanup();
      return *decoded;
    }
    av_frame_free(&frame);
    av_packet_free(&packet);
    throw std::runtime_error("video decoder produced no frame");
  } catch (...) {
    codec_cleanup();
    throw;
  }
}

} // namespace localbooru::native_video
