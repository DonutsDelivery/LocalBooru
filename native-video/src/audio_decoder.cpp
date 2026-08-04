// SPDX-License-Identifier: MIT
#include "audio_decoder.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswresample/swresample.h>
}

namespace localbooru::native_video {
namespace {
std::string av_error(int code) {
  char buffer[AV_ERROR_MAX_STRING_SIZE]{};
  av_strerror(code, buffer, sizeof(buffer));
  return buffer;
}
}  // namespace

struct AudioFrameDecoder::Impl {
  AVFormatContext* format = nullptr;
  AVCodecContext* codec = nullptr;
  AVPacket* packet = nullptr;
  AVFrame* frame = nullptr;
  SwrContext* resampler = nullptr;
  int stream_index = -1;
  int output_sample_rate = 0;
  int output_channels = 0;
  bool input_eof = false;
  bool flush_sent = false;
  double seek_target = 0.0;

  ~Impl() {
    if (resampler) swr_free(&resampler);
    if (frame) av_frame_free(&frame);
    if (packet) av_packet_free(&packet);
    if (codec) avcodec_free_context(&codec);
    if (format) avformat_close_input(&format);
  }

  DecodedAudioChunk convert() {
    const int output_capacity = static_cast<int>(av_rescale_rnd(
        swr_get_delay(resampler, codec->sample_rate) + frame->nb_samples,
        output_sample_rate, codec->sample_rate, AV_ROUND_UP));
    if (output_capacity <= 0) {
      throw std::runtime_error("invalid resampled audio capacity");
    }

    DecodedAudioChunk chunk;
    chunk.sample_rate = output_sample_rate;
    chunk.channels = output_channels;
    chunk.samples.resize(static_cast<std::size_t>(output_capacity) *
                         static_cast<std::size_t>(output_channels));
    std::uint8_t* output[] = {
        reinterpret_cast<std::uint8_t*>(chunk.samples.data())};
    const int converted =
        swr_convert(resampler, output, output_capacity,
                    const_cast<const std::uint8_t**>(frame->extended_data),
                    frame->nb_samples);
    if (converted < 0) {
      throw std::runtime_error("failed to resample audio frame: " +
                               av_error(converted));
    }
    chunk.samples.resize(static_cast<std::size_t>(converted) *
                         static_cast<std::size_t>(output_channels));
    if (frame->best_effort_timestamp != AV_NOPTS_VALUE) {
      chunk.pts_seconds =
          static_cast<double>(frame->best_effort_timestamp) *
          av_q2d(format->streams[stream_index]->time_base);
    }
    return chunk;
  }
};

AudioFrameDecoder::AudioFrameDecoder(const std::string& path,
                                     int output_sample_rate,
                                     int output_channels, int stream_index)
    : impl_(std::make_unique<Impl>()) {
  if (output_sample_rate <= 0 || output_channels <= 0) {
    throw std::runtime_error("audio output format must be positive");
  }
  impl_->output_sample_rate = output_sample_rate;
  impl_->output_channels = output_channels;

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
  if (stream_index >= 0) {
    if (stream_index >= static_cast<int>(impl_->format->nb_streams) ||
        impl_->format->streams[stream_index]->codecpar->codec_type !=
            AVMEDIA_TYPE_AUDIO) {
      throw std::runtime_error("selected audio stream is invalid");
    }
    impl_->stream_index = stream_index;
    decoder = avcodec_find_decoder(
        impl_->format->streams[stream_index]->codecpar->codec_id);
  } else {
    impl_->stream_index = av_find_best_stream(
        impl_->format, AVMEDIA_TYPE_AUDIO, -1, -1, &decoder, 0);
  }
  if (impl_->stream_index < 0 || !decoder) {
    throw std::runtime_error("media has no decodable audio stream");
  }

  impl_->codec = avcodec_alloc_context3(decoder);
  if (!impl_->codec) throw std::runtime_error("failed to allocate audio decoder");
  result = avcodec_parameters_to_context(
      impl_->codec, impl_->format->streams[impl_->stream_index]->codecpar);
  if (result < 0 || avcodec_open2(impl_->codec, decoder, nullptr) < 0) {
    throw std::runtime_error("failed to open audio decoder");
  }

  AVChannelLayout output_layout{};
  av_channel_layout_default(&output_layout, output_channels);
  result = swr_alloc_set_opts2(
      &impl_->resampler, &output_layout, AV_SAMPLE_FMT_FLT, output_sample_rate,
      &impl_->codec->ch_layout, impl_->codec->sample_fmt,
      impl_->codec->sample_rate, 0, nullptr);
  av_channel_layout_uninit(&output_layout);
  if (result < 0 || !impl_->resampler || swr_init(impl_->resampler) < 0) {
    throw std::runtime_error("failed to initialize audio resampler");
  }

  impl_->packet = av_packet_alloc();
  impl_->frame = av_frame_alloc();
  if (!impl_->packet || !impl_->frame) {
    throw std::runtime_error("failed to allocate audio decoder frames");
  }
}

AudioFrameDecoder::~AudioFrameDecoder() = default;

std::optional<DecodedAudioChunk> AudioFrameDecoder::next_chunk() {
  for (;;) {
    const int receive = avcodec_receive_frame(impl_->codec, impl_->frame);
    if (receive == 0) {
      auto chunk = impl_->convert();
      if (chunk.pts_seconds + 1e-6 < impl_->seek_target) continue;
      impl_->seek_target = 0.0;
      return chunk;
    }
    if (receive == AVERROR_EOF) return std::nullopt;
    if (receive != AVERROR(EAGAIN)) {
      throw std::runtime_error("failed while decoding audio frame: " +
                               av_error(receive));
    }

    if (impl_->input_eof) {
      if (impl_->flush_sent) return std::nullopt;
      const int send = avcodec_send_packet(impl_->codec, nullptr);
      if (send < 0 && send != AVERROR_EOF) {
        throw std::runtime_error("failed to flush audio decoder: " +
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
        throw std::runtime_error("failed to submit audio packet: " +
                                 av_error(send));
      }
      break;
    } while (read >= 0);
  }
}

void AudioFrameDecoder::seek(double position_seconds) {
  const double target =
      std::isfinite(position_seconds) ? std::max(0.0, position_seconds) : 0.0;
  const auto timestamp = static_cast<std::int64_t>(target * AV_TIME_BASE);
  const int result =
      av_seek_frame(impl_->format, -1, timestamp, AVSEEK_FLAG_BACKWARD);
  if (result < 0) {
    throw std::runtime_error("failed to seek audio: " + av_error(result));
  }
  avcodec_flush_buffers(impl_->codec);
  swr_close(impl_->resampler);
  if (swr_init(impl_->resampler) < 0) {
    throw std::runtime_error("failed to reset audio resampler after seek");
  }
  av_packet_unref(impl_->packet);
  av_frame_unref(impl_->frame);
  impl_->input_eof = false;
  impl_->flush_sent = false;
  impl_->seek_target = target;
}

}  // namespace localbooru::native_video
