// SPDX-License-Identifier: MIT
#include "subtitles.h"

#include <algorithm>
#include <regex>
#include <sstream>
#include <stdexcept>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
}

namespace localbooru::native_video {
namespace {
double parse_timestamp(const std::string &value) {
  static const std::regex pattern(R"((?:(\d+):)?(\d{2}):(\d{2})\.(\d{3}))");
  std::smatch match;
  if (!std::regex_match(value, match, pattern))
    throw std::runtime_error("invalid WebVTT timestamp: " + value);
  const double hours = match[1].matched ? std::stod(match[1].str()) : 0.0;
  return hours * 3600.0 + std::stod(match[2].str()) * 60.0 +
         std::stod(match[3].str()) + std::stod(match[4].str()) / 1000.0;
}

std::string trim_cr(std::string line) {
  if (!line.empty() && line.back() == '\r')
    line.pop_back();
  return line;
}

std::string subtitle_plain_text(const AVSubtitleRect &rect) {
  std::string text;
  if (rect.text) {
    text = rect.text;
  } else if (rect.ass) {
    text = rect.ass;
    std::size_t offset = 0;
    for (int field = 0; field < 8; ++field) {
      const auto comma = text.find(',', offset);
      if (comma == std::string::npos)
        break;
      offset = comma + 1;
    }
    text.erase(0, offset);
  }
  text = std::regex_replace(text, std::regex(R"(\{[^}]*\})"), "");
  text = std::regex_replace(text, std::regex(R"(\\[Nn])"), "\n");
  return text;
}

void replace_all(std::string &value, const std::string &from,
                 const std::string &to) {
  std::size_t offset = 0;
  while ((offset = value.find(from, offset)) != std::string::npos) {
    value.replace(offset, from.size(), to);
    offset += to.size();
  }
}

std::string webvtt_plain_text(std::string text) {
  text = std::regex_replace(text, std::regex(R"(<[^>]*>)"), "");
  replace_all(text, "&amp;", "&");
  replace_all(text, "&lt;", "<");
  replace_all(text, "&gt;", ">");
  replace_all(text, "&nbsp;", " ");
  return text;
}
} // namespace

SubtitleTrack SubtitleTrack::from_webvtt(const std::string &source) {
  std::istringstream input(source);
  std::string line;
  std::getline(input, line);
  if (trim_cr(line).rfind("WEBVTT", 0) != 0)
    throw std::runtime_error("subtitle track is not WebVTT");

  SubtitleTrack track;
  while (std::getline(input, line)) {
    line = trim_cr(line);
    if (line.empty())
      continue;
    if (line.find("-->") == std::string::npos) {
      if (!std::getline(input, line))
        break;
      line = trim_cr(line);
    }
    const auto arrow = line.find("-->");
    if (arrow == std::string::npos)
      continue;
    std::string start = line.substr(0, arrow);
    std::string end = line.substr(arrow + 3);
    start.erase(start.find_last_not_of(" \t") + 1);
    end.erase(0, end.find_first_not_of(" \t"));
    if (const auto settings = end.find_first_of(" \t");
        settings != std::string::npos)
      end.resize(settings);

    SubtitleCue cue{parse_timestamp(start), parse_timestamp(end), {}};
    while (std::getline(input, line)) {
      line = trim_cr(line);
      if (line.empty())
        break;
      if (!cue.text.empty())
        cue.text += '\n';
      cue.text += line;
    }
    cue.text = webvtt_plain_text(std::move(cue.text));
    if (cue.end_seconds >= cue.start_seconds && !cue.text.empty())
      track.cues_.push_back(std::move(cue));
  }
  std::stable_sort(track.cues_.begin(), track.cues_.end(),
                   [](const auto &left, const auto &right) {
                     return left.start_seconds < right.start_seconds;
                   });
  return track;
}

SubtitleTrack SubtitleTrack::from_embedded(const std::string &media_path,
                                           int stream_index) {
  AVFormatContext *format = nullptr;
  AVCodecContext *decoder = nullptr;
  AVPacket *packet = nullptr;
  try {
    if (avformat_open_input(&format, media_path.c_str(), nullptr, nullptr) <
            0 ||
        avformat_find_stream_info(format, nullptr) < 0) {
      throw std::runtime_error("failed to open embedded subtitle source");
    }
    if (stream_index < 0 ||
        stream_index >= static_cast<int>(format->nb_streams) ||
        format->streams[stream_index]->codecpar->codec_type !=
            AVMEDIA_TYPE_SUBTITLE) {
      throw std::runtime_error("invalid embedded subtitle stream");
    }
    AVStream *stream = format->streams[stream_index];
    const AVCodec *codec = avcodec_find_decoder(stream->codecpar->codec_id);
    if (!codec)
      throw std::runtime_error("unsupported embedded subtitle codec");
    decoder = avcodec_alloc_context3(codec);
    if (!decoder ||
        avcodec_parameters_to_context(decoder, stream->codecpar) < 0 ||
        avcodec_open2(decoder, codec, nullptr) < 0) {
      throw std::runtime_error(
          "failed to initialize embedded subtitle decoder");
    }
    packet = av_packet_alloc();
    if (!packet)
      throw std::runtime_error("failed to allocate subtitle packet");

    SubtitleTrack track;
    while (av_read_frame(format, packet) >= 0) {
      if (packet->stream_index == stream_index) {
        AVSubtitle subtitle{};
        int decoded = 0;
        const int result =
            avcodec_decode_subtitle2(decoder, &subtitle, &decoded, packet);
        if (result < 0) {
          av_packet_unref(packet);
          throw std::runtime_error("failed to decode embedded subtitle packet");
        }
        if (decoded) {
          const double packet_time =
              packet->pts == AV_NOPTS_VALUE
                  ? 0.0
                  : packet->pts * av_q2d(stream->time_base);
          const double start =
              packet_time +
              static_cast<double>(subtitle.start_display_time) / 1000.0;
          double end = packet_time +
                       static_cast<double>(subtitle.end_display_time) / 1000.0;
          if (end <= start && packet->duration > 0) {
            end = packet_time + packet->duration * av_q2d(stream->time_base);
          }
          for (unsigned int index = 0; index < subtitle.num_rects; ++index) {
            auto text = subtitle_plain_text(*subtitle.rects[index]);
            if (!text.empty() && end >= start) {
              track.cues_.push_back({start, end, std::move(text)});
            }
          }
          avsubtitle_free(&subtitle);
        }
      }
      av_packet_unref(packet);
    }
    std::stable_sort(track.cues_.begin(), track.cues_.end(),
                     [](const auto &left, const auto &right) {
                       return left.start_seconds < right.start_seconds;
                     });
    av_packet_free(&packet);
    avcodec_free_context(&decoder);
    avformat_close_input(&format);
    return track;
  } catch (...) {
    av_packet_free(&packet);
    avcodec_free_context(&decoder);
    if (format)
      avformat_close_input(&format);
    throw;
  }
}

std::vector<std::string> SubtitleTrack::text_at(double position_seconds,
                                                double delay_seconds) const {
  const double cue_time = position_seconds - delay_seconds;
  std::vector<std::string> result;
  for (const auto &cue : cues_) {
    if (cue.start_seconds > cue_time)
      break;
    if (cue_time >= cue.start_seconds && cue_time < cue.end_seconds)
      result.push_back(cue.text);
  }
  return result;
}

} // namespace localbooru::native_video
