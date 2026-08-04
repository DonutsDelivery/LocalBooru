// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace localbooru::native_video {

struct TrackMetadata {
  int index = -1;
  std::string kind;
  std::string language;
  std::string label;
};

struct MediaMetadata {
  double duration_seconds = 0.0;
  int width = 0;
  int height = 0;
  double sample_aspect_ratio = 1.0;
  int rotation_degrees = 0;
  std::string color_space;
  std::string color_range;
  std::string chroma_location;
  double frame_rate = 0.0;
  int frame_rate_numerator = 0;
  int frame_rate_denominator = 1;
  std::vector<TrackMetadata> tracks;
};

struct DecodedVideoFrame {
  int width = 0;
  int height = 0;
  double pts_seconds = 0.0;
  bool yuv420p = false;
  std::vector<std::uint8_t> rgba;
};

class VideoFrameDecoder {
public:
  explicit VideoFrameDecoder(const std::string &path);
  ~VideoFrameDecoder();

  VideoFrameDecoder(const VideoFrameDecoder &) = delete;
  VideoFrameDecoder &operator=(const VideoFrameDecoder &) = delete;

  std::optional<DecodedVideoFrame> next_frame();
  void seek(double position_seconds);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

MediaMetadata probe_media(const std::string &path);
DecodedVideoFrame decode_first_video_frame(const std::string &path);

} // namespace localbooru::native_video
