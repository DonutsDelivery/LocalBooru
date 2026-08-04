// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "decoder.h"

namespace localbooru::native_video {

struct PlaybackCallbacks {
  std::function<void(const DecodedVideoFrame&)> on_frame;
  std::function<void()> on_ended;
  std::function<void(const std::string&)> on_error;
};

struct PlaybackOpenOptions {
  bool autoplay = true;
  std::string interpolation_engine = "off";
  std::string interpolation_preset = "balanced";
  std::uint32_t target_fps = 60;
};

class VideoPlaybackSession {
 public:
  explicit VideoPlaybackSession(PlaybackCallbacks callbacks);
  ~VideoPlaybackSession();

  VideoPlaybackSession(const VideoPlaybackSession&) = delete;
  VideoPlaybackSession& operator=(const VideoPlaybackSession&) = delete;

  void open(const std::string& path, double resume_position, bool autoplay);
  void open(const std::string& path, double resume_position,
            PlaybackOpenOptions options);
  void set_paused(bool paused);
  void set_speed(double speed);
  void seek(double position_seconds);
  void stop();

  [[nodiscard]] double speed() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace localbooru::native_video
