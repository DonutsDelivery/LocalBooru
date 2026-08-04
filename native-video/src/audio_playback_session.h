// SPDX-License-Identifier: MIT
#pragma once

#include <memory>
#include <string>

namespace localbooru::native_video {

class AudioPlaybackSession {
 public:
  AudioPlaybackSession();
  ~AudioPlaybackSession();

  AudioPlaybackSession(const AudioPlaybackSession&) = delete;
  AudioPlaybackSession& operator=(const AudioPlaybackSession&) = delete;

  void open(const std::string& path, double resume_position, bool autoplay,
            int stream_index = -1);
  void set_paused(bool paused);
  void set_volume(double volume);
  void set_speed(double speed);
  void seek(double position_seconds);
  void stop();

  [[nodiscard]] double submitted_seconds() const;
  [[nodiscard]] double playback_position() const;
  [[nodiscard]] double volume() const;
  [[nodiscard]] double speed() const;
  [[nodiscard]] std::string last_error() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace localbooru::native_video
