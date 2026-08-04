// SPDX-License-Identifier: MIT
#pragma once

#include <string>
#include <vector>

namespace localbooru::native_video {

struct SubtitleCue {
  double start_seconds = 0.0;
  double end_seconds = 0.0;
  std::string text;
};

class SubtitleTrack {
public:
  static SubtitleTrack from_webvtt(const std::string &source);
  static SubtitleTrack from_embedded(const std::string &media_path,
                                     int stream_index);
  [[nodiscard]] std::vector<std::string>
  text_at(double position_seconds, double delay_seconds = 0.0) const;
  [[nodiscard]] const std::vector<SubtitleCue> &cues() const { return cues_; }

private:
  std::vector<SubtitleCue> cues_;
};

} // namespace localbooru::native_video
