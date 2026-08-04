// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace localbooru::native_video {

struct DecodedAudioChunk {
  int sample_rate = 0;
  int channels = 0;
  double pts_seconds = 0.0;
  std::vector<float> samples;
};

class AudioFrameDecoder {
 public:
  AudioFrameDecoder(const std::string& path, int output_sample_rate,
                    int output_channels, int stream_index = -1);
  ~AudioFrameDecoder();

  AudioFrameDecoder(const AudioFrameDecoder&) = delete;
  AudioFrameDecoder& operator=(const AudioFrameDecoder&) = delete;

  std::optional<DecodedAudioChunk> next_chunk();
  void seek(double position_seconds);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace localbooru::native_video
