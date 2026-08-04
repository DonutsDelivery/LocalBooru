// SPDX-License-Identifier: MIT
#pragma once

#include <SDL2/SDL.h>

#include "audio_decoder.h"

namespace localbooru::native_video {

void apply_audio_gain(DecodedAudioChunk& chunk, double gain);
void apply_audio_speed(DecodedAudioChunk& chunk, double speed);

class SdlAudioOutput {
 public:
  SdlAudioOutput() = default;
  ~SdlAudioOutput();

  SdlAudioOutput(const SdlAudioOutput&) = delete;
  SdlAudioOutput& operator=(const SdlAudioOutput&) = delete;

  void open(int sample_rate, int channels);
  void queue(const DecodedAudioChunk& chunk);
  void set_paused(bool paused);
  void clear();
  void close();

  [[nodiscard]] bool is_open() const { return device_ != 0; }
  [[nodiscard]] double queued_seconds() const;

 private:
  SDL_AudioDeviceID device_ = 0;
  int sample_rate_ = 0;
  int channels_ = 0;
  bool owns_audio_subsystem_ = false;
};

}  // namespace localbooru::native_video
