// SPDX-License-Identifier: MIT
#include "audio_output.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace localbooru::native_video {
namespace {
std::runtime_error sdl_error(const char* operation) {
  return std::runtime_error(std::string(operation) + ": " + SDL_GetError());
}
}  // namespace

void apply_audio_gain(DecodedAudioChunk& chunk, double gain) {
  const float value = static_cast<float>(
      std::isfinite(gain) ? std::clamp(gain, 0.0, 1.0) : 1.0);
  for (auto& sample : chunk.samples) sample *= value;
}

void apply_audio_speed(DecodedAudioChunk& chunk, double speed) {
  if (chunk.channels <= 0 || chunk.samples.empty()) return;
  const double value =
      std::isfinite(speed) ? std::clamp(speed, 0.5, 2.0) : 1.0;
  if (std::abs(value - 1.0) < 1e-9) return;
  const std::size_t channels = static_cast<std::size_t>(chunk.channels);
  const std::size_t input_frames = chunk.samples.size() / channels;
  const std::size_t output_frames = std::max<std::size_t>(
      1, static_cast<std::size_t>(std::llround(input_frames / value)));
  std::vector<float> adjusted(output_frames * channels);
  for (std::size_t output = 0; output < output_frames; ++output) {
    const std::size_t input =
        std::min(input_frames - 1,
                 static_cast<std::size_t>(static_cast<double>(output) * value));
    for (std::size_t channel = 0; channel < channels; ++channel) {
      adjusted[output * channels + channel] =
          chunk.samples[input * channels + channel];
    }
  }
  chunk.samples = std::move(adjusted);
}

SdlAudioOutput::~SdlAudioOutput() { close(); }

void SdlAudioOutput::open(int sample_rate, int channels) {
  if (sample_rate <= 0 || channels <= 0) {
    throw std::runtime_error("audio output format must be positive");
  }
  close();
  if ((SDL_WasInit(SDL_INIT_AUDIO) & SDL_INIT_AUDIO) == 0) {
    if (SDL_InitSubSystem(SDL_INIT_AUDIO) != 0) {
      throw sdl_error("failed to initialize SDL audio");
    }
    owns_audio_subsystem_ = true;
  }

  SDL_AudioSpec desired{};
  desired.freq = sample_rate;
  desired.format = AUDIO_F32SYS;
  desired.channels = static_cast<Uint8>(channels);
  desired.samples = 1024;
  SDL_AudioSpec obtained{};
  device_ = SDL_OpenAudioDevice(nullptr, 0, &desired, &obtained, 0);
  if (device_ == 0) throw sdl_error("failed to open SDL audio device");
  if (obtained.freq != sample_rate || obtained.channels != channels ||
      obtained.format != AUDIO_F32SYS) {
    close();
    throw std::runtime_error("SDL audio device did not accept float output format");
  }
  sample_rate_ = sample_rate;
  channels_ = channels;
  SDL_PauseAudioDevice(device_, 0);
}

void SdlAudioOutput::queue(const DecodedAudioChunk& chunk) {
  if (!is_open()) throw std::runtime_error("SDL audio output is not open");
  if (chunk.sample_rate != sample_rate_ || chunk.channels != channels_) {
    throw std::runtime_error("audio chunk format does not match SDL output");
  }
  if (chunk.samples.empty()) return;
  const auto bytes = static_cast<Uint32>(chunk.samples.size() * sizeof(float));
  if (SDL_QueueAudio(device_, chunk.samples.data(), bytes) != 0) {
    throw sdl_error("failed to queue SDL audio");
  }
}

void SdlAudioOutput::set_paused(bool paused) {
  if (is_open()) SDL_PauseAudioDevice(device_, paused ? 1 : 0);
}

void SdlAudioOutput::clear() {
  if (is_open()) SDL_ClearQueuedAudio(device_);
}

double SdlAudioOutput::queued_seconds() const {
  if (!is_open() || sample_rate_ <= 0 || channels_ <= 0) return 0.0;
  const double bytes_per_second = static_cast<double>(sample_rate_) *
                                  static_cast<double>(channels_) * sizeof(float);
  return static_cast<double>(SDL_GetQueuedAudioSize(device_)) / bytes_per_second;
}

void SdlAudioOutput::close() {
  if (device_ != 0) SDL_CloseAudioDevice(device_);
  device_ = 0;
  sample_rate_ = 0;
  channels_ = 0;
  if (owns_audio_subsystem_) {
    SDL_QuitSubSystem(SDL_INIT_AUDIO);
    owns_audio_subsystem_ = false;
  }
}

}  // namespace localbooru::native_video
