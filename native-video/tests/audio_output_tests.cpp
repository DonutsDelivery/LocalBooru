// SPDX-License-Identifier: MIT
#include <cassert>
#include <cstdlib>
#include <iostream>

#include "audio_output.h"

using namespace localbooru::native_video;

int main() {
  setenv("SDL_AUDIODRIVER", "dummy", 1);

  SdlAudioOutput output;
  output.open(48000, 2);
  assert(output.is_open());

  DecodedAudioChunk chunk;
  chunk.sample_rate = 48000;
  chunk.channels = 2;
  chunk.samples.assign(4800U * 2U, 0.25F);
  apply_audio_gain(chunk, 0.5);
  assert(chunk.samples.front() == 0.125F);
  const auto original_samples = chunk.samples.size();
  apply_audio_speed(chunk, 2.0);
  assert(chunk.samples.size() == original_samples / 2);
  output.queue(chunk);
  assert(output.queued_seconds() > 0.02);

  output.set_paused(true);
  output.clear();
  assert(output.queued_seconds() == 0.0);
  output.close();
  assert(!output.is_open());

  std::cout << "native video audio output tests passed\n";
}
