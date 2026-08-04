// SPDX-License-Identifier: MIT
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <iostream>

#include "audio_decoder.h"

using namespace localbooru::native_video;

int main() {
  const auto fixture = std::filesystem::temp_directory_path() /
                       "localbooru-native-video-audio-test.mka";
  const std::string command =
      "ffmpeg -hide_banner -loglevel error -y -f lavfi -i "
      "sine=frequency=440:sample_rate=48000:duration=0.25 -f lavfi -i "
      "sine=frequency=880:sample_rate=48000:duration=0.25 "
      "-map 0:a -map 1:a -c:a aac \"" +
      fixture.string() + "\"";
  assert(std::system(command.c_str()) == 0);

  AudioFrameDecoder decoder(fixture.string(), 48000, 2);
  std::size_t sample_count = 0;
  double previous_pts = -1.0;
  while (auto chunk = decoder.next_chunk()) {
    assert(chunk->sample_rate == 48000);
    assert(chunk->channels == 2);
    assert(chunk->pts_seconds >= previous_pts);
    assert(chunk->samples.size() % 2 == 0);
    previous_pts = chunk->pts_seconds;
    sample_count += chunk->samples.size();
  }
  assert(sample_count >= 48000U / 4U * 2U);

  decoder.seek(0.1);
  const auto after_seek = decoder.next_chunk();
  assert(after_seek.has_value());
  assert(after_seek->pts_seconds >= 0.09);

  AudioFrameDecoder second_track(fixture.string(), 48000, 2, 1);
  assert(second_track.next_chunk().has_value());
  bool rejected_non_audio_or_missing_stream = false;
  try {
    AudioFrameDecoder invalid_track(fixture.string(), 48000, 2, 99);
  } catch (const std::exception&) {
    rejected_non_audio_or_missing_stream = true;
  }
  assert(rejected_non_audio_or_missing_stream);

  std::filesystem::remove(fixture);
  std::cout << "native video audio decoder tests passed\n";
}
