// SPDX-License-Identifier: MIT
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "decoder.h"

using namespace localbooru::native_video;

int main() {
  bool failed = false;
  try {
    (void)probe_media("/definitely/missing/localbooru-video.mp4");
  } catch (const std::runtime_error& error) {
    failed = std::string(error.what()).find("failed to open media") != std::string::npos;
  }
  assert(failed);

  const auto fixture = std::filesystem::temp_directory_path() /
                       "localbooru-native-video-decoder-test.mp4";
  const std::string command =
      "ffmpeg -hide_banner -loglevel error -y -f lavfi -i "
      "color=c=red:s=64x48:r=10:d=1.0 -c:v mpeg4 \"" +
      fixture.string() + "\"";
  assert(std::system(command.c_str()) == 0);

  const auto frame = decode_first_video_frame(fixture.string());
  assert(frame.width == 64);
  assert(frame.height == 48);
  assert(frame.rgba.size() == 64U * 48U * 4U);
  assert(frame.pts_seconds >= 0.0);

  VideoFrameDecoder decoder(fixture.string());
  double previous_pts = -1.0;
  int decoded_frames = 0;
  while (auto next = decoder.next_frame()) {
    assert(next->width == 64);
    assert(next->height == 48);
    assert(next->yuv420p);
    assert(next->rgba.size() == 64U * 48U * 3U / 2U);
    assert(next->pts_seconds >= previous_pts);
    previous_pts = next->pts_seconds;
    ++decoded_frames;
  }
  assert(decoded_frames == 10);

  decoder.seek(0.5);
  const auto after_seek = decoder.next_frame();
  assert(after_seek.has_value());
  assert(after_seek->pts_seconds >= 0.5);

  std::filesystem::remove(fixture);
  std::cout << "native video decoder tests passed\n";
}
