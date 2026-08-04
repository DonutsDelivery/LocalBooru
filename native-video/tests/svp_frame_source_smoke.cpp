// SPDX-License-Identifier: MIT
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "decoder.h"
#include "svp_frame_source.h"

using namespace localbooru::native_video;

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: native-video-svp-smoke <video>\n";
    return 2;
  }
  try {
    const std::string path = argv[1];
    const auto metadata = probe_media(path);
    if (std::abs(metadata.frame_rate - (24000.0 / 1001.0)) > 0.01) {
      throw std::runtime_error("SVP smoke input is not 24000/1001 FPS");
    }
    SvpOptions options;
    options.target_fps = 60;
    SvpFrameSource source(path, metadata, options);
    source.start(0.0);
    double previous_pts = -1.0;
    std::uint64_t previous_hash = 0;
    constexpr int kFrames = 120;
    for (int index = 0; index < kFrames; ++index) {
      auto frame = source.next_frame();
      if (!frame)
        throw std::runtime_error("SVP ended before 120 frames");
      if (frame->pts_seconds <= previous_pts) {
        throw std::runtime_error("SVP output timestamps are not increasing");
      }
      const auto pixels =
          static_cast<std::size_t>(frame->width) * frame->height;
      const auto expected = frame->yuv420p ? pixels * 3 / 2 : pixels * 4;
      if (frame->rgba.size() != expected) {
        throw std::runtime_error("SVP output payload size is invalid");
      }
      std::uint64_t hash = 1469598103934665603ULL;
      for (std::size_t offset = 0; offset < frame->rgba.size(); offset += 257) {
        hash ^= frame->rgba[offset];
        hash *= 1099511628211ULL;
      }
      if (index > 0 && hash == previous_hash) {
        throw std::runtime_error("SVP emitted adjacent duplicate frames");
      }
      const double expected_pts = static_cast<double>(index) / 60.0;
      if (std::abs(frame->pts_seconds - expected_pts) > 1e-6) {
        throw std::runtime_error("SVP output timestamp does not match 60 FPS");
      }
      previous_hash = hash;
      previous_pts = frame->pts_seconds;
    }
    source.seek(1.0);
    auto seek_frame = source.next_frame();
    if (!seek_frame || seek_frame->pts_seconds < 1.0) {
      throw std::runtime_error("SVP seek epoch did not restart at the target");
    }
    const auto diagnostics = source.diagnostics();
    source.stop();
    std::cout << "frames_read=" << diagnostics.frames_read
              << " restarts=" << diagnostics.restarts << " adjacent_equal=0"
              << " seek_pts=" << seek_frame->pts_seconds << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
