// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "decoder.h"

namespace localbooru::native_video {

struct SvpOptions {
  std::string preset = "balanced";
  std::uint32_t target_fps = 60;
  std::string plugin_path;
};

struct SvpDiagnostics {
  std::uint64_t frames_read = 0;
  std::uint64_t restarts = 0;
  std::string last_error;
};

class SvpFrameSource {
public:
  SvpFrameSource(std::string media_path, MediaMetadata metadata,
                 SvpOptions options = {});
  ~SvpFrameSource();

  SvpFrameSource(const SvpFrameSource &) = delete;
  SvpFrameSource &operator=(const SvpFrameSource &) = delete;

  void start(double position_seconds);
  std::optional<DecodedVideoFrame> next_frame();
  void seek(double position_seconds);
  void interrupt();
  void stop();

  [[nodiscard]] SvpDiagnostics diagnostics() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace localbooru::native_video
