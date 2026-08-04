// SPDX-License-Identifier: MIT
#pragma once

#include <chrono>
#include <optional>

namespace localbooru::native_video {

enum class HudAction {
  Previous,
  TogglePlay,
  Next,
  Seek,
  SetVolume,
  ToggleFullscreen,
  Close,
};

struct HudHit {
  HudAction action;
  double normalized_value = 0.0;
};

class HudController {
 public:
  using Clock = std::chrono::steady_clock;

  void set_viewport(int width, int height, double scale_factor);
  void pointer_move(double x, double y, Clock::time_point now);
  std::optional<HudHit> pointer_down(double x, double y,
                                     Clock::time_point now);
  void tick(Clock::time_point now);
  [[nodiscard]] bool visible() const { return visible_; }

 private:
  int width_ = 1;
  int height_ = 1;
  double scale_factor_ = 1.0;
  bool visible_ = true;
  Clock::time_point last_activity_ = Clock::time_point{};
};

}  // namespace localbooru::native_video
