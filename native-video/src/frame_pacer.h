// SPDX-License-Identifier: MIT
#pragma once

#include <algorithm>
#include <cmath>

namespace localbooru::native_video {

class FramePacer {
 public:
  void reset() { started_ = false; }

  void anchor(double media_seconds, double wall_seconds) {
    media_anchor_ = finite_or_zero(media_seconds);
    wall_anchor_ = finite_or_zero(wall_seconds);
    started_ = true;
  }

  void set_speed(double speed, double media_seconds, double wall_seconds) {
    anchor(media_seconds, wall_seconds);
    speed_ = std::isfinite(speed) ? std::clamp(speed, 0.5, 2.0) : 1.0;
  }

  bool due(double media_seconds, double wall_seconds) {
    if (!started_) {
      anchor(media_seconds, wall_seconds);
      return true;
    }
    return finite_or_zero(wall_seconds) + 1e-6 >= target_wall(media_seconds);
  }

  double target_wall(double media_seconds) const {
    const double elapsed =
        std::max(0.0, finite_or_zero(media_seconds) - media_anchor_);
    return wall_anchor_ + elapsed / speed_;
  }

 private:
  static double finite_or_zero(double value) {
    return std::isfinite(value) ? value : 0.0;
  }

  bool started_ = false;
  double media_anchor_ = 0.0;
  double wall_anchor_ = 0.0;
  double speed_ = 1.0;
};

}  // namespace localbooru::native_video
