// SPDX-License-Identifier: MIT
#include "hud.h"

#include <algorithm>

namespace localbooru::native_video {
namespace {
bool inside(double x, double y, double left, double top, double right,
            double bottom) {
  return x >= left && x <= right && y >= top && y <= bottom;
}
}

void HudController::set_viewport(int width, int height, double scale_factor) {
  width_ = std::max(width, 1);
  height_ = std::max(height, 1);
  scale_factor_ = std::max(scale_factor, 0.25);
}

void HudController::pointer_move(double, double, Clock::time_point now) {
  visible_ = true;
  last_activity_ = now;
}

std::optional<HudHit> HudController::pointer_down(double x, double y,
                                                  Clock::time_point now) {
  pointer_move(x, y, now);
  const double unit = 44.0 * scale_factor_;
  const double margin = 16.0 * scale_factor_;
  const double bottom = static_cast<double>(height_) - margin;
  const double top = bottom - unit;

  if (inside(x, y, margin, top, margin + unit, bottom))
    return HudHit{HudAction::Previous};
  if (inside(x, y, margin + unit, top, margin + 2 * unit, bottom))
    return HudHit{HudAction::TogglePlay};
  if (inside(x, y, margin + 2 * unit, top, margin + 3 * unit, bottom))
    return HudHit{HudAction::Next};
  if (inside(x, y, width_ - margin - 2 * unit, top,
             width_ - margin - unit, bottom))
    return HudHit{HudAction::ToggleFullscreen};
  if (inside(x, y, width_ - margin - unit, top, width_ - margin, bottom))
    return HudHit{HudAction::Close};

  const double timeline_left = margin + 3.25 * unit;
  const double timeline_right = width_ - margin - 4.25 * unit;
  const double timeline_top = top + unit * 0.25;
  const double timeline_bottom = top + unit * 0.5;
  if (timeline_right > timeline_left &&
      inside(x, y, timeline_left, timeline_top, timeline_right,
             timeline_bottom)) {
    return HudHit{HudAction::Seek,
                  std::clamp((x - timeline_left) /
                                 (timeline_right - timeline_left),
                             0.0, 1.0)};
  }

  const double volume_left = timeline_right + unit * 0.25;
  const double volume_right = width_ - margin - 2.25 * unit;
  if (volume_right > volume_left &&
      inside(x, y, volume_left, timeline_top, volume_right,
             timeline_bottom)) {
    return HudHit{HudAction::SetVolume,
                  std::clamp((x - volume_left) / (volume_right - volume_left),
                             0.0, 1.0)};
  }
  return std::nullopt;
}

void HudController::tick(Clock::time_point now) {
  if (last_activity_ != Clock::time_point{} &&
      now - last_activity_ >= std::chrono::seconds(3)) {
    visible_ = false;
  }
}

}  // namespace localbooru::native_video
