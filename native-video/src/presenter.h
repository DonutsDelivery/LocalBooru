// SPDX-License-Identifier: MIT
#pragma once

#include <SDL2/SDL.h>

#include <vector>

#include "decoder.h"
#include "hud.h"

namespace localbooru::native_video {

class SdlVideoPresenter {
public:
  SdlVideoPresenter() = default;
  ~SdlVideoPresenter();

  SdlVideoPresenter(const SdlVideoPresenter &) = delete;
  SdlVideoPresenter &operator=(const SdlVideoPresenter &) = delete;

  void show(const DecodedVideoFrame &frame, bool visible = true);
  void set_playback_state(double position, double duration, bool paused);
  void set_fullscreen(bool fullscreen);
  std::vector<HudHit> poll_actions();
  void close();

  [[nodiscard]] bool is_open() const { return window_ != nullptr; }
  [[nodiscard]] int width() const { return width_; }
  [[nodiscard]] int height() const { return height_; }
  [[nodiscard]] bool hud_visible() const { return hud_.visible(); }
  [[nodiscard]] bool is_fullscreen() const { return fullscreen_; }

private:
  void ensure_window(int width, int height, bool yuv420p, bool visible);
  void render();
  void draw_hud(int output_width, int output_height);
  void destroy_video_objects();

  SDL_Window *window_ = nullptr;
  SDL_Renderer *renderer_ = nullptr;
  SDL_Texture *texture_ = nullptr;
  int width_ = 0;
  int height_ = 0;
  bool yuv420p_ = false;
  double position_ = 0.0;
  double duration_ = 0.0;
  bool paused_ = true;
  bool fullscreen_ = false;
  bool owns_video_subsystem_ = false;
  HudController hud_;
};

} // namespace localbooru::native_video
