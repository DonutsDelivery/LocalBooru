// SPDX-License-Identifier: MIT
#include "presenter.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>

namespace localbooru::native_video {
namespace {
std::runtime_error sdl_error(const char *operation) {
  return std::runtime_error(std::string(operation) + ": " + SDL_GetError());
}
} // namespace

SdlVideoPresenter::~SdlVideoPresenter() { close(); }

void SdlVideoPresenter::ensure_window(int width, int height, bool yuv420p,
                                      bool visible) {
  if (width <= 0 || height <= 0) {
    throw std::runtime_error("video frame dimensions must be positive");
  }
  if ((SDL_WasInit(SDL_INIT_VIDEO) & SDL_INIT_VIDEO) == 0) {
    if (SDL_InitSubSystem(SDL_INIT_VIDEO) != 0) {
      throw sdl_error("failed to initialize SDL video");
    }
    owns_video_subsystem_ = true;
  }

  if (window_ != nullptr && width_ == width && height_ == height &&
      yuv420p_ == yuv420p) {
    if (visible)
      SDL_ShowWindow(window_);
    return;
  }
  destroy_video_objects();

  const Uint32 flags = visible ? SDL_WINDOW_SHOWN : SDL_WINDOW_HIDDEN;
  window_ = SDL_CreateWindow("LocalBooru Video", SDL_WINDOWPOS_CENTERED,
                             SDL_WINDOWPOS_CENTERED, width, height,
                             flags | SDL_WINDOW_RESIZABLE);
  if (window_ == nullptr)
    throw sdl_error("failed to create SDL video window");

  renderer_ = SDL_CreateRenderer(
      window_, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);
  if (renderer_ == nullptr) {
    renderer_ = SDL_CreateRenderer(window_, -1, SDL_RENDERER_SOFTWARE);
  }
  if (renderer_ == nullptr)
    throw sdl_error("failed to create SDL renderer");

  texture_ = SDL_CreateTexture(
      renderer_, yuv420p ? SDL_PIXELFORMAT_IYUV : SDL_PIXELFORMAT_RGBA32,
      SDL_TEXTUREACCESS_STREAMING, width, height);
  if (texture_ == nullptr)
    throw sdl_error("failed to create SDL video texture");
  width_ = width;
  height_ = height;
  yuv420p_ = yuv420p;
}

void SdlVideoPresenter::show(const DecodedVideoFrame &frame, bool visible) {
  const std::size_t pixels = static_cast<std::size_t>(frame.width) *
                             static_cast<std::size_t>(frame.height);
  const std::size_t required = frame.yuv420p ? pixels * 3U / 2U : pixels * 4U;
  if (frame.rgba.size() != required) {
    throw std::runtime_error("video frame size does not match dimensions");
  }
  ensure_window(frame.width, frame.height, frame.yuv420p, visible);
  int update_result = 0;
  if (frame.yuv420p) {
    const std::uint8_t *y = frame.rgba.data();
    const std::uint8_t *u = y + pixels;
    const std::uint8_t *v = u + pixels / 4U;
    update_result = SDL_UpdateYUVTexture(texture_, nullptr, y, frame.width, u,
                                         frame.width / 2, v, frame.width / 2);
  } else {
    update_result = SDL_UpdateTexture(texture_, nullptr, frame.rgba.data(),
                                      frame.width * 4);
  }
  if (update_result != 0) {
    throw sdl_error("failed to upload SDL video frame");
  }
  render();
}

void SdlVideoPresenter::set_playback_state(double position, double duration,
                                           bool paused) {
  position_ = std::isfinite(position) ? std::max(0.0, position) : 0.0;
  duration_ = std::isfinite(duration) ? std::max(0.0, duration) : 0.0;
  paused_ = paused;
  if (is_open())
    render();
}

void SdlVideoPresenter::set_fullscreen(bool fullscreen) {
  if (!is_open())
    return;
  const Uint32 flags = fullscreen ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0;
  if (SDL_SetWindowFullscreen(window_, flags) != 0) {
    throw sdl_error("failed to change SDL fullscreen state");
  }
  fullscreen_ = fullscreen;
}

void SdlVideoPresenter::render() {
  if (!renderer_ || !texture_)
    return;
  int output_width = width_;
  int output_height = height_;
  SDL_GetRendererOutputSize(renderer_, &output_width, &output_height);
  hud_.set_viewport(output_width, output_height, 1.0);
  SDL_SetRenderDrawColor(renderer_, 0, 0, 0, 255);
  if (SDL_RenderClear(renderer_) != 0 ||
      SDL_RenderCopy(renderer_, texture_, nullptr, nullptr) != 0) {
    throw sdl_error("failed to render SDL video frame");
  }
  if (hud_.visible())
    draw_hud(output_width, output_height);
  SDL_RenderPresent(renderer_);
}

void SdlVideoPresenter::draw_hud(int output_width, int output_height) {
  SDL_SetRenderDrawBlendMode(renderer_, SDL_BLENDMODE_BLEND);
  SDL_SetRenderDrawColor(renderer_, 8, 10, 14, 210);
  SDL_Rect panel{0, std::max(0, output_height - 76), output_width,
                 std::min(76, output_height)};
  SDL_RenderFillRect(renderer_, &panel);

  constexpr int margin = 16;
  constexpr int unit = 44;
  const int top = output_height - margin - unit;
  SDL_SetRenderDrawColor(renderer_, 225, 230, 238, 220);
  for (int index = 0; index < 3; ++index) {
    SDL_Rect button{margin + index * unit + 6, top + 6, unit - 12, unit - 12};
    SDL_RenderDrawRect(renderer_, &button);
  }
  SDL_Rect fullscreen{output_width - margin - 2 * unit + 6, top + 6, unit - 12,
                      unit - 12};
  SDL_Rect close{output_width - margin - unit + 6, top + 6, unit - 12,
                 unit - 12};
  SDL_RenderDrawRect(renderer_, &fullscreen);
  SDL_RenderDrawRect(renderer_, &close);

  const int timeline_left = margin + static_cast<int>(3.25 * unit);
  const int timeline_right =
      output_width - margin - static_cast<int>(4.25 * unit);
  if (timeline_right > timeline_left) {
    const int timeline_y = top + unit / 3;
    SDL_SetRenderDrawColor(renderer_, 110, 120, 134, 220);
    SDL_Rect track{timeline_left, timeline_y, timeline_right - timeline_left,
                   6};
    SDL_RenderFillRect(renderer_, &track);
    const double progress =
        duration_ > 0.0 ? std::clamp(position_ / duration_, 0.0, 1.0) : 0.0;
    SDL_SetRenderDrawColor(renderer_, paused_ ? 170 : 92, paused_ ? 180 : 190,
                           paused_ ? 190 : 255, 255);
    SDL_Rect fill{timeline_left, timeline_y,
                  static_cast<int>((timeline_right - timeline_left) * progress),
                  6};
    SDL_RenderFillRect(renderer_, &fill);
  }
  SDL_SetRenderDrawBlendMode(renderer_, SDL_BLENDMODE_NONE);
}

std::vector<HudHit> SdlVideoPresenter::poll_actions() {
  std::vector<HudHit> actions;
  if (!is_open())
    return actions;
  SDL_Event event{};
  const auto now = HudController::Clock::now();
  while (SDL_PollEvent(&event)) {
    switch (event.type) {
    case SDL_MOUSEMOTION:
      hud_.pointer_move(event.motion.x, event.motion.y, now);
      break;
    case SDL_MOUSEBUTTONDOWN:
      if (event.button.button == SDL_BUTTON_LEFT) {
        if (auto hit = hud_.pointer_down(event.button.x, event.button.y, now)) {
          actions.push_back(*hit);
        }
      }
      break;
    case SDL_KEYDOWN:
      if (event.key.keysym.sym == SDLK_SPACE)
        actions.push_back(HudHit{HudAction::TogglePlay});
      else if (event.key.keysym.sym == SDLK_f)
        actions.push_back(HudHit{HudAction::ToggleFullscreen});
      else if (event.key.keysym.sym == SDLK_ESCAPE)
        actions.push_back(HudHit{HudAction::Close});
      break;
    case SDL_QUIT:
      actions.push_back(HudHit{HudAction::Close});
      break;
    default:
      break;
    }
  }
  hud_.tick(now);
  render();
  return actions;
}

void SdlVideoPresenter::destroy_video_objects() {
  if (texture_ != nullptr)
    SDL_DestroyTexture(texture_);
  if (renderer_ != nullptr)
    SDL_DestroyRenderer(renderer_);
  if (window_ != nullptr)
    SDL_DestroyWindow(window_);
  texture_ = nullptr;
  renderer_ = nullptr;
  window_ = nullptr;
  width_ = 0;
  height_ = 0;
  yuv420p_ = false;
  fullscreen_ = false;
}

void SdlVideoPresenter::close() {
  destroy_video_objects();
  if (owns_video_subsystem_) {
    SDL_QuitSubSystem(SDL_INIT_VIDEO);
    owns_video_subsystem_ = false;
  }
}

} // namespace localbooru::native_video
