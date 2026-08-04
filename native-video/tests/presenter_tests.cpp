// SPDX-License-Identifier: MIT
#include <cassert>
#include <cstdlib>
#include <iostream>

#include "presenter.h"

using namespace localbooru::native_video;

int main() {
  setenv("SDL_VIDEODRIVER", "dummy", 1);

  DecodedVideoFrame frame;
  frame.width = 640;
  frame.height = 360;
  frame.rgba.assign(640U * 360U * 4U, 0xff);

  SdlVideoPresenter presenter;
  presenter.show(frame, false);
  assert(presenter.width() == 640);
  assert(presenter.height() == 360);
  assert(presenter.is_open());

  SDL_Event move{};
  move.type = SDL_MOUSEMOTION;
  move.motion.x = 20;
  move.motion.y = 320;
  assert(SDL_PushEvent(&move) == 1);
  SDL_Event click{};
  click.type = SDL_MOUSEBUTTONDOWN;
  click.button.button = SDL_BUTTON_LEFT;
  click.button.x = 20;
  click.button.y = 320;
  assert(SDL_PushEvent(&click) == 1);
  const auto actions = presenter.poll_actions();
  assert(actions.size() == 1);
  assert(actions.front().action == HudAction::Previous);
  assert(presenter.hud_visible());
  presenter.set_fullscreen(true);
  assert(presenter.is_fullscreen());
  presenter.set_fullscreen(false);
  assert(!presenter.is_fullscreen());

  DecodedVideoFrame yuv_frame;
  yuv_frame.width = 640;
  yuv_frame.height = 360;
  yuv_frame.yuv420p = true;
  yuv_frame.rgba.assign(640U * 360U * 3U / 2U, 0x80);
  presenter.show(yuv_frame, false);
  assert(presenter.is_open());

  presenter.close();
  assert(!presenter.is_open());

  std::cout << "native video presenter tests passed\n";
}
