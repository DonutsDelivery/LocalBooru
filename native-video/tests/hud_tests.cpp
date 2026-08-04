// SPDX-License-Identifier: MIT
#include <cassert>
#include <chrono>
#include <iostream>

#include "hud.h"

using namespace localbooru::native_video;

int main() {
  HudController hud;
  hud.set_viewport(1280, 720, 1.0);
  const auto start = HudController::Clock::now();
  hud.pointer_move(20, 20, start);
  assert(hud.visible());
  hud.tick(start + std::chrono::milliseconds(2999));
  assert(hud.visible());
  hud.tick(start + std::chrono::seconds(3));
  assert(!hud.visible());

  const auto previous = hud.pointer_down(20, 680, start + std::chrono::seconds(4));
  assert(previous && previous->action == HudAction::Previous);
  assert(hud.visible());

  const auto play = hud.pointer_down(70, 680, start + std::chrono::seconds(4));
  assert(play && play->action == HudAction::TogglePlay);
  const auto next = hud.pointer_down(110, 680, start + std::chrono::seconds(4));
  assert(next && next->action == HudAction::Next);

  std::cout << "native video HUD tests passed\n";
}
