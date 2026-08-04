// SPDX-License-Identifier: MIT
#include <cassert>
#include <cmath>
#include <iostream>

#include "frame_pacer.h"

using localbooru::native_video::FramePacer;

int main() {
  FramePacer pacer;

  assert(pacer.due(10.0, 100.0));
  assert(!pacer.due(10.04, 100.02));
  assert(pacer.due(10.04, 100.04));

  pacer.reset();
  assert(pacer.due(133.0, 200.0));
  assert(!pacer.due(133.04, 200.01));

  pacer.set_speed(2.0, 133.0, 300.0);
  assert(std::abs(pacer.target_wall(134.0) - 300.5) < 1e-9);
  assert(!pacer.due(134.0, 300.49));
  assert(pacer.due(134.0, 300.5));

  pacer.set_speed(1.0, 134.0, 301.0);
  assert(std::abs(pacer.target_wall(135.0) - 302.0) < 1e-9);

  std::cout << "native video frame pacer tests passed\n";
}
