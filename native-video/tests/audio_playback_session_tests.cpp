// SPDX-License-Identifier: MIT
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <thread>

#include "audio_playback_session.h"

using namespace localbooru::native_video;
using namespace std::chrono_literals;

int main() {
  setenv("SDL_AUDIODRIVER", "dummy", 1);
  const auto fixture = std::filesystem::temp_directory_path() /
                       "localbooru-native-video-audio-session-test.m4a";
  const std::string command =
      "ffmpeg -hide_banner -loglevel error -y -f lavfi -i "
      "sine=frequency=440:sample_rate=48000:duration=1.5 -c:a aac \"" +
      fixture.string() + "\"";
  assert(std::system(command.c_str()) == 0);

  AudioPlaybackSession session;
  session.set_volume(0.35);
  assert(session.volume() == 0.35);
  session.set_speed(1.5);
  assert(session.speed() == 1.5);
  session.open(fixture.string(), 0.0, true);
  const auto deadline = std::chrono::steady_clock::now() + 2s;
  while (session.submitted_seconds() < 0.2 &&
         std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(10ms);
  }
  assert(session.submitted_seconds() >= 0.2);

  session.set_paused(true);
  session.seek(0.25);
  session.set_paused(false);
  std::this_thread::sleep_for(100ms);
  assert(session.last_error().empty());

  const auto end_deadline = std::chrono::steady_clock::now() + 3s;
  while (session.submitted_seconds() < 1.4 &&
         std::chrono::steady_clock::now() < end_deadline) {
    std::this_thread::sleep_for(10ms);
  }
  assert(session.submitted_seconds() >= 1.4);
  std::this_thread::sleep_for(800ms);

  session.set_paused(true);
  session.seek(0.25);
  const auto restart_deadline = std::chrono::steady_clock::now() + 500ms;
  while (session.submitted_seconds() >= 1.0 &&
         std::chrono::steady_clock::now() < restart_deadline) {
    std::this_thread::sleep_for(5ms);
  }
  assert(session.submitted_seconds() < 1.0);
  session.stop();

  std::filesystem::remove(fixture);
  std::cout << "native video audio playback session tests passed\n";
}
