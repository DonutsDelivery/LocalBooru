// SPDX-License-Identifier: MIT
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

#include "playback_session.h"

using namespace localbooru::native_video;
using namespace std::chrono_literals;

int main() {
  const auto fixture = std::filesystem::temp_directory_path() /
                       "localbooru-native-video-session-test.mp4";
  const std::string command =
      "ffmpeg -hide_banner -loglevel error -y -f lavfi -i "
      "color=c=green:s=64x48:r=10:d=1.0 -c:v mpeg4 \"" +
      fixture.string() + "\"";
  assert(std::system(command.c_str()) == 0);

  std::mutex mutex;
  std::condition_variable condition;
  std::vector<double> positions;
  bool ended = false;
  std::string error;

  PlaybackCallbacks callbacks;
  callbacks.on_frame = [&](const DecodedVideoFrame& frame) {
    std::lock_guard lock(mutex);
    positions.push_back(frame.pts_seconds);
    condition.notify_all();
  };
  callbacks.on_ended = [&] {
    std::lock_guard lock(mutex);
    ended = true;
    condition.notify_all();
  };
  callbacks.on_error = [&](const std::string& message) {
    std::lock_guard lock(mutex);
    error = message;
    condition.notify_all();
  };

  VideoPlaybackSession session(std::move(callbacks));
  session.set_speed(1.5);
  assert(session.speed() == 1.5);
  session.set_speed(1.0);
  session.open(fixture.string(), 0.0, true);

  {
    std::unique_lock lock(mutex);
    assert(condition.wait_for(lock, 500ms, [&] { return !positions.empty() || !error.empty(); }));
    assert(error.empty());
  }

  session.set_paused(true);
  std::size_t paused_count = 0;
  {
    std::lock_guard lock(mutex);
    paused_count = positions.size();
  }
  std::this_thread::sleep_for(250ms);
  {
    std::lock_guard lock(mutex);
    assert(positions.size() <= paused_count + 1);
  }

  session.seek(0.5);
  session.set_paused(false);
  {
    std::unique_lock lock(mutex);
    assert(condition.wait_for(lock, 2s, [&] { return ended || !error.empty(); }));
    assert(error.empty());
    assert(ended);
    assert(!positions.empty());
    assert(positions.back() >= 0.9);
  }

  std::size_t ended_count = 0;
  {
    std::lock_guard lock(mutex);
    ended = false;
    ended_count = positions.size();
  }
  session.seek(0.0);
  session.set_paused(false);
  {
    std::unique_lock lock(mutex);
    assert(condition.wait_for(lock, 1s, [&] {
      return positions.size() > ended_count || !error.empty();
    }));
    assert(error.empty());
    assert(positions.size() > ended_count);
    assert(positions.back() < 0.2);
  }

  session.stop();

  {
    std::lock_guard lock(mutex);
    positions.clear();
    ended = false;
    error.clear();
  }
  session.open(fixture.string(), 0.0, false);
  {
    std::unique_lock lock(mutex);
    assert(condition.wait_for(lock, 500ms,
                              [&] { return !positions.empty() || !error.empty(); }));
    assert(error.empty());
    assert(positions.size() == 1);
  }
  std::this_thread::sleep_for(250ms);
  {
    std::lock_guard lock(mutex);
    assert(positions.size() == 1);
  }
  session.stop();

  std::filesystem::remove(fixture);
  std::cout << "native video playback session tests passed\n";
}
