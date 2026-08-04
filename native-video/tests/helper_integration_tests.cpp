// SPDX-License-Identifier: MIT
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef NATIVE_VIDEO_HELPER_PATH
#error NATIVE_VIDEO_HELPER_PATH must name the helper executable
#endif

int main() {
  const auto temp = std::filesystem::temp_directory_path();
  const auto fixture = temp / "localbooru-native-video-helper-test.mp4";
  const auto input = temp / "localbooru-native-video-helper-test.in";
  const auto output = temp / "localbooru-native-video-helper-test.out";

  const std::string make_fixture =
      "ffmpeg -hide_banner -loglevel error -y -f lavfi -i "
      "color=c=blue:s=64x48:d=0.1 -frames:v 1 -c:v mpeg4 \"" +
      fixture.string() + "\"";
  assert(std::system(make_fixture.c_str()) == 0);

  {
    std::ofstream stream(input);
    stream << R"({"type":"hello","protocol_version":1000})" << '\n';
    stream << R"({"type":"open_media","generation":7,"item_id":4,"path":")"
           << fixture.string()
           << R"(","resume_position":0.0,"autoplay":true})" << '\n';
    stream << R"({"type":"set_volume","volume":0.4})" << '\n';
    stream << R"({"type":"set_muted","muted":true})" << '\n';
    stream << R"({"type":"set_speed","speed":1.5})" << '\n';
    stream << R"({"type":"set_subtitle_delay","seconds":0.75})" << '\n';
  }

  const std::string run =
      "(cat \"" + input.string() + "\"; sleep 0.25) | "
      "SDL_VIDEODRIVER=dummy \"" + std::string(NATIVE_VIDEO_HELPER_PATH) +
      "\" > \"" + output.string() + "\"";
  assert(std::system(run.c_str()) == 0);

  std::ifstream stream(output);
  std::string line;
  std::vector<nlohmann::json> events;
  while (std::getline(stream, line)) events.push_back(nlohmann::json::parse(line));

  assert(events.size() >= 3);
  assert(events.front().at("type") == "ready");
  const auto opened = std::find_if(events.begin(), events.end(), [](const auto& event) {
    return event.value("type", "") == "media_opened";
  });
  const auto first_frame =
      std::find_if(events.begin(), events.end(), [](const auto& event) {
        return event.value("type", "") == "first_frame_ready";
      });
  assert(opened != events.end());
  assert(opened->at("generation") == 7);
  assert(first_frame != events.end());
  assert(first_frame->at("generation") == 7);
  const auto playback_state = std::find_if(
      events.rbegin(), events.rend(), [](const auto& event) {
        return event.value("type", "") == "playback_state" &&
               event.value("muted", false) &&
               event.value("speed", 1.0) == 1.5;
      });
  assert(playback_state != events.rend());
  assert(playback_state->at("volume") == 0.4);
  assert(playback_state->at("subtitle_delay") == 0.75);
  assert(playback_state->at("interpolation_engine") == "off");
  assert(playback_state->at("interpolation_target_fps") == 60);
  assert(playback_state->at("selected_audio_track").is_null());
  assert(playback_state->at("selected_subtitle_track").is_null());

  std::filesystem::remove(fixture);
  std::filesystem::remove(input);
  std::filesystem::remove(output);
  std::cout << "native video helper integration tests passed\n";
}
