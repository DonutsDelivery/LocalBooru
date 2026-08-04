// SPDX-License-Identifier: MIT
#include <cassert>
#include <iostream>

#include "protocol.h"

using namespace localbooru::native_video;

int main() {
  const auto fixture = nlohmann::json::parse(
      R"({"type":"open_media","generation":42,"item_id":7,"path":"/tmp/video.mp4","resume_position":3.5,"autoplay":true,"future":"ignored"})");
  assert(fixture.at("type") == "open_media");
  assert(generation_of(fixture) == 42);
  assert(!generation_of(nlohmann::json{{"type", "set_paused"}}));

  validate_protocol_version(kProtocolVersion);
  bool rejected = false;
  try {
    validate_protocol_version(kProtocolVersion + 1000);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  assert(rejected);

  const auto ready = ready_event();
  assert(ready.at("type") == "ready");
  assert(ready.at("protocol_version") == 1000);
  std::cout << "native video protocol tests passed\n";
}
