// SPDX-License-Identifier: MIT
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

#include "dmabuf_probe.h"

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3) {
    std::cerr << "usage: native-video-dmabuf-probe MEDIA [RENDER_NODE]\n";
    return 2;
  }
  const std::string device = argc == 3 ? argv[2] : "/dev/dri/renderD128";
  const auto probe = localbooru::native_video::probe_dmabuf_export(argv[1], device);
  nlohmann::json output{
      {"available", probe.available},
      {"reason", probe.reason},
      {"device", probe.device},
      {"width", probe.width},
      {"height", probe.height},
      {"objects", nlohmann::json::array()},
      {"layers", nlohmann::json::array()},
  };
  for (const auto& object : probe.objects) {
    output["objects"].push_back(
        {{"size", object.size}, {"modifier", object.modifier}});
  }
  for (const auto& layer : probe.layers) {
    nlohmann::json encoded{
        {"format", layer.format}, {"planes", nlohmann::json::array()}};
    for (const auto& plane : layer.planes) {
      encoded["planes"].push_back({{"object_index", plane.object_index},
                                   {"offset", plane.offset},
                                   {"pitch", plane.pitch}});
    }
    output["layers"].push_back(std::move(encoded));
  }
  std::cout << output.dump() << '\n';
  return probe.available ? 0 : 1;
}
