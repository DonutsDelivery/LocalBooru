// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace localbooru::native_video {

inline constexpr std::uint32_t kProtocolVersion = 1000;
inline constexpr std::uint32_t kProtocolMajorDivisor = 1000;

inline void validate_protocol_version(std::uint32_t peer_version) {
  if (peer_version / kProtocolMajorDivisor !=
      kProtocolVersion / kProtocolMajorDivisor) {
    throw std::runtime_error("incompatible native-video protocol: local " +
                             std::to_string(kProtocolVersion) + ", peer " +
                             std::to_string(peer_version));
  }
}

inline std::optional<std::uint64_t> generation_of(const nlohmann::json& message) {
  if (!message.contains("generation")) return std::nullopt;
  return message.at("generation").get<std::uint64_t>();
}

inline nlohmann::json ready_event() {
  return {{"type", "ready"}, {"protocol_version", kProtocolVersion}};
}

inline nlohmann::json fatal_error(std::optional<std::uint64_t> generation,
                                  const std::string& message) {
  return {{"type", "fatal_error"},
          {"generation", generation ? nlohmann::json(*generation)
                                     : nlohmann::json(nullptr)},
          {"message", message}};
}

}  // namespace localbooru::native_video
