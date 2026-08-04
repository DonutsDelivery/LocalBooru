// SPDX-License-Identifier: MIT
#pragma once

#include <nlohmann/json.hpp>

#include <optional>
#include <vector>

namespace localbooru::native_video {

class ReceivedSurfacePacket {
 public:
  ReceivedSurfacePacket(nlohmann::json message, std::vector<int> fds);
  ~ReceivedSurfacePacket();
  ReceivedSurfacePacket(const ReceivedSurfacePacket&) = delete;
  ReceivedSurfacePacket& operator=(const ReceivedSurfacePacket&) = delete;
  ReceivedSurfacePacket(ReceivedSurfacePacket&& other) noexcept;
  ReceivedSurfacePacket& operator=(ReceivedSurfacePacket&& other) noexcept;

  nlohmann::json message;
  std::vector<int> fds;
};

class SurfaceChannel {
 public:
  explicit SurfaceChannel(int fd, bool owns_fd = true);
  ~SurfaceChannel();
  SurfaceChannel(const SurfaceChannel&) = delete;
  SurfaceChannel& operator=(const SurfaceChannel&) = delete;
  SurfaceChannel(SurfaceChannel&& other) noexcept;
  SurfaceChannel& operator=(SurfaceChannel&& other) noexcept;

  static std::optional<SurfaceChannel> from_environment();

  void send(const nlohmann::json& message,
            const std::vector<int>& fds = {}) const;
  [[nodiscard]] ReceivedSurfacePacket receive() const;
  [[nodiscard]] std::optional<ReceivedSurfacePacket> try_receive() const;
  [[nodiscard]] int fd() const { return fd_; }

 private:
  int fd_ = -1;
  bool owns_fd_ = true;

  [[nodiscard]] std::optional<ReceivedSurfacePacket> receive_with_flags(
      int flags) const;
};

}  // namespace localbooru::native_video
