// SPDX-License-Identifier: MIT
#include "surface_channel.h"

#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace localbooru::native_video {
namespace {
constexpr std::size_t kMaxMessageBytes = 64U * 1024U;
constexpr std::size_t kMaxMessageFds = 5;

std::runtime_error system_error(const char* action) {
  return std::runtime_error(std::string(action) + ": " + std::strerror(errno));
}

void close_all(std::vector<int>& fds) {
  for (const int fd : fds) {
    if (fd >= 0) ::close(fd);
  }
  fds.clear();
}
}  // namespace

ReceivedSurfacePacket::ReceivedSurfacePacket(nlohmann::json message,
                                             std::vector<int> fds)
    : message(std::move(message)), fds(std::move(fds)) {}

ReceivedSurfacePacket::~ReceivedSurfacePacket() { close_all(fds); }

ReceivedSurfacePacket::ReceivedSurfacePacket(ReceivedSurfacePacket&& other) noexcept
    : message(std::move(other.message)), fds(std::move(other.fds)) {
  other.fds.clear();
}

ReceivedSurfacePacket& ReceivedSurfacePacket::operator=(
    ReceivedSurfacePacket&& other) noexcept {
  if (this == &other) return *this;
  close_all(fds);
  message = std::move(other.message);
  fds = std::move(other.fds);
  other.fds.clear();
  return *this;
}

SurfaceChannel::SurfaceChannel(int fd, bool owns_fd)
    : fd_(fd), owns_fd_(owns_fd) {
  if (fd < 0) throw std::invalid_argument("surface channel fd is invalid");
}

SurfaceChannel::~SurfaceChannel() {
  if (owns_fd_ && fd_ >= 0) ::close(fd_);
}

SurfaceChannel::SurfaceChannel(SurfaceChannel&& other) noexcept
    : fd_(std::exchange(other.fd_, -1)), owns_fd_(other.owns_fd_) {}

SurfaceChannel& SurfaceChannel::operator=(SurfaceChannel&& other) noexcept {
  if (this == &other) return *this;
  if (owns_fd_ && fd_ >= 0) ::close(fd_);
  fd_ = std::exchange(other.fd_, -1);
  owns_fd_ = other.owns_fd_;
  return *this;
}

std::optional<SurfaceChannel> SurfaceChannel::from_environment() {
  const char* value = std::getenv("LOCALBOORU_SURFACE_FD");
  if (value == nullptr || *value == '\0') return std::nullopt;
  char* end = nullptr;
  errno = 0;
  const long parsed = std::strtol(value, &end, 10);
  if (errno != 0 || end == value || *end != '\0' || parsed < 0 ||
      parsed > std::numeric_limits<int>::max()) {
    throw std::runtime_error("LOCALBOORU_SURFACE_FD is invalid");
  }
  return SurfaceChannel(static_cast<int>(parsed));
}

void SurfaceChannel::send(const nlohmann::json& message,
                          const std::vector<int>& fds) const {
  if (fds.size() > kMaxMessageFds) {
    throw std::invalid_argument("surface message has too many file descriptors");
  }
  const std::string payload = message.dump();
  if (payload.size() > kMaxMessageBytes) {
    throw std::invalid_argument("surface message exceeds bounded payload size");
  }
  iovec iov{const_cast<char*>(payload.data()), payload.size()};
  std::array<std::byte, CMSG_SPACE(kMaxMessageFds * sizeof(int))> control{};
  msghdr header{};
  header.msg_iov = &iov;
  header.msg_iovlen = 1;
  if (!fds.empty()) {
    header.msg_control = control.data();
    header.msg_controllen = CMSG_SPACE(fds.size() * sizeof(int));
    cmsghdr* cmsg = CMSG_FIRSTHDR(&header);
    cmsg->cmsg_level = SOL_SOCKET;
    cmsg->cmsg_type = SCM_RIGHTS;
    cmsg->cmsg_len = CMSG_LEN(fds.size() * sizeof(int));
    std::memcpy(CMSG_DATA(cmsg), fds.data(), fds.size() * sizeof(int));
  }
  const ssize_t sent = ::sendmsg(fd_, &header, MSG_NOSIGNAL);
  if (sent < 0) throw system_error("failed to send surface packet");
  if (static_cast<std::size_t>(sent) != payload.size()) {
    throw std::runtime_error("surface channel sent a partial packet");
  }
}

ReceivedSurfacePacket SurfaceChannel::receive() const {
  auto packet = receive_with_flags(0);
  if (!packet) {
    throw std::logic_error("blocking surface receive returned no packet");
  }
  return std::move(*packet);
}

std::optional<ReceivedSurfacePacket> SurfaceChannel::try_receive() const {
  return receive_with_flags(MSG_DONTWAIT);
}

std::optional<ReceivedSurfacePacket> SurfaceChannel::receive_with_flags(
    int flags) const {
  std::array<char, kMaxMessageBytes> payload{};
  std::array<std::byte, CMSG_SPACE(kMaxMessageFds * sizeof(int))> control{};
  iovec iov{payload.data(), payload.size()};
  msghdr header{};
  header.msg_iov = &iov;
  header.msg_iovlen = 1;
  header.msg_control = control.data();
  header.msg_controllen = control.size();
  const ssize_t received = ::recvmsg(fd_, &header, MSG_CMSG_CLOEXEC | flags);
  if (received < 0 && flags != 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
    return std::nullopt;
  }
  if (received < 0) throw system_error("failed to receive surface packet");
  if (received == 0) throw std::runtime_error("surface channel closed");
  if ((header.msg_flags & (MSG_TRUNC | MSG_CTRUNC)) != 0) {
    throw std::runtime_error("surface channel packet or descriptors truncated");
  }

  std::vector<int> fds;
  for (cmsghdr* cmsg = CMSG_FIRSTHDR(&header); cmsg != nullptr;
       cmsg = CMSG_NXTHDR(&header, cmsg)) {
    if (cmsg->cmsg_level != SOL_SOCKET || cmsg->cmsg_type != SCM_RIGHTS) continue;
    const std::size_t bytes = cmsg->cmsg_len - CMSG_LEN(0);
    const std::size_t count = bytes / sizeof(int);
    const int* received_fds = reinterpret_cast<const int*>(CMSG_DATA(cmsg));
    fds.insert(fds.end(), received_fds, received_fds + count);
  }
  if (fds.size() > kMaxMessageFds) {
    close_all(fds);
    throw std::runtime_error("surface channel received too many descriptors");
  }
  try {
    return ReceivedSurfacePacket(
        nlohmann::json::parse(payload.data(), payload.data() + received),
        std::move(fds));
  } catch (...) {
    close_all(fds);
    throw;
  }
}

}  // namespace localbooru::native_video
