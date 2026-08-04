// SPDX-License-Identifier: MIT
#include "surface_channel.h"

#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cassert>
#include <stdexcept>
#include <vector>

using namespace localbooru::native_video;

int main() {
  int sockets[2] = {-1, -1};
  assert(socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, sockets) == 0);
  SurfaceChannel sender(sockets[0]);
  SurfaceChannel receiver(sockets[1]);

  const int file = open("/dev/null", O_RDONLY | O_CLOEXEC);
  assert(file >= 0);
  const nlohmann::json message = {
      {"type", "surface_created"},
      {"descriptor",
       {{"generation", 5},
        {"buffer_id", 1},
        {"width", 640},
        {"height", 360}}}};
  sender.send(message, {file});
  close(file);

  auto packet = receiver.receive();
  assert(packet.message == message);
  assert(packet.fds.size() == 1);
  assert(fcntl(packet.fds.front(), F_GETFD) >= 0);
  assert((fcntl(packet.fds.front(), F_GETFD) & FD_CLOEXEC) != 0);

  const int duplicate = open("/dev/null", O_RDONLY | O_CLOEXEC);
  assert(duplicate >= 0);
  bool rejected = false;
  try {
    sender.send(message, std::vector<int>(6, duplicate));
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  close(duplicate);
  assert(rejected);
  return 0;
}
