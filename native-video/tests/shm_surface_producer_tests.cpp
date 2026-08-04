// SPDX-License-Identifier: MIT
#include "shm_surface_producer.h"

#include <sys/mman.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>

using namespace localbooru::native_video;

int main() {
  int sockets[2] = {-1, -1};
  assert(socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, sockets) == 0);
  SurfaceChannel producer_channel(sockets[0]);
  SurfaceChannel consumer_channel(sockets[1]);
  ShmSurfaceProducer producer(producer_channel);
  producer.configure(11, 4, 2, false, 4.0 / 3.0, 90, "bt601", "full",
                     "left");

  std::vector<ReceivedSurfacePacket> descriptors;
  for (int index = 0; index < 3; ++index) {
    descriptors.push_back(consumer_channel.receive());
    assert(descriptors.back().message.at("type") == "surface_created");
    assert(descriptors.back().fds.size() == 1);
    const auto &descriptor = descriptors.back().message.at("descriptor");
    assert(descriptor.at("sample_aspect_ratio") == 4.0 / 3.0);
    assert(descriptor.at("rotation_degrees") == 90);
    assert(descriptor.at("color_space") == "bt601");
    assert(descriptor.at("color_range") == "full");
    assert(descriptor.at("chroma_location") == "left");
  }

  DecodedVideoFrame frame;
  frame.width = 4;
  frame.height = 2;
  frame.pts_seconds = 1.25;
  frame.rgba.resize(4U * 2U * 4U);
  for (std::size_t index = 0; index < frame.rgba.size(); ++index) {
    frame.rgba[index] = static_cast<std::uint8_t>(index);
  }
  const auto lease = producer.publish(frame);
  assert(lease && lease->buffer_id == 0);
  auto ready = consumer_channel.receive();
  assert(ready.message.at("type") == "frame_ready");
  assert(ready.fds.empty());

  const int frame_fd = descriptors.at(lease->buffer_id).fds.front();
  void* mapping = mmap(nullptr, frame.rgba.size(), PROT_READ, MAP_SHARED, frame_fd, 0);
  assert(mapping != MAP_FAILED);
  const auto* pixels = static_cast<const std::uint8_t*>(mapping);
  assert(std::equal(frame.rgba.begin(), frame.rgba.end(), pixels));
  munmap(mapping, frame.rgba.size());

  consumer_channel.send({{"type", "frame_release"},
                         {"release",
                          {{"generation", lease->generation},
                           {"buffer_id", lease->buffer_id},
                           {"sequence", lease->sequence}}}});
  assert(producer.receive_release());
  assert(producer.available() == 3);
  return 0;
}
