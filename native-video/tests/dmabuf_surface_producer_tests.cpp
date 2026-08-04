// SPDX-License-Identifier: MIT
#include "dmabuf_frame_decoder.h"
#include "dmabuf_surface_producer.h"
#include "surface_channel.h"

#include <fcntl.h>
#include <sys/socket.h>

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

using namespace localbooru::native_video;

int main() {
  const char* media = std::getenv("LOCALBOORU_DMABUF_TEST_MEDIA");
  const char* device = std::getenv("LOCALBOORU_DMABUF_TEST_DEVICE");
  if (!media || !std::filesystem::is_regular_file(media)) return 0;
  const std::string render_node = device ? device : "/dev/dri/renderD128";
  if (!std::filesystem::exists(render_node)) return 0;

  int sockets[2] = {-1, -1};
  assert(socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, sockets) == 0);
  SurfaceChannel producer_channel(sockets[0]);
  SurfaceChannel consumer_channel(sockets[1]);
  DmabufSurfaceProducer producer(producer_channel, render_node, 3);
  DmabufFrameDecoder decoder(media, render_node);
  producer.configure(41);

  std::vector<FrameLease> leases;
  std::optional<ReceivedSurfacePacket> first_descriptor;
  for (int index = 0; index < 3; ++index) {
    auto frame = decoder.next_frame();
    assert(frame);
    const auto lease = producer.publish(std::move(*frame));
    assert(lease);
    leases.push_back(*lease);

    auto descriptor = consumer_channel.receive();
    assert(descriptor.message.at("type") == "surface_created");
    const auto& metadata = descriptor.message.at("descriptor");
    assert(metadata.at("generation") == 41);
    assert(metadata.at("buffer_id") == index);
    assert(metadata.at("handle_kind") == "dma_buf");
    assert(metadata.at("producer_drm_node") == render_node);
    assert(metadata.at("dmabuf").at("objects").size() == descriptor.fds.size());
    assert(!descriptor.fds.empty());
    for (const int fd : descriptor.fds) assert(fcntl(fd, F_GETFD) >= 0);

    auto ready = consumer_channel.receive();
    assert(ready.fds.empty());
    assert(ready.message.at("type") == "frame_ready");
    assert(ready.message.at("frame").at("sequence") == lease->sequence);
    if (index == 0) first_descriptor.emplace(std::move(descriptor));
  }
  assert(producer.available() == 0);

  auto fourth = decoder.next_frame();
  assert(fourth);
  assert(!producer.publish(std::move(*fourth)));

  consumer_channel.send(
      {{"type", "frame_release"},
       {"release",
        {{"generation", leases[0].generation},
         {"buffer_id", leases[0].buffer_id},
         {"sequence", leases[0].sequence + 1}}}});
  assert(producer.drain_releases() == 0);
  assert(producer.available() == 0);

  consumer_channel.send(
      {{"type", "frame_release"},
       {"release",
        {{"generation", leases[0].generation},
         {"buffer_id", leases[0].buffer_id},
         {"sequence", leases[0].sequence}}}});
  assert(producer.drain_releases() == 1);
  assert(producer.available() == 1);
  // SCM_RIGHTS gives the consumer independent references even after producer release.
  for (const int fd : first_descriptor->fds) assert(fcntl(fd, F_GETFD) >= 0);

  auto replacement = decoder.next_frame();
  assert(replacement);
  const auto replacement_lease = producer.publish(std::move(*replacement));
  assert(replacement_lease);
  assert(replacement_lease->buffer_id == leases[0].buffer_id);
  assert(replacement_lease->sequence > leases[2].sequence);
  return 0;
}
