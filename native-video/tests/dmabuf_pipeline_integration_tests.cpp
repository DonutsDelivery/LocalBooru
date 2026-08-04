// SPDX-License-Identifier: MIT
#include "dmabuf_frame_decoder.h"
#include "dmabuf_surface_producer.h"
#include "surface_channel.h"

#include <fcntl.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cassert>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

#include <nlohmann/json.hpp>

using namespace localbooru::native_video;

namespace {

int consume_frames(int fd, std::uint64_t generation, std::uint64_t frame_count) {
  SurfaceChannel channel(fd);
  std::uint64_t previous_sequence = 0;
  for (std::uint64_t index = 0; index < frame_count; ++index) {
    auto descriptor = channel.receive();
    if (descriptor.message.value("type", "") != "surface_created") return 10;
    const auto& metadata = descriptor.message.at("descriptor");
    if (metadata.at("generation").get<std::uint64_t>() != generation ||
        metadata.value("handle_kind", "") != "dma_buf" ||
        descriptor.fds.empty()) {
      return 11;
    }
    for (const int object_fd : descriptor.fds) {
      if (fcntl(object_fd, F_GETFD) < 0) return 12;
    }

    auto ready = channel.receive();
    if (ready.message.value("type", "") != "frame_ready" ||
        !ready.fds.empty()) {
      return 13;
    }
    const auto& frame = ready.message.at("frame");
    const auto sequence = frame.at("sequence").get<std::uint64_t>();
    if (frame.at("generation").get<std::uint64_t>() != generation ||
        sequence <= previous_sequence) {
      return 14;
    }
    previous_sequence = sequence;
    channel.send(
        {{"type", "frame_release"},
         {"release",
          {{"generation", generation},
           {"buffer_id", frame.at("buffer_id")},
           {"sequence", sequence}}}});
  }
  return 0;
}

std::size_t open_fd_count() {
  std::size_t count = 0;
  for (const auto& ignored : std::filesystem::directory_iterator("/proc/self/fd")) {
    static_cast<void>(ignored);
    ++count;
  }
  return count;
}

}  // namespace

int main() {
  const char* media = std::getenv("LOCALBOORU_DMABUF_TEST_MEDIA");
  const char* device = std::getenv("LOCALBOORU_DMABUF_TEST_DEVICE");
  const char* frame_count_text = std::getenv("LOCALBOORU_DMABUF_TEST_FRAMES");
  if (!media || !std::filesystem::is_regular_file(media)) return 0;
  const std::string render_node = device ? device : "/dev/dri/renderD128";
  if (!std::filesystem::exists(render_node)) return 0;
  const std::uint64_t frame_count =
      frame_count_text ? std::stoull(frame_count_text) : 120;
  if (frame_count < 3) return 2;

  int sockets[2] = {-1, -1};
  assert(socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, sockets) == 0);
  constexpr std::uint64_t kGeneration = 73;
  const pid_t child = fork();
  assert(child >= 0);
  if (child == 0) {
    close(sockets[0]);
    const int result = consume_frames(sockets[1], kGeneration, frame_count);
    _exit(result);
  }

  close(sockets[1]);
  SurfaceChannel producer_channel(sockets[0]);
  DmabufSurfaceProducer producer(producer_channel, render_node, 3);
  DmabufFrameDecoder decoder(media, render_node);
  producer.configure(kGeneration);

  std::uint64_t published = 0;
  std::size_t steady_fd_count = 0;
  const auto started = std::chrono::steady_clock::now();
  while (published < frame_count) {
    while (producer.available() == 0) {
      assert(producer.receive_release());
    }
    auto frame = decoder.next_frame();
    assert(frame);
    const auto lease = producer.publish(std::move(*frame));
    assert(lease);
    ++published;
    if (published == 3) steady_fd_count = open_fd_count();
  }
  while (producer.available() < 3) assert(producer.receive_release());

  int status = 0;
  while (waitpid(child, &status, 0) < 0 && errno == EINTR) {
  }
  assert(WIFEXITED(status));
  assert(WEXITSTATUS(status) == 0);
  assert(producer.available() == 3);
  const std::size_t final_fd_count = open_fd_count();
  assert(final_fd_count <= steady_fd_count);
  const double elapsed_seconds = std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - started)
                                     .count();
  std::cout << nlohmann::json({{"cross_process", true},
                               {"generation", kGeneration},
                               {"published_frames", published},
                               {"released_frames", published},
                               {"pool_size", 3},
                               {"available_after_release", producer.available()},
                               {"steady_fd_count", steady_fd_count},
                               {"final_fd_count", final_fd_count},
                               {"elapsed_seconds", elapsed_seconds},
                               {"pipeline_fps", published / elapsed_seconds}})
                   .dump()
            << '\n';
  return 0;
}
