// SPDX-License-Identifier: MIT
#include "surface_channel.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#ifndef NATIVE_VIDEO_HELPER_PATH
#error NATIVE_VIDEO_HELPER_PATH must name the helper executable
#endif

using namespace localbooru::native_video;

int main() {
  const auto temp = std::filesystem::temp_directory_path();
  const auto fixture = temp / "localbooru-native-surface-helper-test.mp4";
  const auto output = temp / "localbooru-native-surface-helper-test.out";
  const std::string make_fixture =
      "ffmpeg -hide_banner -loglevel error -y -f lavfi -i "
      "color=c=blue:s=64x48:d=0.1 -frames:v 1 -c:v mpeg4 \"" +
      fixture.string() + "\"";
  assert(std::system(make_fixture.c_str()) == 0);

  int sockets[2] = {-1, -1};
  int commands[2] = {-1, -1};
  assert(socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, sockets) == 0);
  assert(pipe2(commands, O_CLOEXEC) == 0);
  const int output_fd =
      open(output.c_str(), O_CREAT | O_TRUNC | O_WRONLY | O_CLOEXEC, 0600);
  assert(output_fd >= 0);

  const pid_t child = fork();
  assert(child >= 0);
  if (child == 0) {
    assert(dup2(commands[0], STDIN_FILENO) >= 0);
    assert(dup2(output_fd, STDOUT_FILENO) >= 0);
    assert(dup2(sockets[1], 3) >= 0);
    assert(fcntl(3, F_SETFD, 0) == 0);
    setenv("LOCALBOORU_SURFACE_FD", "3", 1);
    setenv("SDL_VIDEODRIVER", "dummy", 1);
    execl(NATIVE_VIDEO_HELPER_PATH, NATIVE_VIDEO_HELPER_PATH, nullptr);
    _exit(127);
  }

  close(commands[0]);
  close(output_fd);
  close(sockets[1]);
  SurfaceChannel channel(sockets[0]);
  const std::string input =
      R"({"type":"hello","protocol_version":1000})"
      "\n" +
      std::string(
          R"({"type":"open_media","generation":21,"item_id":9,"path":")") +
      fixture.string() +
      R"(","resume_position":0.0,"autoplay":true})"
      "\n";
  assert(write(commands[1], input.data(), input.size()) ==
         static_cast<ssize_t>(input.size()));

  std::vector<ReceivedSurfacePacket> descriptors;
  for (int index = 0; index < 3; ++index) {
    descriptors.push_back(channel.receive());
    assert(descriptors.back().message.at("type") == "surface_created");
    assert(descriptors.back().message.at("descriptor").at("generation") == 21);
    assert(descriptors.back().message.at("descriptor").at("fourcc") ==
           0x32315559U);
    assert(descriptors.back().fds.size() == 1);
  }
  auto ready = channel.receive();
  assert(ready.message.at("type") == "frame_ready");
  const auto& frame = ready.message.at("frame");
  assert(frame.at("generation") == 21);
  const std::size_t buffer_id = frame.at("buffer_id");
  assert(buffer_id < descriptors.size());

  constexpr std::size_t frame_bytes = 64U * 48U * 3U / 2U;
  void* pixels = mmap(nullptr, frame_bytes, PROT_READ, MAP_SHARED,
                      descriptors[buffer_id].fds.front(), 0);
  assert(pixels != MAP_FAILED);
  const auto* bytes = static_cast<const unsigned char*>(pixels);
  assert(std::any_of(bytes, bytes + frame_bytes,
                     [](unsigned char value) { return value != 0; }));
  munmap(pixels, frame_bytes);

  channel.send({{"type", "frame_release"},
                {"release",
                 {{"generation", frame.at("generation")},
                  {"buffer_id", frame.at("buffer_id")},
                  {"sequence", frame.at("sequence")}}}});
  close(commands[1]);
  int status = 0;
  assert(waitpid(child, &status, 0) == child);
  assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);

  std::ifstream output_stream(output);
  std::string line;
  std::vector<nlohmann::json> events;
  while (std::getline(output_stream, line)) {
    events.push_back(nlohmann::json::parse(line));
  }
  const auto capabilities =
      std::find_if(events.begin(), events.end(), [](const auto& event) {
        return event.value("type", "") == "capabilities_changed";
      });
  assert(capabilities != events.end());
  assert(capabilities->at("copy_mode") == "shared_memory_yuv420p");
  assert(std::none_of(events.begin(), events.end(), [](const auto& event) {
    return event.value("type", "") == "first_frame_ready";
  }));

  std::filesystem::remove(fixture);
  std::filesystem::remove(output);
  return 0;
}
