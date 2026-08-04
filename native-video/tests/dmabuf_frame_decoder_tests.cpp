// SPDX-License-Identifier: MIT
#include "dmabuf_frame_decoder.h"

#include <fcntl.h>

#include <cassert>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>

using localbooru::native_video::DmabufFrameDecoder;

int main() {
  bool rejected_missing = false;
  try {
    DmabufFrameDecoder missing(
        "/definitely/missing/localbooru-dmabuf-decoder.mp4",
        "/dev/dri/renderD128");
  } catch (const std::runtime_error&) {
    rejected_missing = true;
  }
  assert(rejected_missing);

  const char* media = std::getenv("LOCALBOORU_DMABUF_TEST_MEDIA");
  const char* device = std::getenv("LOCALBOORU_DMABUF_TEST_DEVICE");
  if (!media || !std::filesystem::is_regular_file(media)) return 0;
  const std::string render_node = device ? device : "/dev/dri/renderD128";
  if (!std::filesystem::exists(render_node)) return 0;

  DmabufFrameDecoder decoder(media, render_node);
  auto first = decoder.next_frame();
  assert(first.has_value());
  assert(first->width() > 0);
  assert(first->height() > 0);
  assert(!first->objects().empty());
  assert(!first->layers().empty());
  assert(first->color_space() == "bt601" || first->color_space() == "bt709" ||
         first->color_space() == "bt2020");
  assert(first->color_range() == "narrow" || first->color_range() == "full");
  assert(first->chroma_location() == "left" ||
         first->chroma_location() == "center" ||
         first->chroma_location() == "top_left" ||
         first->chroma_location() == "top");
  for (const auto& object : first->objects()) {
    assert(object.fd >= 0);
    assert(fcntl(object.fd, F_GETFD) >= 0);
    assert(object.size > 0);
  }

  int decoded = 1;
  while (decoded < 120) {
    auto frame = decoder.next_frame();
    if (!frame) break;
    ++decoded;
  }
  assert(decoded == 120);
  // The mapped frame retains its hardware surface while later frames decode.
  for (const auto& object : first->objects()) {
    assert(fcntl(object.fd, F_GETFD) >= 0);
  }

  decoder.seek(0.5);
  auto sought = decoder.next_frame();
  assert(sought.has_value());
  assert(sought->pts_seconds() >= 0.5 - 1e-3);
  return 0;
}
