// SPDX-License-Identifier: MIT
#include "dmabuf_probe.h"

#include <cassert>
#include <string>

using localbooru::native_video::probe_dmabuf_export;

int main() {
  const auto missing = probe_dmabuf_export(
      "/definitely/missing/localbooru-native-video-probe.mp4",
      "/dev/dri/renderD128");
  assert(!missing.available);
  assert(missing.reason.find("open failed") != std::string::npos);
  assert(missing.objects.empty());
  assert(missing.layers.empty());
  return 0;
}
