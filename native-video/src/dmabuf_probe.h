// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace localbooru::native_video {

struct DmabufPlaneProbe {
  int object_index = -1;
  std::uint32_t offset = 0;
  std::uint32_t pitch = 0;
};

struct DmabufLayerProbe {
  std::uint32_t format = 0;
  std::vector<DmabufPlaneProbe> planes;
};

struct DmabufObjectProbe {
  std::uint64_t size = 0;
  std::uint64_t modifier = 0;
};

struct DmabufProbeResult {
  bool available = false;
  std::string reason;
  std::string device;
  int width = 0;
  int height = 0;
  std::vector<DmabufObjectProbe> objects;
  std::vector<DmabufLayerProbe> layers;
};

// Decodes one frame through FFmpeg VA-API and maps it to AV_PIX_FMT_DRM_PRIME.
// The returned metadata intentionally excludes live descriptors; this probe
// establishes export compatibility before frame ownership is wired into the
// bounded surface protocol.
DmabufProbeResult probe_dmabuf_export(const std::string& media_path,
                                      const std::string& render_node);

}  // namespace localbooru::native_video
