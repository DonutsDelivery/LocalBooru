// SPDX-License-Identifier: MIT
#pragma once

#include "decoder.h"
#include "dmabuf_frame_decoder.h"

#include <cstddef>
#include <memory>
#include <optional>
#include <string>

namespace localbooru::native_video {

class DmabufFrameUploader {
public:
  DmabufFrameUploader(const std::string &render_node, int width, int height,
                      double sample_aspect_ratio, int rotation_degrees,
                      std::string color_space, std::string color_range,
                      std::string chroma_location);
  ~DmabufFrameUploader();
  DmabufFrameUploader(const DmabufFrameUploader &) = delete;
  DmabufFrameUploader &operator=(const DmabufFrameUploader &) = delete;

  [[nodiscard]] DmabufVideoFrame upload(
      const DecodedVideoFrame &frame, std::size_t buffer_id);
private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace localbooru::native_video
