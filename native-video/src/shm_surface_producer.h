// SPDX-License-Identifier: MIT
#pragma once

#include "decoder.h"
#include "surface_channel.h"
#include "surface_pool.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace localbooru::native_video {

class ShmSurfaceProducer {
public:
  explicit ShmSurfaceProducer(SurfaceChannel &channel,
                              std::size_t pool_size = 3);
  ~ShmSurfaceProducer();
  ShmSurfaceProducer(const ShmSurfaceProducer &) = delete;
  ShmSurfaceProducer &operator=(const ShmSurfaceProducer &) = delete;

  void configure(std::uint64_t generation, int width, int height,
                 bool yuv420p = false, double sample_aspect_ratio = 1.0,
                 int rotation_degrees = 0, std::string color_space = {},
                 std::string color_range = {},
                 std::string chroma_location = {});
  void reset();
  [[nodiscard]] std::optional<FrameLease>
  publish(const DecodedVideoFrame &frame);
  [[nodiscard]] bool receive_release();
  [[nodiscard]] std::size_t drain_releases();
  [[nodiscard]] std::size_t available() const { return pool_.available(); }

private:
  struct Buffer {
    int fd = -1;
    void *mapping = nullptr;
    std::size_t size = 0;
  };

  void clear();
  [[nodiscard]] bool apply_release(const ReceivedSurfacePacket &packet);

  SurfaceChannel &channel_;
  SurfacePool pool_;
  std::vector<Buffer> buffers_;
  std::uint64_t generation_ = 0;
  int width_ = 0;
  int height_ = 0;
  bool yuv420p_ = false;
  double sample_aspect_ratio_ = 1.0;
  int rotation_degrees_ = 0;
  std::string color_space_;
  std::string color_range_;
  std::string chroma_location_;
};

} // namespace localbooru::native_video
