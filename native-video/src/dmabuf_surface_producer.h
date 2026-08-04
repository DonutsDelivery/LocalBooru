// SPDX-License-Identifier: MIT
#pragma once

#include "dmabuf_frame_decoder.h"
#include "surface_channel.h"
#include "surface_pool.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace localbooru::native_video {

class DmabufSurfaceProducer {
 public:
  DmabufSurfaceProducer(SurfaceChannel& channel, std::string producer_drm_node,
                        std::size_t pool_size = 3);
  DmabufSurfaceProducer(const DmabufSurfaceProducer&) = delete;
  DmabufSurfaceProducer& operator=(const DmabufSurfaceProducer&) = delete;

  void configure(std::uint64_t generation);
  void reset();
  [[nodiscard]] std::optional<std::size_t> acquire_buffer();
  [[nodiscard]] bool cancel_buffer(std::size_t buffer_id);
  [[nodiscard]] std::optional<FrameLease> publish(DmabufVideoFrame frame);
  [[nodiscard]] std::optional<FrameLease> publish(
      std::size_t buffer_id, DmabufVideoFrame frame,
      bool reusable_dmabuf = false);
  [[nodiscard]] bool receive_release();
  [[nodiscard]] std::size_t drain_releases();
  [[nodiscard]] std::size_t available() const { return pool_.available(); }

 private:
  [[nodiscard]] bool apply_release(const ReceivedSurfacePacket& packet);

  SurfaceChannel& channel_;
  std::string producer_drm_node_;
  SurfacePool pool_;
  std::vector<std::optional<DmabufVideoFrame>> retained_frames_;
  std::uint64_t generation_ = 0;
};

}  // namespace localbooru::native_video
