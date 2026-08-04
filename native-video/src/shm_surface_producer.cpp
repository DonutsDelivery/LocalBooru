// SPDX-License-Identifier: MIT
#include "shm_surface_producer.h"

#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace localbooru::native_video {
namespace {
constexpr std::uint32_t kDrmFormatAbgr8888 = 0x34324241U; // DRM fourcc AB24.
constexpr std::uint32_t kDrmFormatYuv420 = 0x32315559U;   // DRM fourcc YU12.

std::runtime_error system_error(const char *action) {
  return std::runtime_error(std::string(action) + ": " + std::strerror(errno));
}
} // namespace

ShmSurfaceProducer::ShmSurfaceProducer(SurfaceChannel &channel,
                                       std::size_t pool_size)
    : channel_(channel), pool_(pool_size) {}

ShmSurfaceProducer::~ShmSurfaceProducer() { clear(); }

void ShmSurfaceProducer::reset() { clear(); }

void ShmSurfaceProducer::clear() {
  for (auto &buffer : buffers_) {
    if (buffer.mapping != nullptr && buffer.mapping != MAP_FAILED) {
      munmap(buffer.mapping, buffer.size);
    }
    if (buffer.fd >= 0)
      close(buffer.fd);
  }
  buffers_.clear();
  width_ = 0;
  height_ = 0;
  yuv420p_ = false;
  sample_aspect_ratio_ = 1.0;
  rotation_degrees_ = 0;
  color_space_.clear();
  color_range_.clear();
  chroma_location_.clear();
}

void ShmSurfaceProducer::configure(
    std::uint64_t generation, int width, int height, bool yuv420p,
    double sample_aspect_ratio, int rotation_degrees, std::string color_space,
    std::string color_range, std::string chroma_location) {
  if (width <= 0 || height <= 0 || width > 16384 || height > 16384) {
    throw std::invalid_argument("shared surface dimensions are invalid");
  }
  if (!std::isfinite(sample_aspect_ratio) || sample_aspect_ratio <= 0.0 ||
      (rotation_degrees % 90) != 0) {
    throw std::invalid_argument("shared surface display metadata is invalid");
  }
  const std::size_t row_bytes =
      static_cast<std::size_t>(width) * (yuv420p ? 1U : 4U);
  if (static_cast<std::size_t>(height) >
      std::numeric_limits<std::size_t>::max() / row_bytes) {
    throw std::overflow_error("shared surface size overflow");
  }
  const std::size_t luma_size = row_bytes * static_cast<std::size_t>(height);
  const std::size_t size = yuv420p ? luma_size + luma_size / 2U : luma_size;

  clear();
  generation_ = generation;
  width_ = width;
  height_ = height;
  yuv420p_ = yuv420p;
  sample_aspect_ratio_ = sample_aspect_ratio;
  rotation_degrees_ = (rotation_degrees % 360 + 360) % 360;
  color_space_ = std::move(color_space);
  color_range_ = std::move(color_range);
  chroma_location_ = std::move(chroma_location);
  pool_.configure(generation);
  buffers_.reserve(pool_.capacity());
  try {
    for (std::size_t index = 0; index < pool_.capacity(); ++index) {
      const int fd = memfd_create("localbooru-native-frame", MFD_CLOEXEC);
      if (fd < 0)
        throw system_error("failed to allocate shared frame");
      if (ftruncate(fd, static_cast<off_t>(size)) != 0) {
        const auto error = system_error("failed to size shared frame");
        close(fd);
        throw error;
      }
      void *mapping =
          mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
      if (mapping == MAP_FAILED) {
        const auto error = system_error("failed to map shared frame");
        close(fd);
        throw error;
      }
      buffers_.push_back(Buffer{fd, mapping, size});
      const nlohmann::json planes =
          yuv420p_
              ? nlohmann::json::array(
                    {{{"stride", width_}, {"offset", 0}},
                     {{"stride", width_ / 2}, {"offset", luma_size}},
                     {{"stride", width_ / 2},
                      {"offset", luma_size + luma_size / 4U}}})
              : nlohmann::json::array({{{"stride", row_bytes}, {"offset", 0}}});
      channel_.send(
          {{"type", "surface_created"},
           {"descriptor",
            {{"generation", generation_},
             {"buffer_id", index},
             {"width", width_},
             {"height", height_},
             {"sample_aspect_ratio", sample_aspect_ratio_},
             {"rotation_degrees", rotation_degrees_},
             {"fourcc", yuv420p_ ? kDrmFormatYuv420 : kDrmFormatAbgr8888},
             {"modifier", 0},
             {"handle_kind", "shared_memory"},
             {"producer_drm_node", nullptr},
             {"color_space", color_space_.empty()
                                 ? nlohmann::json(nullptr)
                                 : nlohmann::json(color_space_)},
             {"color_range", color_range_.empty()
                                 ? nlohmann::json(nullptr)
                                 : nlohmann::json(color_range_)},
             {"chroma_location", chroma_location_.empty()
                                     ? nlohmann::json(nullptr)
                                     : nlohmann::json(chroma_location_)},
             {"planes", planes}}}},
          {fd});
    }
  } catch (...) {
    clear();
    throw;
  }
}

std::optional<FrameLease>
ShmSurfaceProducer::publish(const DecodedVideoFrame &frame) {
  if (frame.width != width_ || frame.height != height_) {
    throw std::invalid_argument(
        "decoded frame dimensions do not match surface pool");
  }
  const std::size_t pixels =
      static_cast<std::size_t>(width_) * static_cast<std::size_t>(height_);
  const std::size_t expected = yuv420p_ ? pixels * 3U / 2U : pixels * 4U;
  if (frame.yuv420p != yuv420p_) {
    throw std::invalid_argument(
        "decoded frame format does not match surface pool");
  }
  if (frame.rgba.size() != expected) {
    throw std::invalid_argument(
        "decoded frame RGBA payload has an invalid size");
  }
  const auto buffer_id = pool_.acquire_for_producer();
  if (!buffer_id)
    return std::nullopt;
  auto &buffer = buffers_.at(*buffer_id);
  std::memcpy(buffer.mapping, frame.rgba.data(), expected);
  const auto lease = pool_.publish(*buffer_id);
  if (!lease)
    throw std::logic_error("failed to publish acquired surface");
  channel_.send({{"type", "frame_ready"},
                 {"frame",
                  {{"generation", lease->generation},
                   {"buffer_id", lease->buffer_id},
                   {"sequence", lease->sequence},
                   {"pts_seconds", frame.pts_seconds},
                   {"has_native_fence", false}}}});
  return lease;
}

bool ShmSurfaceProducer::receive_release() {
  auto packet = channel_.receive();
  return apply_release(packet);
}

std::size_t ShmSurfaceProducer::drain_releases() {
  std::size_t released = 0;
  while (auto packet = channel_.try_receive()) {
    if (apply_release(*packet))
      ++released;
  }
  return released;
}

bool ShmSurfaceProducer::apply_release(const ReceivedSurfacePacket &packet) {
  if (!packet.fds.empty() ||
      packet.message.value("type", "") != "frame_release") {
    return false;
  }
  const auto &release = packet.message.at("release");
  return pool_.release(FrameLease{release.at("buffer_id").get<std::size_t>(),
                                  release.at("generation").get<std::uint64_t>(),
                                  release.at("sequence").get<std::uint64_t>()});
}

} // namespace localbooru::native_video
