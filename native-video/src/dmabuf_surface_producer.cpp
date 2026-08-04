// SPDX-License-Identifier: MIT
#include "dmabuf_surface_producer.h"

#include <stdexcept>
#include <utility>

namespace localbooru::native_video {

DmabufSurfaceProducer::DmabufSurfaceProducer(SurfaceChannel &channel,
                                             std::string producer_drm_node,
                                             std::size_t pool_size)
    : channel_(channel), producer_drm_node_(std::move(producer_drm_node)),
      pool_(pool_size), retained_frames_(pool_size) {
  if (producer_drm_node_.empty()) {
    throw std::invalid_argument("DMA-BUF producer DRM node is empty");
  }
}

void DmabufSurfaceProducer::configure(std::uint64_t generation) {
  generation_ = generation;
  for (auto &frame : retained_frames_)
    frame.reset();
  pool_.configure(generation);
}

void DmabufSurfaceProducer::reset() {
  generation_ = 0;
  for (auto &frame : retained_frames_)
    frame.reset();
  pool_.configure(0);
}

std::optional<std::size_t> DmabufSurfaceProducer::acquire_buffer() {
  if (generation_ == 0) {
    throw std::logic_error("DMA-BUF surface producer is not configured");
  }
  return pool_.acquire_for_producer();
}

bool DmabufSurfaceProducer::cancel_buffer(std::size_t buffer_id) {
  return pool_.cancel_producer(buffer_id);
}

std::optional<FrameLease>
DmabufSurfaceProducer::publish(DmabufVideoFrame frame) {
  const auto buffer_id = acquire_buffer();
  if (!buffer_id)
    return std::nullopt;
  try {
    return publish(*buffer_id, std::move(frame));
  } catch (...) {
    static_cast<void>(cancel_buffer(*buffer_id));
    throw;
  }
}

std::optional<FrameLease>
DmabufSurfaceProducer::publish(std::size_t buffer_id,
                               DmabufVideoFrame frame,
                               bool reusable_dmabuf) {
  if (generation_ == 0) {
    throw std::logic_error("DMA-BUF surface producer is not configured");
  }
  if (frame.width() <= 0 || frame.height() <= 0 || frame.width() > 16384 ||
      frame.height() > 16384) {
    throw std::invalid_argument("DMA-BUF frame dimensions are invalid");
  }
  if (frame.objects().empty() || frame.objects().size() > 5 ||
      frame.layers().empty()) {
    throw std::invalid_argument("DMA-BUF frame layout is invalid");
  }

  nlohmann::json objects = nlohmann::json::array();
  nlohmann::json layers = nlohmann::json::array();
  nlohmann::json legacy_planes = nlohmann::json::array();
  std::vector<int> object_fds;
  object_fds.reserve(frame.objects().size());
  for (const auto &object : frame.objects()) {
    if (object.fd < 0 || object.size == 0) {
      throw std::invalid_argument("DMA-BUF object is invalid");
    }
    objects.push_back({{"size", object.size}, {"modifier", object.modifier}});
    object_fds.push_back(object.fd);
  }
  for (const auto &layer : frame.layers()) {
    if (layer.width <= 0 || layer.height <= 0 || layer.width > frame.width() ||
        layer.height > frame.height() || layer.planes.empty() ||
        layer.planes.size() > 4) {
      throw std::invalid_argument("DMA-BUF layer geometry is invalid");
    }
    nlohmann::json planes = nlohmann::json::array();
    for (const auto &plane : layer.planes) {
      if (plane.object_index < 0 ||
          static_cast<std::size_t>(plane.object_index) >=
              frame.objects().size() ||
          plane.pitch == 0) {
        throw std::invalid_argument("DMA-BUF plane metadata is invalid");
      }
      const nlohmann::json encoded{{"object_index", plane.object_index},
                                   {"stride", plane.pitch},
                                   {"offset", plane.offset}};
      planes.push_back(encoded);
      if (legacy_planes.size() < 4) {
        legacy_planes.push_back(
            {{"stride", plane.pitch}, {"offset", plane.offset}});
      }
    }
    layers.push_back({{"fourcc", layer.format},
                      {"width", layer.width},
                      {"height", layer.height},
                      {"planes", std::move(planes)}});
  }
  if (legacy_planes.empty()) {
    throw std::invalid_argument("DMA-BUF frame contains no planes");
  }

  try {
    channel_.send({{"type", "surface_created"},
                   {"descriptor",
                    {{"generation", generation_},
                     {"buffer_id", buffer_id},
                     {"width", frame.width()},
                     {"height", frame.height()},
                     {"sample_aspect_ratio", frame.sample_aspect_ratio()},
                     {"rotation_degrees", frame.rotation_degrees()},
                     {"fourcc", frame.layers().front().format},
                     {"modifier", frame.objects().front().modifier},
                     {"handle_kind", "dma_buf"},
                     {"reusable_dmabuf", reusable_dmabuf},
                     {"producer_drm_node", producer_drm_node_},
                     {"color_space", frame.color_space()},
                     {"color_range", frame.color_range()},
                     {"chroma_location", frame.chroma_location()},
                     {"planes", std::move(legacy_planes)},
                     {"dmabuf",
                      {{"objects", std::move(objects)},
                       {"layers", std::move(layers)}}}}}},
                  object_fds);
  } catch (...) {
    static_cast<void>(pool_.cancel_producer(buffer_id));
    throw;
  }

  const auto lease = pool_.publish(buffer_id);
  if (!lease) {
    static_cast<void>(pool_.cancel_producer(buffer_id));
    throw std::logic_error("failed to publish acquired DMA-BUF surface");
  }
  const double pts_seconds = frame.pts_seconds();
  retained_frames_[buffer_id].emplace(std::move(frame));
  try {
    channel_.send({{"type", "frame_ready"},
                   {"frame",
                    {{"generation", lease->generation},
                     {"buffer_id", lease->buffer_id},
                     {"sequence", lease->sequence},
                     {"pts_seconds", pts_seconds},
                     {"has_native_fence", false}}}});
  } catch (...) {
    static_cast<void>(pool_.release(*lease));
    retained_frames_[buffer_id].reset();
    throw;
  }
  return lease;
}

bool DmabufSurfaceProducer::receive_release() {
  auto packet = channel_.receive();
  return apply_release(packet);
}

std::size_t DmabufSurfaceProducer::drain_releases() {
  std::size_t released = 0;
  while (auto packet = channel_.try_receive()) {
    if (apply_release(*packet))
      ++released;
  }
  return released;
}

bool DmabufSurfaceProducer::apply_release(const ReceivedSurfacePacket &packet) {
  if (!packet.fds.empty() ||
      packet.message.value("type", "") != "frame_release") {
    return false;
  }
  const auto &release = packet.message.at("release");
  const FrameLease lease{release.at("buffer_id").get<std::size_t>(),
                         release.at("generation").get<std::uint64_t>(),
                         release.at("sequence").get<std::uint64_t>()};
  if (!pool_.release(lease))
    return false;
  retained_frames_.at(lease.buffer_id).reset();
  return true;
}

} // namespace localbooru::native_video
