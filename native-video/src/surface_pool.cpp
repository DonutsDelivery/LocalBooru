// SPDX-License-Identifier: MIT
#include "surface_pool.h"

#include <algorithm>
#include <stdexcept>

namespace localbooru::native_video {

SurfacePool::SurfacePool(std::size_t capacity) : slots_(capacity) {
  if (capacity == 0) throw std::invalid_argument("surface pool cannot be empty");
}

void SurfacePool::configure(std::uint64_t generation) {
  generation_ = generation;
  for (auto& slot : slots_) {
    slot.ownership = Ownership::Free;
    slot.sequence = 0;
  }
}

std::optional<std::size_t> SurfacePool::acquire_for_producer() {
  for (std::size_t index = 0; index < slots_.size(); ++index) {
    if (slots_[index].ownership != Ownership::Free) continue;
    slots_[index].ownership = Ownership::Producer;
    return index;
  }
  return std::nullopt;
}

bool SurfacePool::cancel_producer(std::size_t buffer_id) {
  if (buffer_id >= slots_.size() ||
      slots_[buffer_id].ownership != Ownership::Producer) {
    return false;
  }
  slots_[buffer_id].ownership = Ownership::Free;
  return true;
}

std::optional<FrameLease> SurfacePool::publish(std::size_t buffer_id) {
  if (buffer_id >= slots_.size()) return std::nullopt;
  auto& slot = slots_[buffer_id];
  if (slot.ownership != Ownership::Producer) return std::nullopt;
  slot.ownership = Ownership::Consumer;
  slot.sequence = next_sequence_++;
  return FrameLease{buffer_id, generation_, slot.sequence};
}

bool SurfacePool::release(const FrameLease& lease) {
  if (lease.buffer_id >= slots_.size() || lease.generation != generation_) {
    return false;
  }
  auto& slot = slots_[lease.buffer_id];
  if (slot.ownership != Ownership::Consumer ||
      slot.sequence != lease.sequence) {
    return false;
  }
  slot.ownership = Ownership::Free;
  return true;
}

std::size_t SurfacePool::available() const {
  return static_cast<std::size_t>(std::count_if(
      slots_.begin(), slots_.end(),
      [](const Slot& slot) { return slot.ownership == Ownership::Free; }));
}

std::size_t SurfacePool::consumer_owned() const {
  return static_cast<std::size_t>(std::count_if(
      slots_.begin(), slots_.end(), [](const Slot& slot) {
        return slot.ownership == Ownership::Consumer;
      }));
}

}  // namespace localbooru::native_video
