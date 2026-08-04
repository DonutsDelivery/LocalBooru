// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace localbooru::native_video {

struct FrameLease {
  std::size_t buffer_id = 0;
  std::uint64_t generation = 0;
  std::uint64_t sequence = 0;

  bool operator==(const FrameLease&) const = default;
};

class SurfacePool {
 public:
  explicit SurfacePool(std::size_t capacity);

  void configure(std::uint64_t generation);
  [[nodiscard]] std::optional<std::size_t> acquire_for_producer();
  [[nodiscard]] bool cancel_producer(std::size_t buffer_id);
  [[nodiscard]] std::optional<FrameLease> publish(std::size_t buffer_id);
  [[nodiscard]] bool release(const FrameLease& lease);

  [[nodiscard]] std::size_t capacity() const { return slots_.size(); }
  [[nodiscard]] std::size_t available() const;
  [[nodiscard]] std::size_t consumer_owned() const;

 private:
  enum class Ownership { Free, Producer, Consumer };

  struct Slot {
    Ownership ownership = Ownership::Free;
    std::uint64_t sequence = 0;
  };

  std::vector<Slot> slots_;
  std::uint64_t generation_ = 0;
  std::uint64_t next_sequence_ = 1;
};

}  // namespace localbooru::native_video
