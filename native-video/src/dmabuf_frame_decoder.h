// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

struct AVBufferRef;
struct AVFrame;

namespace localbooru::native_video {

struct DmabufFramePlane {
  int object_index = -1;
  std::uint32_t offset = 0;
  std::uint32_t pitch = 0;
};

struct DmabufFrameLayer {
  std::uint32_t format = 0;
  int width = 0;
  int height = 0;
  std::vector<DmabufFramePlane> planes;
};

struct DmabufFrameObject {
  int fd = -1; // Borrowed; valid for the lifetime of DmabufVideoFrame.
  std::uint64_t size = 0;
  std::uint64_t modifier = 0;
};

class DmabufVideoFrame {
public:
  ~DmabufVideoFrame();
  DmabufVideoFrame(const DmabufVideoFrame &) = delete;
  DmabufVideoFrame &operator=(const DmabufVideoFrame &) = delete;
  DmabufVideoFrame(DmabufVideoFrame &&) noexcept;
  DmabufVideoFrame &operator=(DmabufVideoFrame &&) noexcept;

  [[nodiscard]] int width() const;
  [[nodiscard]] int height() const;
  [[nodiscard]] double sample_aspect_ratio() const;
  [[nodiscard]] int rotation_degrees() const;
  [[nodiscard]] double pts_seconds() const;
  [[nodiscard]] const std::string &color_space() const;
  [[nodiscard]] const std::string &color_range() const;
  [[nodiscard]] const std::string &chroma_location() const;
  [[nodiscard]] const std::vector<DmabufFrameObject> &objects() const;
  [[nodiscard]] const std::vector<DmabufFrameLayer> &layers() const;

private:
  struct Impl;
  explicit DmabufVideoFrame(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
  friend class DmabufFrameDecoder;
  friend class DmabufFrameUploader;
};

class DmabufFrameDecoder {
public:
  DmabufFrameDecoder(const std::string &media_path,
                     const std::string &render_node);
  ~DmabufFrameDecoder();
  DmabufFrameDecoder(const DmabufFrameDecoder &) = delete;
  DmabufFrameDecoder &operator=(const DmabufFrameDecoder &) = delete;

  [[nodiscard]] std::optional<DmabufVideoFrame> next_frame();
  void seek(double position_seconds);
  [[nodiscard]] static DmabufVideoFrame
  export_frame(AVFrame *hardware_frame, AVBufferRef *hardware_device,
               double pts_seconds, double sample_aspect_ratio,
               int rotation_degrees, const std::string &color_space,
               const std::string &color_range,
               const std::string &chroma_location);

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace localbooru::native_video
