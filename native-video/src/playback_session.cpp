// SPDX-License-Identifier: MIT
#include "playback_session.h"

#include "svp_frame_source.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

namespace localbooru::native_video {

namespace {
class FrameSource {
public:
  virtual ~FrameSource() = default;
  virtual std::optional<DecodedVideoFrame> next_frame() = 0;
  virtual void seek(double position_seconds) = 0;
  virtual void interrupt() = 0;
};

class DecoderSource final : public FrameSource {
public:
  DecoderSource(const std::string &path, double position) : decoder_(path) {
    if (position > 0.0)
      decoder_.seek(position);
  }

  std::optional<DecodedVideoFrame> next_frame() override {
    return decoder_.next_frame();
  }
  void seek(double position_seconds) override {
    decoder_.seek(position_seconds);
  }
  void interrupt() override {}

private:
  VideoFrameDecoder decoder_;
};

class SvpSource final : public FrameSource {
public:
  SvpSource(const std::string &path, double position,
            const PlaybackOpenOptions &options)
      : source_(path, probe_media(path),
                {.preset = options.interpolation_preset,
                 .target_fps = options.target_fps,
                 .plugin_path = {}}) {
    source_.start(position);
  }

  std::optional<DecodedVideoFrame> next_frame() override {
    return source_.next_frame();
  }
  void seek(double position_seconds) override {
    source_.seek(position_seconds);
  }
  void interrupt() override { source_.interrupt(); }

private:
  SvpFrameSource source_;
};
} // namespace

struct VideoPlaybackSession::Impl {
  explicit Impl(PlaybackCallbacks value) : callbacks(std::move(value)) {}

  PlaybackCallbacks callbacks;
  std::mutex mutex;
  std::condition_variable condition;
  std::thread worker;
  bool stop_requested = false;
  bool paused = true;
  std::atomic<double> speed{1.0};
  std::optional<double> pending_seek;
  std::shared_ptr<FrameSource> active_source;

  void stop() {
    std::shared_ptr<FrameSource> source;
    {
      std::lock_guard lock(mutex);
      stop_requested = true;
      source = active_source;
      condition.notify_all();
    }
    if (source)
      source->interrupt();
    if (worker.joinable())
      worker.join();
    {
      std::lock_guard lock(mutex);
      active_source.reset();
    }
  }

  void run(std::string path, double resume_position,
           PlaybackOpenOptions options) {
    try {
      std::shared_ptr<FrameSource> source;
      if (options.interpolation_engine == "svp") {
        source = std::make_shared<SvpSource>(path, resume_position, options);
      } else if (options.interpolation_engine == "off") {
        source = std::make_shared<DecoderSource>(path, resume_position);
      } else {
        throw std::runtime_error("unsupported interpolation engine: " +
                                 options.interpolation_engine);
      }
      {
        std::lock_guard lock(mutex);
        active_source = source;
      }

      using Clock = std::chrono::steady_clock;
      double media_anchor = resume_position;
      auto wall_anchor = Clock::now();
      bool timing_anchored = false;
      bool needs_paused_preview = true;

      for (;;) {
        {
          std::unique_lock lock(mutex);
          condition.wait(lock, [&] {
            return stop_requested || !paused || pending_seek.has_value() ||
                   needs_paused_preview;
          });
          if (stop_requested)
            return;
          if (pending_seek) {
            const double target = *pending_seek;
            pending_seek.reset();
            lock.unlock();
            source->seek(target);
            media_anchor = target;
            wall_anchor = Clock::now();
            timing_anchored = false;
            needs_paused_preview = true;
          }
        }

        auto frame = source->next_frame();
        if (!frame) {
          std::unique_lock lock(mutex);
          if (stop_requested)
            return;
          if (pending_seek)
            continue;
          lock.unlock();
          if (callbacks.on_ended)
            callbacks.on_ended();
          lock.lock();
          condition.wait(lock,
                         [&] { return stop_requested || pending_seek.has_value(); });
          if (stop_requested)
            return;
          continue;
        }
        if (!timing_anchored) {
          media_anchor = frame->pts_seconds;
          wall_anchor = Clock::now();
          timing_anchored = true;
        }

        bool paused_preview = false;
        {
          std::lock_guard lock(mutex);
          paused_preview = paused && needs_paused_preview;
          if (paused_preview)
            needs_paused_preview = false;
        }
        if (paused_preview) {
          if (callbacks.on_frame)
            callbacks.on_frame(*frame);
          continue;
        }

        bool discard_for_seek = false;
        for (;;) {
          std::unique_lock lock(mutex);
          if (stop_requested)
            return;
          if (pending_seek) {
            discard_for_seek = true;
            break;
          }
          if (paused) {
            const auto paused_at = Clock::now();
            condition.wait(lock, [&] {
              return stop_requested || !paused || pending_seek.has_value();
            });
            wall_anchor += Clock::now() - paused_at;
            continue;
          }

          const auto target =
              wall_anchor + std::chrono::duration_cast<Clock::duration>(
                                std::chrono::duration<double>(std::max(
                                    0.0, (frame->pts_seconds - media_anchor) /
                                             speed.load())));
          if (condition.wait_until(lock, target, [&] {
                return stop_requested || paused || pending_seek.has_value();
              })) {
            continue;
          }
          break;
        }
        if (discard_for_seek)
          continue;
        needs_paused_preview = false;
        if (callbacks.on_frame)
          callbacks.on_frame(*frame);
      }
      {
        std::lock_guard lock(mutex);
        active_source.reset();
      }
    } catch (const std::exception &error) {
      {
        std::lock_guard lock(mutex);
        active_source.reset();
      }
      if (callbacks.on_error)
        callbacks.on_error(error.what());
    }
  }
};

VideoPlaybackSession::VideoPlaybackSession(PlaybackCallbacks callbacks)
    : impl_(std::make_unique<Impl>(std::move(callbacks))) {}

VideoPlaybackSession::~VideoPlaybackSession() { stop(); }

void VideoPlaybackSession::open(const std::string &path, double resume_position,
                                bool autoplay) {
  open(path, resume_position, PlaybackOpenOptions{.autoplay = autoplay});
}

void VideoPlaybackSession::open(const std::string &path, double resume_position,
                                PlaybackOpenOptions options) {
  stop();
  const double position =
      std::isfinite(resume_position) ? std::max(0.0, resume_position) : 0.0;
  {
    std::lock_guard lock(impl_->mutex);
    impl_->stop_requested = false;
    impl_->paused = !options.autoplay;
    impl_->pending_seek.reset();
  }
  impl_->worker = std::thread([impl = impl_.get(), path, position,
                               options = std::move(options)]() mutable {
    impl->run(path, position, std::move(options));
  });
}

void VideoPlaybackSession::set_paused(bool paused) {
  std::lock_guard lock(impl_->mutex);
  impl_->paused = paused;
  impl_->condition.notify_all();
}

void VideoPlaybackSession::set_speed(double speed) {
  impl_->speed.store(std::isfinite(speed) ? std::clamp(speed, 0.5, 2.0) : 1.0);
  impl_->condition.notify_all();
}

void VideoPlaybackSession::seek(double position_seconds) {
  const double position =
      std::isfinite(position_seconds) ? std::max(0.0, position_seconds) : 0.0;
  std::shared_ptr<FrameSource> source;
  {
    std::lock_guard lock(impl_->mutex);
    impl_->pending_seek = position;
    source = impl_->active_source;
    impl_->condition.notify_all();
  }
  if (source)
    source->interrupt();
}

void VideoPlaybackSession::stop() { impl_->stop(); }

double VideoPlaybackSession::speed() const { return impl_->speed.load(); }

} // namespace localbooru::native_video
