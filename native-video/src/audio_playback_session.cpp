// SPDX-License-Identifier: MIT
#include "audio_playback_session.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <thread>

#include "audio_decoder.h"
#include "audio_output.h"

namespace localbooru::native_video {

struct AudioPlaybackSession::Impl {
  mutable std::mutex mutex;
  std::condition_variable condition;
  std::thread worker;
  bool stop_requested = false;
  bool paused = true;
  std::optional<double> pending_seek;
  std::atomic<double> submitted{0.0};
  std::atomic<double> queued{0.0};
  std::atomic<double> volume{1.0};
  std::atomic<double> speed{1.0};
  std::string error;

  void stop() {
    {
      std::lock_guard lock(mutex);
      stop_requested = true;
      condition.notify_all();
    }
    if (worker.joinable()) worker.join();
  }

  void run(std::string path, double resume_position, int stream_index) {
    try {
      constexpr int sample_rate = 48000;
      constexpr int channels = 2;
      AudioFrameDecoder decoder(path, sample_rate, channels, stream_index);
      SdlAudioOutput output;
      output.open(sample_rate, channels);
      if (resume_position > 0.0) decoder.seek(resume_position);
      submitted.store(resume_position);
      queued.store(0.0);

      for (;;) {
        {
          std::unique_lock lock(mutex);
          output.set_paused(paused);
          condition.wait(lock, [&] {
            return stop_requested || !paused || pending_seek.has_value();
          });
          if (stop_requested) return;
          if (pending_seek) {
            const double target = *pending_seek;
            pending_seek.reset();
            lock.unlock();
            output.clear();
            queued.store(0.0);
            decoder.seek(target);
            submitted.store(target);
          }
          output.set_paused(false);
        }

        if (output.queued_seconds() >= 0.5) {
          queued.store(output.queued_seconds());
          std::unique_lock lock(mutex);
          condition.wait_for(lock, std::chrono::milliseconds(10), [&] {
            return stop_requested || paused || pending_seek.has_value();
          });
          continue;
        }

        auto chunk = decoder.next_chunk();
        if (!chunk) {
          while (output.queued_seconds() > 0.0) {
            std::unique_lock lock(mutex);
            if (condition.wait_for(lock, std::chrono::milliseconds(10), [&] {
                  return stop_requested || paused || pending_seek.has_value();
                })) {
              break;
            }
          }
          std::unique_lock lock(mutex);
          if (stop_requested) return;
          if (pending_seek || paused) continue;
          condition.wait(lock, [&] {
            return stop_requested || pending_seek.has_value() || paused;
          });
          if (stop_requested) return;
          continue;
        }
        apply_audio_speed(*chunk, speed.load());
        apply_audio_gain(*chunk, volume.load());
        output.queue(*chunk);
        queued.store(output.queued_seconds());
        const double chunk_duration =
            static_cast<double>(chunk->samples.size()) /
            static_cast<double>(chunk->sample_rate * chunk->channels);
        submitted.store(chunk->pts_seconds + chunk_duration);
      }
    } catch (const std::exception& exception) {
      std::lock_guard lock(mutex);
      error = exception.what();
      condition.notify_all();
    }
  }
};

AudioPlaybackSession::AudioPlaybackSession() : impl_(std::make_unique<Impl>()) {}
AudioPlaybackSession::~AudioPlaybackSession() { stop(); }

void AudioPlaybackSession::open(const std::string& path, double resume_position,
                                bool autoplay, int stream_index) {
  stop();
  const double position = std::isfinite(resume_position)
                              ? std::max(0.0, resume_position)
                              : 0.0;
  {
    std::lock_guard lock(impl_->mutex);
    impl_->stop_requested = false;
    impl_->paused = !autoplay;
    impl_->pending_seek.reset();
    impl_->error.clear();
    impl_->submitted.store(position);
    impl_->queued.store(0.0);
  }
  impl_->worker = std::thread([impl = impl_.get(), path, position, stream_index] {
    impl->run(path, position, stream_index);
  });
}

void AudioPlaybackSession::set_paused(bool paused) {
  std::lock_guard lock(impl_->mutex);
  impl_->paused = paused;
  impl_->condition.notify_all();
}

void AudioPlaybackSession::set_volume(double volume) {
  impl_->volume.store(std::isfinite(volume) ? std::clamp(volume, 0.0, 1.0)
                                            : 1.0);
}

void AudioPlaybackSession::set_speed(double speed) {
  impl_->speed.store(std::isfinite(speed) ? std::clamp(speed, 0.5, 2.0)
                                          : 1.0);
}

void AudioPlaybackSession::seek(double position_seconds) {
  const double position = std::isfinite(position_seconds)
                              ? std::max(0.0, position_seconds)
                              : 0.0;
  std::lock_guard lock(impl_->mutex);
  impl_->pending_seek = position;
  impl_->condition.notify_all();
}

void AudioPlaybackSession::stop() { impl_->stop(); }

double AudioPlaybackSession::submitted_seconds() const {
  return impl_->submitted.load();
}

double AudioPlaybackSession::playback_position() const {
  return std::max(0.0,
                  impl_->submitted.load() -
                      impl_->queued.load() * impl_->speed.load());
}

double AudioPlaybackSession::volume() const { return impl_->volume.load(); }

double AudioPlaybackSession::speed() const { return impl_->speed.load(); }

std::string AudioPlaybackSession::last_error() const {
  std::lock_guard lock(impl_->mutex);
  return impl_->error;
}

}  // namespace localbooru::native_video
