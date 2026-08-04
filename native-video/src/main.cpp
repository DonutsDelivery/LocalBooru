// SPDX-License-Identifier: MIT
#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "audio_playback_session.h"
#include "decoder.h"
#include "dmabuf_frame_decoder.h"
#include "dmabuf_frame_uploader.h"
#include "dmabuf_surface_producer.h"
#include "frame_pacer.h"
#include "playback_session.h"
#include "presenter.h"
#include "protocol.h"
#include "shm_surface_producer.h"
#include "subtitles.h"
#include "surface_channel.h"

using namespace localbooru::native_video;

namespace {
std::mutex output_mutex;

void emit(const nlohmann::json &event) {
  std::lock_guard lock(output_mutex);
  std::cout << event.dump() << '\n' << std::flush;
}

std::optional<std::filesystem::path> discover_svp_plugins() {
  std::vector<std::filesystem::path> candidates;
  if (const char *configured = std::getenv("LOCALBOORU_SVP_PLUGIN_PATH")) {
    candidates.emplace_back(configured);
  }
  candidates.emplace_back("/opt/svp/plugins");
  candidates.emplace_back("/usr/lib/svp/plugins");
  for (const auto &candidate : candidates) {
    if (std::filesystem::is_regular_file(candidate / "libsvpflow1.so") &&
        std::filesystem::is_regular_file(candidate / "libsvpflow2.so")) {
      return candidate;
    }
  }
  return std::nullopt;
}

bool external_svp_available() {
  return ::access("/usr/bin/vspipe", X_OK) == 0 &&
         discover_svp_plugins().has_value();
}

struct ExternalSubtitleTrack {
  std::string id;
  std::filesystem::path path;
  std::string label;
  std::string language;
  std::string source_type = "sidecar";
  std::string cue_format = "vtt";
};

std::vector<ExternalSubtitleTrack>
discover_external_subtitles(const std::string &media_path) {
  const std::filesystem::path media(media_path);
  const auto directory = media.parent_path();
  const auto stem = media.stem().string();
  std::vector<ExternalSubtitleTrack> result;
  std::error_code error;
  for (const auto &entry :
       std::filesystem::directory_iterator(directory, error)) {
    if (error || !entry.is_regular_file())
      continue;
    const auto path = entry.path();
    std::string cue_format = path.extension().string();
    std::transform(cue_format.begin(), cue_format.end(), cue_format.begin(),
                   [](unsigned char value) { return std::tolower(value); });
    if (cue_format != ".vtt" && cue_format != ".srt" && cue_format != ".ass")
      continue;
    cue_format.erase(0, 1);
    const auto subtitle_stem = path.stem().string();
    if (subtitle_stem != stem && subtitle_stem.rfind(stem + ".", 0) != 0)
      continue;
    std::string suffix =
        subtitle_stem == stem ? "" : subtitle_stem.substr(stem.size() + 1);
    std::string language = suffix;
    if (const auto translated = language.find(".translated");
        translated != std::string::npos) {
      language.resize(translated);
    }
    std::string source_type = "sidecar";
    std::ifstream metadata(
        std::filesystem::path(path.string() + ".localbooru.json"));
    if (metadata) {
      try {
        const auto document = nlohmann::json::parse(metadata);
        if (document.value("source_type", "") == "whisper") {
          source_type = "whisper";
          language = document.value("language", language);
        }
      } catch (const nlohmann::json::exception &) {
      }
    }
    result.push_back({source_type + ":" + path.filename().string(), path,
                      suffix.empty() ? "External subtitles" : suffix, language,
                      source_type, cue_format});
  }
  std::sort(
      result.begin(), result.end(),
      [](const auto &left, const auto &right) { return left.id < right.id; });
  return result;
}

std::string read_subtitle_file(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("failed to open subtitle track");
  input.seekg(0, std::ios::end);
  const auto size = input.tellg();
  if (size < 0 || size > 16 * 1024 * 1024) {
    throw std::runtime_error("subtitle track exceeds the 16 MiB limit");
  }
  input.seekg(0);
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}
} // namespace

int main() {
  std::string line;
  if (!std::getline(std::cin, line))
    return 2;

  std::optional<SurfaceChannel> surface_channel;
  const char *dmabuf_setting = std::getenv("LOCALBOORU_NATIVE_DMABUF");
  const bool dmabuf_requested =
      dmabuf_setting && std::string(dmabuf_setting) == "1";
  const char *svp_dmabuf_upload_setting =
      std::getenv("LOCALBOORU_SVP_DMABUF_UPLOAD");
  const bool svp_dmabuf_upload_enabled = svp_dmabuf_upload_setting &&
                                         std::string(svp_dmabuf_upload_setting) == "1";
  const char *drm_node_setting = std::getenv("LOCALBOORU_NATIVE_DRM_NODE");
  const std::string drm_node =
      drm_node_setting ? drm_node_setting : "/dev/dri/renderD128";
  const bool svp_available = external_svp_available();

  try {
    const auto hello = nlohmann::json::parse(line);
    if (hello.value("type", "") != "hello") {
      emit(fatal_error(std::nullopt, "expected hello"));
      return 2;
    }
    validate_protocol_version(
        hello.at("protocol_version").get<std::uint32_t>());
    surface_channel = SurfaceChannel::from_environment();
    emit(ready_event());
    emit(
        {{"type", "capabilities_changed"},
         // A request is not proof. Promote this only after a decoder has
         // exported a usable DRM PRIME surface for the opened media.
         {"zero_cpu_copy", false},
         {"copy_mode", surface_channel
                           ? (dmabuf_requested ? "dma_buf_pending_validation"
                                               : "shared_memory_yuv420p")
                           : "cpu_rgba_upload"},
         {"interpolation_engines", svp_available
                                       ? nlohmann::json::array({"off", "svp"})
                                       : nlohmann::json::array({"off"})},
         {"svp_status", svp_available ? "available_external" : "unavailable"}});
  } catch (const std::exception &error) {
    emit(fatal_error(std::nullopt, error.what()));
    return 2;
  }

  SdlVideoPresenter presenter;
  std::optional<ShmSurfaceProducer> shm_surface_producer;
  std::optional<DmabufSurfaceProducer> dmabuf_surface_producer;
  std::optional<DmabufFrameDecoder> dmabuf_decoder;
  std::optional<DmabufFrameUploader> svp_dmabuf_uploader;
  std::optional<DmabufVideoFrame> dmabuf_pending_frame;
  FramePacer dmabuf_pacer;
  bool dmabuf_active = false;
  std::optional<std::string> dmabuf_fallback_reason;
  bool dmabuf_capability_confirmed = false;
  bool dmabuf_preview_pending = false;
  bool dmabuf_audio_anchor_pending = false;
  bool dmabuf_ended = false;
  std::atomic<bool> svp_dmabuf_active{false};
  std::string interpolation_engine = "off";
  std::string interpolation_preset = "balanced";
  std::uint32_t interpolation_target_fps = 60;
  std::string current_media_path;
  std::optional<MediaMetadata> current_metadata;
  std::vector<ExternalSubtitleTrack> external_subtitles;
  std::optional<SubtitleTrack> selected_subtitle;
  std::optional<std::string> selected_subtitle_track_id;
  std::atomic<int> selected_audio_stream{-1};
  double subtitle_delay_seconds = 0.0;
  std::vector<std::string> last_subtitle_lines;
  std::optional<std::chrono::steady_clock::time_point> seek_started;
  std::optional<double> seek_target;
  std::optional<double> seek_latency_ms;
  const auto record_seek_latency = [&](double position) {
    if (!seek_started || !seek_target ||
        std::abs(position - *seek_target) > 0.25) {
      return;
    }
    seek_latency_ms = std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - *seek_started)
                          .count();
    seek_started.reset();
    seek_target.reset();
  };
  if (surface_channel && dmabuf_requested) {
    dmabuf_surface_producer.emplace(*surface_channel, drm_node, 3);
  } else if (surface_channel) {
    setenv("LOCALBOORU_NATIVE_VIDEO_HW_DOWNLOAD", "1", 1);
    shm_surface_producer.emplace(*surface_channel);
  }
  std::atomic<std::uint64_t> current_generation{0};
  std::atomic<double> current_duration{0.0};
  std::atomic<double> current_position{0.0};
  std::atomic<double> current_volume{1.0};
  std::atomic<double> current_speed{1.0};
  std::atomic<bool> muted{false};
  std::atomic<bool> first_frame_sent{false};
  std::atomic<bool> paused{false};
  std::atomic<bool> session_ready_sent{false};
  std::atomic<std::uint64_t> produced_frames{0};
  std::atomic<std::uint64_t> presented_frames{0};
  std::atomic<std::uint64_t> dropped_frames{0};
  std::atomic<bool> svp_active_mode{false};
  std::atomic<bool> svp_ready_sent{false};
  std::mutex frame_mutex;
  std::optional<std::pair<std::uint64_t, DecodedVideoFrame>> pending_frame;
  std::atomic<bool> pending_frame_available{false};
  std::mutex command_mutex;
  std::condition_variable command_ready;
  std::deque<std::string> command_lines;
  bool input_closed = false;

  AudioPlaybackSession audio_session;
  const auto wait_for_audio_clock = [&](double position) {
    if (selected_audio_stream.load() < 0 || paused.load())
      return false;
    bool waited = false;
    for (int attempt = 0; attempt < 500; ++attempt) {
      if (paused.load() ||
          audio_session.playback_position() + 0.020 >= position) {
        return waited;
      }
      waited = true;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return waited;
  };
  const auto playback_state_event = [&](double position) {
    return nlohmann::json{
        {"type", "playback_state"},
        {"generation", current_generation.load()},
        {"position", position},
        {"duration", current_duration.load()},
        {"paused", paused.load()},
        {"volume", current_volume.load()},
        {"muted", muted.load()},
        {"speed", current_speed.load()},
        {"selected_audio_track",
         selected_audio_stream.load() >= 0
             ? nlohmann::json(std::to_string(selected_audio_stream.load()))
             : nlohmann::json(nullptr)},
        {"selected_subtitle_track",
         selected_subtitle_track_id
             ? nlohmann::json(*selected_subtitle_track_id)
             : nlohmann::json(nullptr)},
        {"subtitle_delay", subtitle_delay_seconds},
        {"interpolation_engine", interpolation_engine},
        {"interpolation_preset",
         interpolation_engine == "off"
             ? nlohmann::json(nullptr)
             : nlohmann::json(interpolation_preset)},
        {"interpolation_target_fps", interpolation_target_fps}};
  };
  PlaybackCallbacks callbacks;
  callbacks.on_frame = [&](const DecodedVideoFrame &frame) {
    const bool first_session_frame = !session_ready_sent.exchange(true);
    const bool first_svp_frame =
        svp_active_mode.load() && !svp_ready_sent.exchange(true);
    if (first_session_frame) {
      audio_session.seek(frame.pts_seconds);
      audio_session.set_paused(paused.load());
    }
    if (first_svp_frame) {
      emit({{"type", "capabilities_changed"},
            {"zero_cpu_copy", false},
            {"copy_mode", svp_dmabuf_active.load()
                              ? "svp_cpu_to_vaapi_dmabuf"
                              : "svp_yuv420p_shared_memory_to_gpu"},
            {"interpolation_engines", nlohmann::json::array({"off", "svp"})},
            {"svp_status", "active_external"}});
    }
    if (!first_session_frame)
      wait_for_audio_clock(frame.pts_seconds);
    current_position.store(frame.pts_seconds);
    produced_frames.fetch_add(1);
    {
      std::lock_guard lock(frame_mutex);
      if (pending_frame && pending_frame->first == current_generation.load()) {
        dropped_frames.fetch_add(1);
      }
      pending_frame =
          std::make_pair(current_generation.load(), DecodedVideoFrame(frame));
      pending_frame_available.store(true, std::memory_order_release);
    }
    command_ready.notify_all();
    emit(playback_state_event(frame.pts_seconds));
  };
  callbacks.on_ended = [&] {
    paused.store(true);
    emit(playback_state_event(current_duration.load()));
    emit({{"type", "playback_ended"},
          {"generation", current_generation.load()},
          {"position", current_duration.load()}});
  };
  callbacks.on_error = [&](const std::string &message) {
    emit({{"type", "recoverable_error"},
          {"generation", current_generation.load()},
          {"message", message}});
  };
  VideoPlaybackSession session(std::move(callbacks));

  std::thread input_reader([&] {
    std::string command_line;
    while (std::getline(std::cin, command_line)) {
      {
        std::lock_guard lock(command_mutex);
        command_lines.push_back(std::move(command_line));
      }
      command_ready.notify_all();
    }
    {
      std::lock_guard lock(command_mutex);
      input_closed = true;
    }
    command_ready.notify_all();
  });

  bool running = true;
  bool last_hud_visible = true;
  auto diagnostics_started = std::chrono::steady_clock::now();
  std::uint64_t diagnostic_produced = 0;
  std::uint64_t diagnostic_presented = 0;
  while (running) {
    std::optional<std::string> command_line;
    {
      std::unique_lock lock(command_mutex);
      command_ready.wait_for(lock, std::chrono::milliseconds(16), [&] {
        return !command_lines.empty() || input_closed ||
               pending_frame_available.load(std::memory_order_acquire);
      });
      if (!command_lines.empty()) {
        command_line = std::move(command_lines.front());
        command_lines.pop_front();
      } else if (input_closed) {
        running = false;
      }
    }

    if (command_line) {
      try {
        const auto command = nlohmann::json::parse(*command_line);
        const auto type = command.value("type", "");
        if (type == "open_media") {
          const auto generation = generation_of(command).value_or(0);
          const auto path = command.at("path").get<std::string>();
          const auto metadata = probe_media(path);
          current_media_path = path;
          current_metadata = metadata;
          external_subtitles = discover_external_subtitles(path);
          selected_subtitle.reset();
          selected_subtitle_track_id.reset();
          subtitle_delay_seconds = 0.0;
          last_subtitle_lines.clear();
          seek_started.reset();
          seek_target.reset();
          seek_latency_ms.reset();
          const bool autoplay = command.value("autoplay", true);
          const double resume_position = command.value("resume_position", 0.0);
          session.stop();
          audio_session.stop();
          dmabuf_decoder.reset();
          svp_dmabuf_uploader.reset();
          dmabuf_pending_frame.reset();
          dmabuf_pacer.reset();
          dmabuf_active = false;
          dmabuf_fallback_reason.reset();
          dmabuf_capability_confirmed = false;
          dmabuf_preview_pending = false;
          dmabuf_audio_anchor_pending = false;
          dmabuf_ended = false;
          svp_dmabuf_active = false;
          {
            std::lock_guard lock(frame_mutex);
            pending_frame.reset();
          }
          presenter.close();
          current_generation.store(generation);
          current_duration.store(metadata.duration_seconds);
          current_position.store(resume_position);
          first_frame_sent.store(false);
          session_ready_sent.store(false);
          produced_frames.store(0);
          presented_frames.store(0);
          dropped_frames.store(0);
          diagnostic_produced = 0;
          diagnostic_presented = 0;
          diagnostics_started = std::chrono::steady_clock::now();
          paused.store(!autoplay);
          const bool svp_active = interpolation_engine == "svp";
          svp_active_mode.store(svp_active);
          svp_ready_sent.store(false);
          if (svp_active && !svp_available) {
            throw std::runtime_error("external SVP runtime is unavailable");
          }
          if (svp_active && dmabuf_surface_producer &&
              svp_dmabuf_upload_enabled) {
            try {
              dmabuf_surface_producer->configure(generation);
              svp_dmabuf_uploader.emplace(
                  drm_node, metadata.width, metadata.height,
                  metadata.sample_aspect_ratio, metadata.rotation_degrees,
                  metadata.color_space, metadata.color_range,
                  metadata.chroma_location);
              svp_dmabuf_active = true;
              emit({{"type", "capabilities_changed"},
                    {"zero_cpu_copy", false},
                    {"copy_mode", "svp_cpu_to_vaapi_dmabuf"},
                    {"interpolation_engines",
                     nlohmann::json::array({"off", "svp"})},
                    {"svp_status", "selected_external"}});
            } catch (const std::exception &error) {
              svp_dmabuf_uploader.reset();
              dmabuf_surface_producer->reset();
              if (!shm_surface_producer && surface_channel)
                shm_surface_producer.emplace(*surface_channel);
              if (shm_surface_producer) {
                shm_surface_producer->configure(
                    generation, metadata.width, metadata.height, true,
                    metadata.sample_aspect_ratio, metadata.rotation_degrees,
                    metadata.color_space, metadata.color_range,
                    metadata.chroma_location);
              }
              emit({{"type", "recoverable_error"},
                    {"generation", generation},
                    {"message", std::string("SVP DMA-BUF upload unavailable: ") +
                                    error.what()}});
              emit({{"type", "capabilities_changed"},
                    {"zero_cpu_copy", false},
                    {"copy_mode", "svp_yuv420p_shared_memory_to_gpu"}});
            }
          } else if (svp_active) {
            if (!shm_surface_producer && surface_channel)
              shm_surface_producer.emplace(*surface_channel);
            if (!shm_surface_producer) {
              throw std::runtime_error(
                  "SVP shared-memory surface transport is unavailable");
            }
            shm_surface_producer->configure(
                generation, metadata.width, metadata.height, true,
                metadata.sample_aspect_ratio, metadata.rotation_degrees,
                metadata.color_space, metadata.color_range,
                metadata.chroma_location);
            emit({{"type", "capabilities_changed"},
                  {"zero_cpu_copy", false},
                  {"copy_mode", "svp_yuv420p_shared_memory_to_gpu"}});
          } else if (dmabuf_surface_producer) {
            try {
              dmabuf_surface_producer->configure(generation);
              dmabuf_decoder.emplace(path, drm_node);
              if (resume_position > 0.0)
                dmabuf_decoder->seek(resume_position);
              dmabuf_active = true;
              dmabuf_preview_pending = true;
              dmabuf_audio_anchor_pending = true;

            } catch (const std::exception &error) {
              dmabuf_fallback_reason = error.what();
              setenv("LOCALBOORU_NATIVE_VIDEO_HW_DOWNLOAD", "1", 1);
              dmabuf_decoder.reset();
              dmabuf_surface_producer->reset();
              if (!shm_surface_producer && surface_channel) {
                shm_surface_producer.emplace(*surface_channel);
              }
              if (shm_surface_producer) {
                shm_surface_producer->configure(
                    generation, metadata.width, metadata.height, true,
                    metadata.sample_aspect_ratio, metadata.rotation_degrees,
                    metadata.color_space, metadata.color_range,
                    metadata.chroma_location);
              }
              emit({{"type", "recoverable_error"},
                    {"generation", generation},
                    {"message",
                     std::string("DMA-BUF unavailable: ") + error.what()}});
              emit({{"type", "capabilities_changed"},
                    {"zero_cpu_copy", false},
                    {"copy_mode", "shared_memory_yuv420p"}});
            }
          } else if (shm_surface_producer) {
            shm_surface_producer->configure(
                generation, metadata.width, metadata.height, true,
                metadata.sample_aspect_ratio, metadata.rotation_degrees,
                metadata.color_space, metadata.color_range,
                metadata.chroma_location);
          }
          emit({{"type", "media_opened"},
                {"generation", generation},
                {"duration", metadata.duration_seconds},
                {"width", metadata.width},
                {"height", metadata.height},
                {"frame_rate", metadata.frame_rate}});
          nlohmann::json audio_tracks = nlohmann::json::array();
          nlohmann::json subtitle_tracks = nlohmann::json::array();
          for (const auto &track : metadata.tracks) {
            nlohmann::json info = {
                {"id", std::to_string(track.index)},
                {"language", track.language.empty()
                                 ? nlohmann::json(nullptr)
                                 : nlohmann::json(track.language)},
                {"label", track.label},
                {"is_default", false},
                {"is_forced", false},
                {"source_type", "embedded"},
                {"cue_format", nullptr},
                {"cue_path", nullptr},
                {"generation_status", "ready"},
                {"delay_seconds", 0.0},
                {"style", nullptr}};
            if (track.kind == "audio")
              audio_tracks.push_back(info);
            if (track.kind == "subtitle")
              subtitle_tracks.push_back(info);
          }
          for (const auto &track : external_subtitles) {
            subtitle_tracks.push_back(
                {{"id", track.id},
                 {"language", track.language.empty()
                                  ? nlohmann::json(nullptr)
                                  : nlohmann::json(track.language)},
                 {"label", track.label},
                 {"is_default", false},
                 {"is_forced", false},
                 {"source_type", track.source_type},
                 {"cue_format", track.cue_format},
                 {"cue_path", track.path.string()},
                 {"generation_status", "ready"},
                 {"delay_seconds", 0.0},
                 {"style", nullptr}});
          }
          emit({{"type", "track_list"},
                {"generation", generation},
                {"audio", std::move(audio_tracks)},
                {"subtitles", std::move(subtitle_tracks)}});
          if (!dmabuf_active) {
            session.open(path, resume_position,
                         {.autoplay = autoplay,
                          .interpolation_engine = interpolation_engine,
                          .interpolation_preset = interpolation_preset,
                          .target_fps = interpolation_target_fps});
          }
          const bool has_audio = std::any_of(
              metadata.tracks.begin(), metadata.tracks.end(),
              [](const TrackMetadata &track) { return track.kind == "audio"; });
          selected_audio_stream = -1;
          if (has_audio) {
            const auto first_audio =
                std::find_if(metadata.tracks.begin(), metadata.tracks.end(),
                             [](const TrackMetadata &track) {
                               return track.kind == "audio";
                             });
            selected_audio_stream = first_audio->index;
            audio_session.open(path, resume_position, false,
                               selected_audio_stream.load());
          }
          emit(playback_state_event(current_position.load()));
        } else if (type == "close_media") {
          const auto generation = generation_of(command);
          if (!generation || *generation == current_generation.load()) {
            session.stop();
            audio_session.stop();
            dmabuf_decoder.reset();
            svp_dmabuf_uploader.reset();
            dmabuf_pending_frame.reset();
            dmabuf_pacer.reset();
            dmabuf_active = false;
            dmabuf_preview_pending = false;
            dmabuf_ended = false;
            svp_dmabuf_active = false;
            {
              std::lock_guard lock(frame_mutex);
              pending_frame.reset();
            }
            presenter.close();
            if (shm_surface_producer)
              shm_surface_producer->reset();
            if (dmabuf_surface_producer)
              dmabuf_surface_producer->reset();
            current_media_path.clear();
            current_metadata.reset();
            selected_audio_stream = -1;
            external_subtitles.clear();
            selected_subtitle.reset();
            selected_subtitle_track_id.reset();
            last_subtitle_lines.clear();
          }
        } else if (type == "set_paused") {
          const bool next_paused = command.at("paused").get<bool>();
          if (dmabuf_active && paused.load() && !next_paused) {
            dmabuf_pacer.anchor(
                current_position.load(),
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count());
          }
          paused.store(next_paused);
          if (!dmabuf_active)
            session.set_paused(next_paused);
          audio_session.set_paused(next_paused);
          emit(playback_state_event(current_position.load()));
        } else if (type == "seek") {
          const double position = command.at("position").get<double>();
          seek_started = std::chrono::steady_clock::now();
          seek_target = position;
          if (dmabuf_active && dmabuf_decoder) {
            dmabuf_decoder->seek(position);
            dmabuf_pending_frame.reset();
            dmabuf_pacer.reset();
            dmabuf_preview_pending = true;
            dmabuf_ended = false;
          } else {
            session.seek(position);
          }
          audio_session.seek(position);
        } else if (type == "set_volume") {
          const double volume = command.at("volume").get<double>();
          current_volume.store(volume);
          if (!muted.load())
            audio_session.set_volume(volume);
          emit(playback_state_event(current_position.load()));
        } else if (type == "set_muted") {
          const bool next_muted = command.at("muted").get<bool>();
          muted.store(next_muted);
          audio_session.set_volume(next_muted ? 0.0 : current_volume.load());
          emit(playback_state_event(current_position.load()));
        } else if (type == "set_speed") {
          const double speed = command.at("speed").get<double>();
          current_speed.store(std::isfinite(speed) ? std::clamp(speed, 0.5, 2.0)
                                                   : 1.0);
          if (dmabuf_active) {
            dmabuf_pacer.set_speed(
                current_speed.load(), current_position.load(),
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count());
          }
          if (!dmabuf_active)
            session.set_speed(speed);
          audio_session.set_speed(speed);
          emit(playback_state_event(current_position.load()));
        } else if (type == "select_audio_track") {
          if (!current_metadata || current_media_path.empty()) {
            throw std::runtime_error(
                "no active media for audio track selection");
          }
          const auto track_id = command.at("track_id").get<std::string>();
          std::size_t parsed = 0;
          int stream_index = -1;
          try {
            stream_index = std::stoi(track_id, &parsed);
          } catch (...) {
            throw std::runtime_error("selected audio track id is invalid");
          }
          const auto track = std::find_if(
              current_metadata->tracks.begin(), current_metadata->tracks.end(),
              [&](const TrackMetadata &candidate) {
                return candidate.kind == "audio" &&
                       candidate.index == stream_index;
              });
          if (parsed != track_id.size() ||
              track == current_metadata->tracks.end()) {
            throw std::runtime_error("selected audio track does not exist");
          }
          if (stream_index != selected_audio_stream.load()) {
            selected_audio_stream = stream_index;
            audio_session.open(current_media_path, current_position.load(),
                               !paused.load(), selected_audio_stream.load());
            audio_session.set_volume(muted.load() ? 0.0
                                                  : current_volume.load());
          }
          emit(playback_state_event(current_position.load()));
        } else if (type == "register_subtitle_track") {
          const auto generation = command.at("generation").get<std::uint64_t>();
          if (generation != current_generation.load()) {
            throw std::runtime_error("stale subtitle registration generation");
          }
          const auto requested_path = std::filesystem::weakly_canonical(
              command.at("path").get<std::string>());
          const auto media_parent = std::filesystem::weakly_canonical(
              std::filesystem::path(current_media_path).parent_path());
          if (!std::filesystem::is_regular_file(requested_path) ||
              requested_path.parent_path() != media_parent) {
            throw std::runtime_error("generated subtitle path is outside the "
                                     "active media directory");
          }
          external_subtitles = discover_external_subtitles(current_media_path);
          const auto track =
              std::find_if(external_subtitles.begin(), external_subtitles.end(),
                           [&](const auto &candidate) {
                             return std::filesystem::weakly_canonical(
                                        candidate.path) == requested_path;
                           });
          if (track == external_subtitles.end()) {
            throw std::runtime_error(
                "generated subtitle path is not a valid media sidecar");
          }
          const auto track_info = nlohmann::json{
              {"id", track->id},
              {"language", track->language.empty()
                               ? nlohmann::json(nullptr)
                               : nlohmann::json(track->language)},
              {"label", command.value("label", track->label)},
              {"is_default", false},
              {"is_forced", false},
              {"source_type", track->source_type},
              {"cue_format", track->cue_format},
              {"cue_path", track->path.string()},
              {"generation_status", "ready"},
              {"delay_seconds", subtitle_delay_seconds},
              {"style", nullptr}};
          emit({{"type", "subtitle_track_added"},
                {"generation", generation},
                {"track", track_info}});
          if (command.value("select", true)) {
            selected_subtitle =
                track->cue_format == "vtt"
                    ? SubtitleTrack::from_webvtt(
                          read_subtitle_file(track->path))
                    : SubtitleTrack::from_embedded(track->path.string(), 0);
            selected_subtitle_track_id = track->id;
            last_subtitle_lines = selected_subtitle->text_at(
                current_position.load(), subtitle_delay_seconds);
            emit({{"type", "subtitle_text"},
                  {"generation", generation},
                  {"lines", last_subtitle_lines}});
            emit(playback_state_event(current_position.load()));
          }
        } else if (type == "select_subtitle_track") {
          selected_subtitle.reset();
          selected_subtitle_track_id.reset();
          last_subtitle_lines.clear();
          if (command.contains("track_id") &&
              !command.at("track_id").is_null() &&
              command.at("track_id").get<std::string>() != "__off__") {
            const auto track_id = command.at("track_id").get<std::string>();
            selected_subtitle_track_id = track_id;
            const auto track = std::find_if(external_subtitles.begin(),
                                            external_subtitles.end(),
                                            [&](const auto &candidate) {
                                              return candidate.id == track_id;
                                            });
            if (track == external_subtitles.end()) {
              std::size_t parsed = 0;
              int stream_index = -1;
              try {
                stream_index = std::stoi(track_id, &parsed);
              } catch (const std::exception &) {
                throw std::runtime_error("unknown subtitle track");
              }
              if (parsed != track_id.size()) {
                throw std::runtime_error("unknown subtitle track");
              }
              selected_subtitle = SubtitleTrack::from_embedded(
                  current_media_path, stream_index);
            } else if (track->cue_format == "vtt") {
              selected_subtitle =
                  SubtitleTrack::from_webvtt(read_subtitle_file(track->path));
            } else {
              selected_subtitle =
                  SubtitleTrack::from_embedded(track->path.string(), 0);
            }
          }
          last_subtitle_lines =
              selected_subtitle
                  ? selected_subtitle->text_at(current_position.load(),
                                               subtitle_delay_seconds)
                  : std::vector<std::string>{};
          emit({{"type", "subtitle_text"},
                {"generation", current_generation.load()},
                {"lines", last_subtitle_lines}});
          emit(playback_state_event(current_position.load()));
        } else if (type == "set_subtitle_delay") {
          subtitle_delay_seconds =
              std::clamp(command.at("seconds").get<double>(), -30.0, 30.0);
          last_subtitle_lines =
              selected_subtitle
                  ? selected_subtitle->text_at(current_position.load(),
                                               subtitle_delay_seconds)
                  : std::vector<std::string>{};
          emit({{"type", "subtitle_text"},
                {"generation", current_generation.load()},
                {"lines", last_subtitle_lines}});
          emit(playback_state_event(current_position.load()));
        } else if (type == "set_interpolation") {
          const auto engine = command.at("engine").get<std::string>();
          const auto preset =
              command.contains("preset") && !command.at("preset").is_null()
                  ? command.at("preset").get<std::string>()
                  : "balanced";
          const auto target_fps = command.value("target_fps", 60U);
          if (engine != "off" && engine != "svp") {
            throw std::runtime_error("unsupported interpolation engine: " +
                                     engine);
          }
          if (engine == "svp" && !svp_available) {
            throw std::runtime_error("external SVP runtime is unavailable");
          }
          if (engine == "svp" && preset != "balanced") {
            throw std::runtime_error("only the balanced SVP preset is "
                                     "implemented by the native adapter");
          }
          if (target_fps < 1 || target_fps > 240) {
            throw std::runtime_error(
                "interpolation target FPS must be between 1 and 240");
          }
          const bool interpolation_changed =
              engine != interpolation_engine ||
              preset != interpolation_preset ||
              target_fps != interpolation_target_fps;
          interpolation_engine = engine;
          interpolation_preset = preset;
          interpolation_target_fps = target_fps;
          svp_active_mode.store(engine == "svp");
          svp_ready_sent.store(false);
          emit({{"type", "capabilities_changed"},
                {"zero_cpu_copy", dmabuf_active && engine == "off"},
                {"copy_mode", engine == "svp"
                                  ? "svp_output_pending_validation"
                                  : (dmabuf_active ? "dma_buf_external_oes"
                                                   : "shared_memory_yuv420p")},
                {"interpolation_engines",
                 svp_available ? nlohmann::json::array({"off", "svp"})
                               : nlohmann::json::array({"off"})},
                {"svp_status", engine == "svp"
                                   ? "selected_external"
                                   : (svp_available ? "available_external"
                                                    : "unavailable")}});
          emit(playback_state_event(current_position.load()));
          if (interpolation_changed && !current_media_path.empty() &&
              current_metadata) {
            const double position = current_position.load();
            const bool autoplay = !paused.load();
            if (engine == "svp")
              audio_session.set_paused(true);
            session.stop();
            dmabuf_decoder.reset();
            svp_dmabuf_uploader.reset();
            dmabuf_pending_frame.reset();
            dmabuf_pacer.reset();
            dmabuf_active = false;
            dmabuf_fallback_reason.reset();
            dmabuf_capability_confirmed = false;
            dmabuf_preview_pending = false;
            dmabuf_ended = false;
            svp_dmabuf_active = false;
            if (dmabuf_surface_producer)
              dmabuf_surface_producer->reset();
            {
              std::lock_guard lock(frame_mutex);
              pending_frame.reset();
            }
            if (engine == "off" && dmabuf_surface_producer) {
              try {
                dmabuf_surface_producer->configure(current_generation.load());
                dmabuf_decoder.emplace(current_media_path, drm_node);
                if (position > 0.0)
                  dmabuf_decoder->seek(position);
                dmabuf_active = true;
                dmabuf_fallback_reason.reset();
                dmabuf_preview_pending = true;
                dmabuf_audio_anchor_pending = true;
              } catch (const std::exception &error) {
                dmabuf_fallback_reason = error.what();
                setenv("LOCALBOORU_NATIVE_VIDEO_HW_DOWNLOAD", "1", 1);
                dmabuf_decoder.reset();
                dmabuf_surface_producer->reset();
                emit({{"type", "recoverable_error"},
                      {"generation", current_generation.load()},
                      {"message",
                       std::string("DMA-BUF unavailable after SVP: ") +
                           error.what()}});
              }
            } else if (engine == "svp" && dmabuf_surface_producer &&
                       svp_dmabuf_upload_enabled) {
              try {
                dmabuf_surface_producer->configure(current_generation.load());
                svp_dmabuf_uploader.emplace(
                    drm_node, current_metadata->width, current_metadata->height,
                    current_metadata->sample_aspect_ratio,
                    current_metadata->rotation_degrees,
                    current_metadata->color_space, current_metadata->color_range,
                    current_metadata->chroma_location);
                svp_dmabuf_active = true;
                emit({{"type", "capabilities_changed"},
                      {"zero_cpu_copy", false},
                      {"copy_mode", "svp_cpu_to_vaapi_dmabuf"},
                      {"interpolation_engines",
                       nlohmann::json::array({"off", "svp"})},
                      {"svp_status", "selected_external"}});
              } catch (const std::exception &error) {
                svp_dmabuf_uploader.reset();
                dmabuf_surface_producer->reset();
                emit({{"type", "recoverable_error"},
                      {"generation", current_generation.load()},
                      {"message",
                       std::string("SVP DMA-BUF upload unavailable: ") +
                           error.what()}});
              }
            }
            if (!dmabuf_active) {
              session_ready_sent.store(false);
              audio_session.set_paused(true);
              if (!svp_dmabuf_active && !shm_surface_producer &&
                  surface_channel) {
                shm_surface_producer.emplace(*surface_channel);
              }
              if (!svp_dmabuf_active && shm_surface_producer) {
                shm_surface_producer->configure(
                    current_generation.load(), current_metadata->width,
                    current_metadata->height, true,
                    current_metadata->sample_aspect_ratio,
                    current_metadata->rotation_degrees,
                    current_metadata->color_space,
                    current_metadata->color_range,
                    current_metadata->chroma_location);
              }
              session.open(current_media_path, position,
                           {.autoplay = autoplay,
                            .interpolation_engine = interpolation_engine,
                            .interpolation_preset = interpolation_preset,
                            .target_fps = interpolation_target_fps});
            }
            audio_session.seek(position);
            if (dmabuf_active)
              audio_session.set_paused(!autoplay);
          }
        } else if (type == "set_fullscreen") {
          presenter.set_fullscreen(command.at("fullscreen").get<bool>());
        }
      } catch (const std::exception &error) {
        emit({{"type", "recoverable_error"},
              {"generation", current_generation.load()},
              {"message", error.what()}});
      }
    }

    if (shm_surface_producer) {
      presented_frames.fetch_add(shm_surface_producer->drain_releases());
    }
    if (dmabuf_surface_producer) {
      presented_frames.fetch_add(dmabuf_surface_producer->drain_releases());
    }
    try {
      if (dmabuf_active && dmabuf_decoder && dmabuf_surface_producer &&
          !dmabuf_ended && (!paused.load() || dmabuf_preview_pending) &&
          dmabuf_surface_producer->available() > 0) {
        if (!dmabuf_pending_frame) {
          dmabuf_pending_frame = dmabuf_decoder->next_frame();
        }
        if (dmabuf_pending_frame) {
          const double position = dmabuf_pending_frame->pts_seconds();
          const double wall_seconds =
              std::chrono::duration<double>(
                  std::chrono::steady_clock::now().time_since_epoch())
                  .count();
          if (!dmabuf_preview_pending &&
              !dmabuf_pacer.due(position, wall_seconds)) {
            continue;
          }
          if (dmabuf_preview_pending) {
            dmabuf_pacer.set_speed(current_speed.load(), position,
                                   wall_seconds);
          }
          if (!dmabuf_audio_anchor_pending && wait_for_audio_clock(position)) {
            dmabuf_pacer.set_speed(
                current_speed.load(), position,
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now().time_since_epoch())
                    .count());
          }
          current_position.store(position);
          produced_frames.fetch_add(1);
          if (dmabuf_surface_producer->publish(
                  std::move(*dmabuf_pending_frame))) {
            if (dmabuf_audio_anchor_pending) {
              audio_session.seek(position);
              audio_session.set_paused(paused.load());
              dmabuf_audio_anchor_pending = false;
            }
            if (!dmabuf_capability_confirmed) {
              dmabuf_capability_confirmed = true;
              emit({{"type", "capabilities_changed"},
                    {"zero_cpu_copy", true},
                    {"copy_mode", "dma_buf_external_oes"},
                    {"interpolation_engines",
                     svp_available ? nlohmann::json::array({"off", "svp"})
                                   : nlohmann::json::array({"off"})},
                    {"svp_status",
                     svp_available ? "available_external" : "unavailable"}});
            }
            record_seek_latency(position);
            dmabuf_preview_pending = false;
            emit(playback_state_event(position));
          } else {
            dropped_frames.fetch_add(1);
          }
          dmabuf_pending_frame.reset();
        } else {
          dmabuf_ended = true;
          paused.store(true);
          emit(playback_state_event(current_position.load()));
          emit({{"type", "playback_ended"},
                {"generation", current_generation.load()},
                {"position", current_position.load()}});
        }
      }
    } catch (const std::exception &error) {
      dmabuf_fallback_reason = error.what();
      dmabuf_decoder.reset();
      dmabuf_pending_frame.reset();
      dmabuf_pacer.reset();
      dmabuf_active = false;
      dmabuf_capability_confirmed = false;
      dmabuf_preview_pending = false;
      dmabuf_ended = false;
      setenv("LOCALBOORU_NATIVE_VIDEO_HW_DOWNLOAD", "1", 1);
      if (dmabuf_surface_producer)
        dmabuf_surface_producer->reset();
      emit({{"type", "recoverable_error"},
            {"generation", current_generation.load()},
            {"message",
             std::string("DMA-BUF playback failed: ") + error.what()}});
      emit({{"type", "capabilities_changed"},
            {"zero_cpu_copy", false},
            {"copy_mode", "shared_memory_yuv420p"},
            {"interpolation_engines",
             svp_available ? nlohmann::json::array({"off", "svp"})
                           : nlohmann::json::array({"off"})},
            {"svp_status",
             svp_available ? "available_external" : "unavailable"}});
      if (!shm_surface_producer && surface_channel) {
        shm_surface_producer.emplace(*surface_channel);
      }
      if (current_metadata && shm_surface_producer) {
        shm_surface_producer->configure(
            current_generation.load(), current_metadata->width,
            current_metadata->height, true,
            current_metadata->sample_aspect_ratio,
            current_metadata->rotation_degrees, current_metadata->color_space,
            current_metadata->color_range, current_metadata->chroma_location);
      }
      if (!current_media_path.empty()) {
        session_ready_sent.store(false);
        audio_session.set_paused(true);
        session.open(current_media_path, current_position.load(),
                     {.autoplay = !paused.load(),
                      .interpolation_engine = "off",
                      .interpolation_preset = interpolation_preset,
                      .target_fps = interpolation_target_fps});
      }
    }
    std::optional<std::pair<std::uint64_t, DecodedVideoFrame>> frame;
    {
      std::lock_guard lock(frame_mutex);
      if (pending_frame) {
        frame = std::move(pending_frame);
        pending_frame.reset();
        pending_frame_available.store(false, std::memory_order_release);
      }
    }
    if (frame && frame->first == current_generation.load()) {
      record_seek_latency(frame->second.pts_seconds);
      if (svp_dmabuf_active && svp_dmabuf_uploader &&
          dmabuf_surface_producer) {
        try {
          const auto buffer_id = dmabuf_surface_producer->acquire_buffer();
          if (!buffer_id) {
            dropped_frames.fetch_add(1);
          } else {
            try {
              auto uploaded =
                  svp_dmabuf_uploader->upload(frame->second, *buffer_id);
              if (!dmabuf_surface_producer->publish(
                      *buffer_id, std::move(uploaded), true)) {
                dropped_frames.fetch_add(1);
              }
            } catch (...) {
              static_cast<void>(
                  dmabuf_surface_producer->cancel_buffer(*buffer_id));
              throw;
            }
          }
        } catch (const std::exception &error) {
          svp_dmabuf_active = false;
          svp_dmabuf_uploader.reset();
          dmabuf_surface_producer->reset();
          if (!shm_surface_producer && surface_channel)
            shm_surface_producer.emplace(*surface_channel);
          if (shm_surface_producer && current_metadata) {
            shm_surface_producer->configure(
                current_generation.load(), current_metadata->width,
                current_metadata->height, true,
                current_metadata->sample_aspect_ratio,
                current_metadata->rotation_degrees,
                current_metadata->color_space, current_metadata->color_range,
                current_metadata->chroma_location);
            if (!shm_surface_producer->publish(frame->second))
              dropped_frames.fetch_add(1);
          }
          emit({{"type", "recoverable_error"},
                {"generation", current_generation.load()},
                {"message", std::string("SVP DMA-BUF upload failed: ") +
                                error.what()}});
          emit({{"type", "capabilities_changed"},
                {"zero_cpu_copy", false},
                {"copy_mode", "svp_yuv420p_shared_memory_to_gpu"},
                {"interpolation_engines",
                 nlohmann::json::array({"off", "svp"})},
                {"svp_status", "active_external"}});
        }
      } else if (shm_surface_producer) {
        if (!shm_surface_producer->publish(frame->second)) {
          dropped_frames.fetch_add(1);
        }
      } else {
        presenter.set_playback_state(current_position.load(),
                                     current_duration.load(), paused.load());
        presenter.show(frame->second);
        presented_frames.fetch_add(1);
        if (!first_frame_sent.exchange(true)) {
          emit({{"type", "first_frame_ready"},
                {"generation", current_generation.load()}});
        }
      }
    }

    if (selected_subtitle) {
      auto lines = selected_subtitle->text_at(current_position.load(),
                                              subtitle_delay_seconds);
      if (lines != last_subtitle_lines) {
        last_subtitle_lines = std::move(lines);
        emit({{"type", "subtitle_text"},
              {"generation", current_generation.load()},
              {"lines", last_subtitle_lines}});
      }
    }

    for (const auto &hit : presenter.poll_actions()) {
      switch (hit.action) {
      case HudAction::Previous:
        emit({{"type", "navigate_previous"},
              {"generation", current_generation.load()}});
        break;
      case HudAction::TogglePlay: {
        const bool next_paused = !paused.load();
        paused.store(next_paused);
        session.set_paused(next_paused);
        audio_session.set_paused(next_paused);
        break;
      }
      case HudAction::Next:
        emit({{"type", "navigate_next"},
              {"generation", current_generation.load()}});
        break;
      case HudAction::Seek: {
        const double position = hit.normalized_value * current_duration.load();
        session.seek(position);
        audio_session.seek(position);
        break;
      }
      case HudAction::Close:
        emit({{"type", "close_requested"},
              {"generation", current_generation.load()}});
        break;
      case HudAction::SetVolume:
        current_volume.store(hit.normalized_value);
        muted.store(false);
        audio_session.set_volume(hit.normalized_value);
        break;
      case HudAction::ToggleFullscreen:
        presenter.set_fullscreen(!presenter.is_fullscreen());
        break;
      }
    }
    if (presenter.is_open() && presenter.hud_visible() != last_hud_visible) {
      last_hud_visible = presenter.hud_visible();
      emit({{"type", "hud_visibility_changed"},
            {"generation", current_generation.load()},
            {"visible", last_hud_visible}});
    }
    const auto diagnostics_now = std::chrono::steady_clock::now();
    const double diagnostics_seconds =
        std::chrono::duration<double>(diagnostics_now - diagnostics_started)
            .count();
    if ((presenter.is_open() || shm_surface_producer || dmabuf_active ||
         svp_dmabuf_active.load()) &&
        diagnostics_seconds >= 1.0) {
      const auto produced = produced_frames.load();
      const auto presented = presented_frames.load();
      std::uint64_t queue_depth = 0;
      {
        std::lock_guard lock(frame_mutex);
        queue_depth = pending_frame ? 1 : 0;
      }
      const auto surface_mode =
          interpolation_engine == "svp"
              ? (svp_dmabuf_active.load()
                     ? "svp_cpu_to_vaapi_dmabuf"
                     : "svp_yuv420p_shared_memory_to_gpu")
              : (dmabuf_active ? "dma_buf_external_oes"
                               : (shm_surface_producer ? "shared_memory_yuv420p"
                                                       : "cpu_rgba_upload"));
      emit(
          {{"type", "diagnostics"},
           {"generation", current_generation.load()},
           {"produced_fps",
            static_cast<double>(produced - diagnostic_produced) /
                diagnostics_seconds},
           {"presented_fps",
            static_cast<double>(presented - diagnostic_presented) /
                diagnostics_seconds},
           {"dropped_frames", dropped_frames.load()},
           {"zero_cpu_copy", dmabuf_active && interpolation_engine == "off"},
           {"fallback_reason", dmabuf_fallback_reason
                                   ? nlohmann::json(*dmabuf_fallback_reason)
                                   : nlohmann::json(nullptr)},
           {"decoder",
            interpolation_engine == "svp" ? "vapoursynth_svp" : "ffmpeg"},
           {"hardware_device", dmabuf_active || svp_dmabuf_active.load()
                                   ? nlohmann::json("vaapi_drm_prime")
                                   : nlohmann::json(nullptr)},
           {"source_fps",
            current_metadata ? current_metadata->frame_rate : 0.0},
           {"queue_depth", queue_depth},
           {"av_drift_ms",
            selected_audio_stream.load() >= 0
                ? nlohmann::json((audio_session.playback_position() -
                                  current_position.load()) *
                                 1000.0)
                : nlohmann::json(nullptr)},
           {"interpolation_engine", interpolation_engine},
           {"surface_mode", surface_mode},
           {"width", current_metadata ? current_metadata->width : 0},
           {"height", current_metadata ? current_metadata->height : 0},
           {"first_frame_latency_ms", nullptr},
           {"seek_latency_ms", seek_latency_ms
                                   ? nlohmann::json(*seek_latency_ms)
                                   : nlohmann::json(nullptr)}});
      diagnostic_produced = produced;
      diagnostic_presented = presented;
      diagnostics_started = diagnostics_now;
    }
  }
  if (input_reader.joinable())
    input_reader.join();
  session.stop();
  audio_session.stop();
  return 0;
}
