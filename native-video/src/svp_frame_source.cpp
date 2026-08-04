// SPDX-License-Identifier: MIT
#include "svp_frame_source.h"

#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace localbooru::native_video {
namespace {

std::string python_string(const std::string &value) {
  std::string escaped = "'";
  for (const char ch : value) {
    if (ch == '\\' || ch == '\'')
      escaped.push_back('\\');
    escaped.push_back(ch);
  }
  escaped.push_back('\'');
  return escaped;
}

std::string discover_plugin_path(const std::string &configured) {
  std::vector<std::filesystem::path> candidates;
  if (!configured.empty())
    candidates.emplace_back(configured);
  if (const char *env = std::getenv("LOCALBOORU_SVP_PLUGIN_PATH")) {
    candidates.emplace_back(env);
  }
  candidates.emplace_back("/opt/svp/plugins");
  candidates.emplace_back("/usr/lib/svp/plugins");
  for (const auto &candidate : candidates) {
    if (std::filesystem::is_regular_file(candidate / "libsvpflow1.so") &&
        std::filesystem::is_regular_file(candidate / "libsvpflow2.so")) {
      return candidate.string();
    }
  }
  throw std::runtime_error("SVPflow plugins are unavailable");
}

bool read_exact(int fd, std::uint8_t *data, std::size_t size) {
  std::size_t offset = 0;
  while (offset < size) {
    pollfd ready{fd, POLLIN, 0};
    const int poll_result = ::poll(&ready, 1, 100);
    if (poll_result < 0 && errno == EINTR)
      continue;
    if (poll_result <= 0)
      continue;
    if ((ready.revents & POLLIN) == 0)
      return false;
    const ssize_t count = ::read(fd, data + offset, size - offset);
    if (count > 0) {
      offset += static_cast<std::size_t>(count);
      continue;
    }
    if (count < 0 && errno == EINTR)
      continue;
    return false;
  }
  return true;
}

std::optional<std::string> read_line(int fd) {
  std::string line;
  char ch = 0;
  while (line.size() < 4096) {
    pollfd ready{fd, POLLIN, 0};
    const int poll_result = ::poll(&ready, 1, 100);
    if (poll_result < 0 && errno == EINTR)
      continue;
    if (poll_result <= 0)
      continue;
    if ((ready.revents & POLLIN) == 0)
      return std::nullopt;
    const ssize_t count = ::read(fd, &ch, 1);
    if (count == 1) {
      if (ch == '\n')
        return line;
      line.push_back(ch);
      continue;
    }
    if (count < 0 && errno == EINTR)
      continue;
    return std::nullopt;
  }
  throw std::runtime_error("SVP Y4M header exceeds 4096 bytes");
}

void terminate_process(pid_t pid) {
  if (pid <= 0)
    return;
  ::kill(pid, SIGTERM);
  for (int attempt = 0; attempt < 20; ++attempt) {
    int status = 0;
    const pid_t result = ::waitpid(pid, &status, WNOHANG);
    if (result == pid || (result < 0 && errno == ECHILD))
      return;
    std::this_thread::sleep_for(std::chrono::milliseconds(25));
  }
  ::kill(pid, SIGKILL);
  while (::waitpid(pid, nullptr, 0) < 0 && errno == EINTR) {
  }
}

std::string build_script(const std::string &fifo_path,
                         const std::string &plugin_path, int width, int height,
                         int fps_num, int fps_den, std::uint64_t num_frames,
                         std::uint32_t target_fps) {
  const std::size_t frame_size = static_cast<std::size_t>(width) *
                                 static_cast<std::size_t>(height) * 3 / 2;
  std::ostringstream script;
  script
      << "import ctypes\n"
      << "import threading\n"
      << "import vapoursynth as vs\n"
      << "core = vs.core\n"
      << "WIDTH=" << width << "\nHEIGHT=" << height << "\n"
      << "FPS_NUM=" << fps_num << "\nFPS_DEN=" << fps_den << "\n"
      << "NUM_FRAMES=" << num_frames << "\nFRAME_SIZE=" << frame_size << "\n"
      << "stdin=open(" << python_string(fifo_path) << ", 'rb', buffering=0)\n"
      << "core.std.LoadPlugin("
      << python_string(plugin_path + "/libsvpflow1.so") << ")\n"
      << "core.std.LoadPlugin("
      << python_string(plugin_path + "/libsvpflow2.so") << ")\n"
      << "clip=core.std.BlankClip(width=WIDTH,height=HEIGHT,format=vs.YUV420P8,"
         "length=max(NUM_FRAMES,1),fpsnum=FPS_NUM,fpsden=FPS_DEN)\n"
      << "next_frame=0\nlast_frame=None\nframe_cache={}\n"
      << "source_lock=threading.Lock()\n"
      << "def read_exact(size):\n"
      << " data=bytearray()\n"
      << " while len(data)<size:\n"
      << "  chunk=stdin.read(size-len(data))\n"
      << "  if not chunk: break\n"
      << "  data.extend(chunk)\n"
      << " return data\n"
      << "def write_plane(frame,plane,raw,offset,width,height):\n"
      << " stride=frame.get_stride(plane)\n ptr=frame.get_write_ptr(plane)\n"
      << " src=ctypes.addressof(ctypes.c_char.from_buffer(raw,offset))\n"
      << " if stride==width:\n  ctypes.memmove(ptr.value,src,width*height)\n"
      << " else:\n"
      << "  for y in range(height): "
         "ctypes.memmove(ptr.value+y*stride,src+y*width,width)\n"
      << "def source_frame_locked(n,f):\n"
      << " global next_frame,last_frame\n"
      << " if n in frame_cache: return frame_cache[n]\n"
      << " while next_frame<=n:\n"
      << "  raw=read_exact(FRAME_SIZE)\n"
      << "  if len(raw)<FRAME_SIZE:\n"
      << "   if last_frame is not None: return last_frame\n"
      << "   raw.extend(bytes(FRAME_SIZE-len(raw)))\n"
      << "  out=f.copy()\n  ysize=WIDTH*HEIGHT\n  "
         "uvsize=(WIDTH//2)*(HEIGHT//2)\n"
      << "  write_plane(out,0,raw,0,WIDTH,HEIGHT)\n"
      << "  write_plane(out,1,raw,ysize,WIDTH//2,HEIGHT//2)\n"
      << "  write_plane(out,2,raw,ysize+uvsize,WIDTH//2,HEIGHT//"
         "2)\n"
      << "  frame_cache[next_frame]=out\n  last_frame=out\n  next_frame+=1\n"
      << " for old in list(frame_cache):\n"
      << "  if old<n-32: del frame_cache[old]\n"
      << " return frame_cache.get(n,last_frame)\n"
      << "def source_frame(n,f):\n"
      << " with source_lock: return source_frame_locked(n,f)\n"
      << "clip=core.std.ModifyFrame(clip,clip,source_frame)\n"
      << "vector_clip=clip.resize.Bicubic(width=WIDTH//4,height=HEIGHT//4,"
         "format=vs.YUV420P8)\n"
      << "smooth=core.svp2.SmoothFps_NVOF(clip,"
      << "'{rate:{num:" << target_fps
      << ",den:1,abs:true},gpuid:0,algo:23,mask:{area:100},scene:{}}',"
         "vec_src=vector_clip,src=clip,"
         "fps=clip.fps.numerator/clip.fps.denominator)\n"
      << "smooth.set_output()\n";
  return script.str();
}

} // namespace

struct SvpFrameSource::Impl {
  std::string media_path;
  MediaMetadata metadata;
  SvpOptions options;
  std::atomic<int> output_fd{-1};
  int vspipe_stderr_fd = -1;
  int ffmpeg_stderr_fd = -1;
  pid_t vspipe_pid = -1;
  pid_t ffmpeg_pid = -1;
  std::thread vspipe_stderr_thread;
  std::thread ffmpeg_stderr_thread;
  std::filesystem::path temp_dir;
  std::filesystem::path fifo_path;
  std::filesystem::path script_path;

  double epoch_position = 0.0;
  std::uint64_t output_index = 0;
  bool header_read = false;
  int output_width = 0;
  int output_height = 0;
  int output_fps_num = 0;
  int output_fps_den = 1;
  mutable std::mutex diagnostics_mutex;
  SvpDiagnostics diagnostics;

  Impl(std::string path, MediaMetadata value, SvpOptions opts)
      : media_path(std::move(path)), metadata(std::move(value)),
        options(std::move(opts)) {}

  ~Impl() { stop(); }

  void drain_stderr(int fd, const char *source) {
    std::string captured;
    char buffer[1024];
    for (;;) {
      const ssize_t count = ::read(fd, buffer, sizeof(buffer));
      if (count > 0) {
        if (captured.size() < 8192) {
          captured.append(buffer, static_cast<std::size_t>(count));
          if (captured.size() > 8192)
            captured.resize(8192);
        }
        continue;
      }
      if (count < 0 && errno == EINTR)
        continue;
      break;
    }
    ::close(fd);
    if (!captured.empty()) {
      std::lock_guard lock(diagnostics_mutex);
      diagnostics.last_error = std::string(source) + ": " + captured;
    }
  }

  void start(double position_seconds) {
    stop();
    if (metadata.width <= 0 || metadata.height <= 0 ||
        metadata.frame_rate <= 0.0) {
      throw std::runtime_error(
          "SVP requires valid media dimensions and frame rate");
    }
    if (options.preset != "balanced") {
      throw std::runtime_error("unsupported native SVP preset: " +
                               options.preset);
    }
    if (options.target_fps < 1 || options.target_fps > 240) {
      throw std::runtime_error("SVP target FPS must be between 1 and 240");
    }
    const std::string plugin_path = discover_plugin_path(options.plugin_path);
    char directory_template[] = "/tmp/localbooru-svp-XXXXXX";
    const char *created = ::mkdtemp(directory_template);
    if (!created)
      throw std::runtime_error("failed to create SVP runtime directory");
    temp_dir = created;
    fifo_path = temp_dir / "frames.yuv";
    script_path = temp_dir / "pipeline.vpy";
    if (::mkfifo(fifo_path.c_str(), 0600) != 0) {
      throw std::runtime_error("failed to create SVP frame pipe");
    }
    const int fps_den = metadata.frame_rate_denominator > 0
                            ? metadata.frame_rate_denominator
                            : 1000;
    const int fps_num = metadata.frame_rate_numerator > 0
                            ? metadata.frame_rate_numerator
                            : std::max(1, static_cast<int>(std::llround(
                                              metadata.frame_rate * fps_den)));
    const auto frames =
        static_cast<std::uint64_t>(std::ceil(
            std::max(0.0, metadata.duration_seconds - position_seconds) *
            metadata.frame_rate)) +
        2;
    std::ofstream script_file(script_path);
    script_file << build_script(fifo_path.string(), plugin_path, metadata.width,
                                metadata.height, fps_num, fps_den, frames,
                                options.target_fps);
    script_file.close();
    if (!script_file)
      throw std::runtime_error("failed to write SVP pipeline script");

    int output_pipe[2];
    int vspipe_error_pipe[2];
    int ffmpeg_error_pipe[2];
    if (::pipe2(output_pipe, O_CLOEXEC) != 0 ||
        ::pipe2(vspipe_error_pipe, O_CLOEXEC) != 0 ||
        ::pipe2(ffmpeg_error_pipe, O_CLOEXEC) != 0) {
      throw std::runtime_error("failed to create SVP process pipes");
    }

    vspipe_pid = ::fork();
    if (vspipe_pid == 0) {
      ::dup2(output_pipe[1], STDOUT_FILENO);
      ::dup2(vspipe_error_pipe[1], STDERR_FILENO);
      ::close(output_pipe[0]);
      ::close(output_pipe[1]);
      ::close(vspipe_error_pipe[0]);
      ::close(vspipe_error_pipe[1]);
      ::execlp("vspipe", "vspipe", "--requests", "8", "-c", "y4m",
               script_path.c_str(), "-", static_cast<char *>(nullptr));
      _exit(127);
    }
    if (vspipe_pid < 0)
      throw std::runtime_error("failed to start vspipe");
    ::close(output_pipe[1]);
    ::close(vspipe_error_pipe[1]);
    output_fd.store(output_pipe[0]);
    vspipe_stderr_fd = vspipe_error_pipe[0];

    ffmpeg_pid = ::fork();
    if (ffmpeg_pid == 0) {
      const int fifo = ::open(fifo_path.c_str(), O_WRONLY);
      if (fifo < 0)
        _exit(126);
      ::dup2(fifo, STDOUT_FILENO);
      ::dup2(ffmpeg_error_pipe[1], STDERR_FILENO);
      ::close(fifo);
      ::close(ffmpeg_error_pipe[0]);
      ::close(ffmpeg_error_pipe[1]);
      const std::string position =
          std::to_string(std::max(0.0, position_seconds));
      ::execlp("ffmpeg", "ffmpeg", "-nostdin", "-hide_banner", "-loglevel",
               "warning", "-ss", position.c_str(), "-i", media_path.c_str(),
               "-map", "0:v:0", "-an", "-sn", "-f", "rawvideo", "-pix_fmt",
               "yuv420p", "pipe:1", static_cast<char *>(nullptr));
      _exit(127);
    }
    ::close(ffmpeg_error_pipe[1]);
    if (ffmpeg_pid < 0) {
      stop();
      throw std::runtime_error("failed to start SVP FFmpeg decoder");
    }
    ffmpeg_stderr_fd = ffmpeg_error_pipe[0];
    vspipe_stderr_thread = std::thread(
        [this, fd = vspipe_stderr_fd] { drain_stderr(fd, "vspipe"); });
    ffmpeg_stderr_thread = std::thread(
        [this, fd = ffmpeg_stderr_fd] { drain_stderr(fd, "ffmpeg"); });
    epoch_position = std::max(0.0, position_seconds);
    output_index = 0;
    header_read = false;
    {
      std::lock_guard lock(diagnostics_mutex);
      diagnostics.restarts += 1;
      diagnostics.last_error.clear();
    }
  }

  void parse_header(int fd) {
    const auto line = read_line(fd);
    if (!line || line->rfind("YUV4MPEG2", 0) != 0) {
      throw std::runtime_error("SVP did not produce a valid Y4M stream header");
    }
    std::istringstream fields(*line);
    std::string field;
    fields >> field;
    while (fields >> field) {
      if (field.size() < 2)
        continue;
      if (field[0] == 'W')
        output_width = std::stoi(field.substr(1));
      if (field[0] == 'H')
        output_height = std::stoi(field.substr(1));
      if (field[0] == 'F') {
        const auto separator = field.find(':');
        if (separator != std::string::npos) {
          output_fps_num = std::stoi(field.substr(1, separator - 1));
          output_fps_den = std::stoi(field.substr(separator + 1));
        }
      }
      if (field[0] == 'C' && field.rfind("C420", 0) != 0) {
        throw std::runtime_error("SVP output is not 8-bit YUV420");
      }
    }
    if (output_width <= 0 || output_height <= 0 || output_fps_num <= 0 ||
        output_fps_den <= 0) {
      throw std::runtime_error(
          "SVP Y4M stream has invalid geometry or frame rate");
    }
    header_read = true;
  }

  std::optional<DecodedVideoFrame> next_frame() {
    const int fd = output_fd.load();
    if (fd < 0)
      return std::nullopt;
    if (!header_read)
      parse_header(fd);
    const auto frame_header = read_line(fd);
    if (!frame_header)
      return std::nullopt;
    if (frame_header->rfind("FRAME", 0) != 0) {
      throw std::runtime_error("SVP Y4M frame boundary is invalid");
    }
    const std::size_t y_size =
        static_cast<std::size_t>(output_width) * output_height;
    const std::size_t uv_size = y_size / 4;
    std::vector<std::uint8_t> yuv(y_size + 2 * uv_size);
    if (!read_exact(fd, yuv.data(), yuv.size()))
      return std::nullopt;

    DecodedVideoFrame frame;
    frame.width = output_width;
    frame.height = output_height;
    frame.pts_seconds =
        epoch_position + static_cast<double>(output_index) * output_fps_den /
                             static_cast<double>(output_fps_num);
    frame.yuv420p = true;
    frame.rgba = std::move(yuv);
    ++output_index;
    {
      std::lock_guard lock(diagnostics_mutex);
      diagnostics.frames_read += 1;
    }
    return frame;
  }

  void interrupt() {
    const int fd = output_fd.exchange(-1);
    if (fd >= 0)
      ::close(fd);
    if (ffmpeg_pid > 0)
      ::kill(ffmpeg_pid, SIGTERM);
    if (vspipe_pid > 0)
      ::kill(vspipe_pid, SIGTERM);
  }

  void stop() {
    interrupt();
    terminate_process(ffmpeg_pid);
    terminate_process(vspipe_pid);
    ffmpeg_pid = -1;
    vspipe_pid = -1;
    if (ffmpeg_stderr_thread.joinable())
      ffmpeg_stderr_thread.join();
    if (vspipe_stderr_thread.joinable())
      vspipe_stderr_thread.join();
    ffmpeg_stderr_fd = -1;
    vspipe_stderr_fd = -1;

    std::error_code ignored;
    if (!script_path.empty())
      std::filesystem::remove(script_path, ignored);
    if (!fifo_path.empty())
      std::filesystem::remove(fifo_path, ignored);
    if (!temp_dir.empty())
      std::filesystem::remove(temp_dir, ignored);
    script_path.clear();
    fifo_path.clear();
    temp_dir.clear();
    header_read = false;
  }
};

SvpFrameSource::SvpFrameSource(std::string media_path, MediaMetadata metadata,
                               SvpOptions options)
    : impl_(std::make_unique<Impl>(std::move(media_path), std::move(metadata),
                                   std::move(options))) {}

SvpFrameSource::~SvpFrameSource() = default;

void SvpFrameSource::start(double position_seconds) {
  impl_->start(position_seconds);
}

std::optional<DecodedVideoFrame> SvpFrameSource::next_frame() {
  return impl_->next_frame();
}

void SvpFrameSource::seek(double position_seconds) {
  impl_->start(position_seconds);
}

void SvpFrameSource::interrupt() { impl_->interrupt(); }

void SvpFrameSource::stop() { impl_->stop(); }

SvpDiagnostics SvpFrameSource::diagnostics() const {
  std::lock_guard lock(impl_->diagnostics_mutex);
  return impl_->diagnostics;
}

} // namespace localbooru::native_video
