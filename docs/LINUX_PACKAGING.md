# LocalBooru Linux Packaging Guide

This document covers building and distributing LocalBooru for Linux using Tauri.

## Package Formats

LocalBooru is distributed in three Linux package formats:

| Format    | Description                          | Best For                    |
|-----------|--------------------------------------|-----------------------------|
| AppImage  | Self-contained portable executable   | Universal compatibility     |
| .deb      | Debian/Ubuntu package                | Debian, Ubuntu, Mint        |
| .rpm      | RPM package                          | Fedora, RHEL, openSUSE      |

## Build Prerequisites

### Development Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install -y \
    build-essential \
    curl \
    wget \
    file \
    libssl-dev \
    libgtk-3-dev \
    libwebkit2gtk-4.1-dev \
    libayatana-appindicator3-dev \
    librsvg2-dev \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    python3-dev \
    python3-venv
```

#### Fedora
```bash
sudo dnf install -y \
    gcc \
    gcc-c++ \
    make \
    openssl-devel \
    gtk3-devel \
    webkit2gtk4.1-devel \
    libappindicator-gtk3-devel \
    librsvg2-devel \
    gstreamer1-devel \
    gstreamer1-plugins-base-devel \
    gstreamer1-plugins-good \
    gstreamer1-plugins-bad-free \
    gstreamer1-plugins-ugly-free \
    gstreamer1-libav \
    python3-devel
```

#### Arch Linux
```bash
sudo pacman -S --needed \
    base-devel \
    openssl \
    gtk3 \
    webkit2gtk-4.1 \
    libappindicator-gtk3 \
    librsvg \
    gstreamer \
    gst-plugins-base \
    gst-plugins-good \
    gst-plugins-bad \
    gst-plugins-ugly \
    gst-libav \
    python
```

### Rust and Node.js

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Install Node.js (via nvm recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 20
nvm use 20

# Install Tauri CLI
cargo install tauri-cli
```

## Building

### Quick Build (All Formats)

```bash
./scripts/build-tauri-linux.sh
```

### Build Specific Format

```bash
# AppImage only (fastest)
./scripts/build-tauri-linux.sh --appimage

# Debian package
./scripts/build-tauri-linux.sh --deb

# RPM package
./scripts/build-tauri-linux.sh --rpm
```

### Debug Build

For faster builds during development (larger output, includes debug symbols):

```bash
./scripts/build-tauri-linux.sh --debug
```

### Manual Build

```bash
# 1. Prepare Python bundle (if not exists)
node scripts/prepare-python-bundle-linux.js

# 2. Build frontend
cd frontend && npm install && npm run build && cd ..

# 3. Build Tauri
cargo tauri build
```

## Output

Build artifacts are located at:
- **Release**: `src-tauri/target/release/bundle/`
- **Debug**: `src-tauri/target/debug/bundle/`

```
bundle/
├── appimage/
│   └── local-booru_X.X.X_amd64.AppImage
├── deb/
│   └── local-booru_X.X.X_amd64.deb
└── rpm/
    └── local-booru-X.X.X-1.x86_64.rpm
```

## Runtime Dependencies

### AppImage

AppImage bundles most dependencies, including GStreamer media framework. Users only need:
- A compatible Linux kernel (3.2+)
- FUSE for mounting (`libfuse2` on most systems)
- Optional: GPU drivers for hardware-accelerated video

```bash
# If AppImage fails to run, install FUSE
sudo apt install libfuse2  # Debian/Ubuntu
sudo dnf install fuse      # Fedora
```

### Debian Package (.deb)

The .deb package declares these dependencies:
- `libc6` - C library
- `libgtk-3-0` - GTK3 for UI
- `libwebkit2gtk-4.1-0` - WebKit for webview
- `libgstreamer1.0-0` - GStreamer core
- `gstreamer1.0-plugins-*` - GStreamer plugins for video
- `python3 (>= 3.10)` - Python runtime

Recommended (for better video support):
- `gstreamer1.0-vaapi` - Intel/AMD hardware acceleration
- `gstreamer1.0-gl` - OpenGL video rendering
- `ffmpeg` - Video transcoding

### RPM Package (.rpm)

Similar dependencies for Fedora/RHEL:
- `webkit2gtk4.1`
- `gtk3`
- `gstreamer1` and plugins
- `python3 >= 3.10`

## GStreamer Configuration

LocalBooru uses GStreamer for video playback. The bundled plugins include:

| Plugin Set        | Purpose                              |
|-------------------|--------------------------------------|
| base              | Core elements (filesrc, typefind)    |
| good              | High-quality, well-tested plugins    |
| bad               | Newer/experimental plugins           |
| ugly              | Patent-encumbered formats            |
| libav             | FFmpeg-based decoders                |

### Hardware Acceleration

For best video performance, install platform-specific plugins:

**Intel (VA-API):**
```bash
sudo apt install gstreamer1.0-vaapi intel-media-va-driver
```

**NVIDIA:**
```bash
sudo apt install gstreamer1.0-plugins-bad  # Includes nvcodec
```

**AMD:**
```bash
sudo apt install gstreamer1.0-vaapi mesa-va-drivers
```

### Verifying GStreamer

```bash
# Check available plugins
gst-inspect-1.0 | grep -E "(vaapi|nvcodec|v4l2)"

# Test video playback
gst-launch-1.0 playbin uri=file:///path/to/video.mp4
```

## Bundled Python Environment

LocalBooru bundles a Python virtual environment with all required packages. This ensures:

1. **No Python installation required** - Works on systems without Python
2. **Isolated dependencies** - No conflicts with system packages
3. **Consistent behavior** - Same packages across all installations

The bundle is created by `scripts/prepare-python-bundle-linux.js` and includes:
- Python interpreter (from system at build time)
- FastAPI, Uvicorn for API server
- ONNX Runtime for AI tagging
- Pillow, OpenCV for image processing
- Other requirements from `requirements.txt`

### Bundle Location

- **AppImage**: Extracted to temporary directory on run
- **.deb/.rpm**: `/usr/lib/local-booru/python-venv/`
- **Development**: `python-venv-linux/`

### Updating the Bundle

If dependencies change:

```bash
# Rebuild Python bundle
node scripts/prepare-python-bundle-linux.js

# Then rebuild packages
./scripts/build-tauri-linux.sh
```

## Troubleshooting

### AppImage Won't Start

```bash
# Check if FUSE is available
fusermount --version

# Run with verbose output
APPIMAGE_EXTRACT_AND_RUN=1 ./LocalBooru.AppImage

# Or extract and run directly
./LocalBooru.AppImage --appimage-extract
./squashfs-root/AppRun
```

### GStreamer Errors

```bash
# Check for missing plugins
GST_DEBUG=3 ./LocalBooru.AppImage 2>&1 | grep -i plugin

# Install common codecs
sudo apt install gstreamer1.0-plugins-ugly gstreamer1.0-libav
```

### Python Backend Issues

```bash
# Check bundled Python
ls -la /path/to/localbooru/python-venv/bin/

# Test manually
/path/to/localbooru/python-venv/bin/python -c "import fastapi; print('OK')"
```

### WebKit2GTK Version Mismatch

If you get WebKit version errors on older systems:

```bash
# Check installed version
pkg-config --modversion webkit2gtk-4.1

# The app requires webkit2gtk-4.1 (not 4.0)
# On Ubuntu 20.04, you may need to use the AppImage instead
```

## Flatpak (Future)

Flatpak support is planned for easier dependency management. Benefits:
- Sandboxed execution
- Automatic updates via Flathub
- Bundled runtime (no system deps)

Limitations:
- Sandboxing may restrict file access
- Larger download size
- Requires Flatpak runtime

## CI/CD Integration

For automated builds in GitHub Actions:

```yaml
# .github/workflows/build-linux.yml
jobs:
  build-linux:
    runs-on: ubuntu-22.04
    steps:
      - uses: actions/checkout@v4

      - name: Install dependencies
        run: |
          sudo apt update
          sudo apt install -y libwebkit2gtk-4.1-dev libgtk-3-dev \
            libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

      - uses: actions/setup-node@v4
        with:
          node-version: '20'

      - uses: dtolnay/rust-toolchain@stable

      - name: Install Tauri CLI
        run: cargo install tauri-cli

      - name: Prepare Python bundle
        run: node scripts/prepare-python-bundle-linux.js

      - name: Build
        run: ./scripts/build-tauri-linux.sh

      - name: Upload artifacts
        uses: actions/upload-artifact@v4
        with:
          name: linux-packages
          path: |
            src-tauri/target/release/bundle/appimage/*.AppImage
            src-tauri/target/release/bundle/deb/*.deb
            src-tauri/target/release/bundle/rpm/*.rpm
```

## Version History

- **v0.3.10**: Initial Tauri Linux packaging
- Electron packaging deprecated in favor of Tauri
