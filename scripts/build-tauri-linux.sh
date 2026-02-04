#!/bin/bash
# Build Tauri app for Linux
# Creates AppImage, .deb, and .rpm packages
#
# Usage:
#   ./scripts/build-tauri-linux.sh              # Build all targets
#   ./scripts/build-tauri-linux.sh --appimage   # AppImage only
#   ./scripts/build-tauri-linux.sh --deb        # Debian package only
#   ./scripts/build-tauri-linux.sh --rpm        # RPM package only
#   ./scripts/build-tauri-linux.sh --debug      # Debug build (faster, larger)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
BUILD_TARGET=""
DEBUG_BUILD=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --appimage) BUILD_TARGET="appimage"; shift ;;
        --deb) BUILD_TARGET="deb"; shift ;;
        --rpm) BUILD_TARGET="rpm"; shift ;;
        --debug) DEBUG_BUILD=true; shift ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --appimage    Build AppImage only"
            echo "  --deb         Build Debian package only"
            echo "  --rpm         Build RPM package only"
            echo "  --debug       Create debug build (faster, larger)"
            echo "  --help        Show this help"
            exit 0
            ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "LocalBooru Linux Build Script (Tauri)"
echo "============================================================"
echo ""

cd "$PROJECT_ROOT"

# Step 1: Check prerequisites
log_info "Checking prerequisites..."

# Check Rust
if ! command -v cargo &> /dev/null; then
    log_error "Rust/Cargo not found. Install from https://rustup.rs"
    exit 1
fi
log_success "Rust $(cargo --version | cut -d' ' -f2) found"

# Check Node.js
if ! command -v node &> /dev/null; then
    log_error "Node.js not found. Install Node.js 18+"
    exit 1
fi
log_success "Node.js $(node --version) found"

# Check Python
PYTHON_CMD=""
for cmd in python3.11 python3.12 python3.10 python3; do
    if command -v $cmd &> /dev/null; then
        VERSION=$($cmd --version 2>&1 | cut -d' ' -f2)
        MINOR=$(echo $VERSION | cut -d. -f2)
        if [ "$MINOR" -ge 10 ]; then
            PYTHON_CMD=$cmd
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    log_error "Python 3.10+ not found"
    exit 1
fi
log_success "Python $($PYTHON_CMD --version | cut -d' ' -f2) found"

# Check Tauri CLI
if ! command -v cargo-tauri &> /dev/null; then
    log_warn "Tauri CLI not found, installing..."
    cargo install tauri-cli
fi
log_success "Tauri CLI installed"

# Check system dependencies for Linux builds
log_info "Checking Linux build dependencies..."

MISSING_DEPS=""

# Check for essential libs
pkg-config --exists webkit2gtk-4.1 2>/dev/null || MISSING_DEPS="$MISSING_DEPS webkit2gtk-4.1"
pkg-config --exists gstreamer-1.0 2>/dev/null || MISSING_DEPS="$MISSING_DEPS gstreamer-1.0"
pkg-config --exists gstreamer-video-1.0 2>/dev/null || MISSING_DEPS="$MISSING_DEPS gstreamer-video-1.0"

if [ -n "$MISSING_DEPS" ]; then
    log_error "Missing development packages:$MISSING_DEPS"
    echo ""
    echo "Install on Ubuntu/Debian:"
    echo "  sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev"
    echo ""
    echo "Install on Fedora:"
    echo "  sudo dnf install webkit2gtk4.1-devel gtk3-devel gstreamer1-devel gstreamer1-plugins-base-devel"
    echo ""
    echo "Install on Arch:"
    echo "  sudo pacman -S webkit2gtk-4.1 gtk3 gstreamer gst-plugins-base"
    exit 1
fi
log_success "System development libraries found"

# Step 2: Prepare Python bundle
log_info "Checking Python bundle..."

PYTHON_VENV="$PROJECT_ROOT/python-venv-linux"

if [ ! -d "$PYTHON_VENV" ] || [ ! -f "$PYTHON_VENV/bin/python" ]; then
    log_warn "Python bundle not found, creating..."
    node "$SCRIPT_DIR/prepare-python-bundle-linux.js"
else
    log_success "Python bundle exists at $PYTHON_VENV"
    # Check if requirements might have changed
    if [ "$PROJECT_ROOT/requirements.txt" -nt "$PYTHON_VENV" ]; then
        log_warn "requirements.txt is newer than bundle, consider rebuilding with:"
        echo "  node scripts/prepare-python-bundle-linux.js"
    fi
fi

# Step 3: Install frontend dependencies
log_info "Installing frontend dependencies..."
cd "$PROJECT_ROOT/frontend"
if [ ! -d "node_modules" ]; then
    npm install
fi
log_success "Frontend dependencies ready"

# Step 4: Build frontend
log_info "Building frontend..."
npm run build
log_success "Frontend built"

cd "$PROJECT_ROOT"

# Step 5: Build Tauri app
log_info "Building Tauri application..."

TAURI_ARGS=""

if [ "$DEBUG_BUILD" = true ]; then
    TAURI_ARGS="--debug"
    log_info "Debug build enabled"
fi

if [ -n "$BUILD_TARGET" ]; then
    TAURI_ARGS="$TAURI_ARGS --bundles $BUILD_TARGET"
    log_info "Building target: $BUILD_TARGET"
else
    log_info "Building all targets: appimage, deb, rpm"
fi

# Run Tauri build
cargo tauri build $TAURI_ARGS

# Step 6: Report results
echo ""
echo "============================================================"
log_success "Build complete!"
echo "============================================================"
echo ""

OUTPUT_DIR="$PROJECT_ROOT/src-tauri/target/release/bundle"
if [ "$DEBUG_BUILD" = true ]; then
    OUTPUT_DIR="$PROJECT_ROOT/src-tauri/target/debug/bundle"
fi

if [ -d "$OUTPUT_DIR" ]; then
    log_info "Build artifacts:"
    echo ""

    # AppImage
    if [ -d "$OUTPUT_DIR/appimage" ]; then
        APPIMAGE=$(find "$OUTPUT_DIR/appimage" -name "*.AppImage" 2>/dev/null | head -1)
        if [ -n "$APPIMAGE" ]; then
            SIZE=$(du -h "$APPIMAGE" | cut -f1)
            echo "  AppImage: $(basename "$APPIMAGE") ($SIZE)"
        fi
    fi

    # Deb
    if [ -d "$OUTPUT_DIR/deb" ]; then
        DEB=$(find "$OUTPUT_DIR/deb" -name "*.deb" 2>/dev/null | head -1)
        if [ -n "$DEB" ]; then
            SIZE=$(du -h "$DEB" | cut -f1)
            echo "  Debian:   $(basename "$DEB") ($SIZE)"
        fi
    fi

    # RPM
    if [ -d "$OUTPUT_DIR/rpm" ]; then
        RPM=$(find "$OUTPUT_DIR/rpm" -name "*.rpm" 2>/dev/null | head -1)
        if [ -n "$RPM" ]; then
            SIZE=$(du -h "$RPM" | cut -f1)
            echo "  RPM:      $(basename "$RPM") ($SIZE)"
        fi
    fi

    echo ""
    echo "Output directory: $OUTPUT_DIR"
fi

echo ""
log_info "To test the AppImage:"
echo "  chmod +x $OUTPUT_DIR/appimage/*.AppImage"
echo "  $OUTPUT_DIR/appimage/*.AppImage"
