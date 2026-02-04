//! Frame Interpolation Pipeline
//!
//! This module provides frame interpolation support for GStreamer video playback.
//! It supports multiple backends:
//!
//! 1. **SVP (SmoothVideo Project)**: Uses mpv with SVP pipe for professional-grade
//!    motion interpolation. Requires SVP to be installed at /opt/svp/.
//!
//! 2. **RIFE-NCNN**: Uses the rife-ncnn-vulkan binary for neural network based
//!    interpolation. Works on NVIDIA, AMD, and Intel GPUs via Vulkan.
//!
//! 3. **FFmpeg minterpolate**: Software-based motion interpolation using FFmpeg's
//!    built-in minterpolate filter. Slower but always available.
//!
//! # Architecture
//!
//! The interpolation pipeline works by:
//! 1. Extracting frames from GStreamer via appsink
//! 2. Passing them through an interpolation backend
//! 3. Injecting interpolated frames back via appsrc
//!
//! For real-time playback, we use a pipeline approach where FFmpeg decodes
//! and the interpolator processes frames on-the-fly.

mod config;
pub mod backends;
mod pipeline;

pub use config::{InterpolationConfig, InterpolationPreset, InterpolationBackend};
pub use backends::{detect_available_backends, BackendCapabilities, recommend_backend};
pub use pipeline::{InterpolatedPlayer, InterpolationState};
