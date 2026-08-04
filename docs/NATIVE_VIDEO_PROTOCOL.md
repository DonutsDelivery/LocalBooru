# Native Video Protocol

Protocol major version: `1000` (major = version / 1000).

Rust definitions live in `src-tauri/src/native_video/protocol.rs`. The helper must use tagged messages with a `type` field and ignore unknown optional fields within a compatible major version.

## Commands

- `hello`
- `open_media`, `close_media`
- `set_viewport`, `set_visible`
- `set_paused`, `seek`, `set_volume`, `set_muted`, `set_speed`
- `select_audio_track`, `select_subtitle_track`, `set_subtitle_delay`
- `set_interpolation`
- `pointer_move`, `pointer_down`, `pointer_up`, `scroll`, `key`
- `set_hud_visible`, `set_fullscreen`

## Events

- `ready`, `capabilities_changed`
- `media_opened`, `first_frame_ready`, `playback_state`
- `track_list`, `subtitle_track_added`
- `navigate_previous`, `navigate_next`, `close_requested`
- `hud_visibility_changed`, `diagnostics`
- `recoverable_error`, `fatal_error`, `gpu_path_changed`

## Generation invariant

Every media-bound asynchronous event carries the generation from `open_media`. Consumers discard events that do not match the canonical current generation.

## Shared-surface extension

DMA-BUF/handle transfer uses a local descriptor channel because file descriptors and native handles cannot be represented safely in JSON. Buffer ownership is explicit: producer announces buffers, emits frame-ready with synchronization, and does not reuse a buffer until frame-release.
