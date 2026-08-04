# Video Player Control Parity Contract

This document defines “identical to the original player” as observable behavior. The React Lightbox is the reference player. The native GTK player may use a different rendering implementation, but must preserve the same state transitions, control availability, keyboard and pointer behavior, labels, and failure semantics.

## Ownership and proof vocabulary

- **Canonical state owner:** Rust `NativeVideoState` and the helper playback state. GTK controls project this state; they do not maintain authoritative optimistic state.
- **React owner:** `frontend/src/components/Lightbox/Lightbox.jsx` and the hooks under `frontend/src/components/Lightbox/hooks/`.
- **Native owner:** `src-tauri/src/native_video/platform/linux.rs`, with commands in `commands.rs` and helper protocol state.
- **Automated proof:** a named unit/integration test that can run without subjective inspection.
- **Manual proof:** side-by-side interaction or screenshot evidence. Manual proof supplements but never replaces state/command tests.

Status values are **Implemented**, **Partial**, or **Missing**. Partial and Missing rows are Task 11/12 acceptance work, not waived differences.

## Controls and observable states

| React behavior | Native implementation owner | Required native behavior | Current status | Automated proof | Manual proof |
|---|---|---|---|---|---|
| Previous item | GTK previous semantic action → React navigation | Select the previous item once; preserve close/cast rules | Implemented | Semantic event integration test | Click first HUD button and verify selected item |
| Play/pause | `SetPaused`; helper playback state | Toggle from canonical state; icon changes only when state confirms | Implemented; manual visual confirmation pending | Helper pause/resume and canonical-state tests | Compare paused frame and icon |
| Next item | GTK next semantic action → React navigation | Select the next item once | Implemented | Semantic event integration test | Click next and verify item |
| Current time and duration | GTK labels from `PlaybackState` | Same rounded `m:ss`/`h:mm:ss` display as React | Implemented | GTK playback-state formatting test required | Compare at identical PTS |
| Timeline click | GTK `Scale` → `Seek` | Seek to clicked absolute time and clamp to duration | Implemented | Helper seek test | Click 25%, 50%, 90% |
| Timeline drag/scrub | GTK timeline | Begin/update/end transaction; preview while dragging; commit one destructive seek at end | Implemented; real hit-test pending | Helper seek test; real GDK drag remains required | Slow and fast pointer drags |
| Touch timeline | GTK `GestureDrag` controller | Same seek transaction from touch-capable pointer input | Implemented; real touch-device acceptance pending | Compiled GDK touch-only gesture path; hardware event required | Touch-screen interaction |
| Timeline thumbnail and hover time | GTK HUD | Show timestamp and available thumbnail without moving playhead | Partial: hover timestamp implemented; native thumbnail lookup unavailable | Tooltip formatting test; preview integration remains required | Compare hover at same position |
| Buffered range | GTK HUD | Show SVP/stream buffered region when meaningful; omit for unbuffered local playback | Satisfied by omission: native local/SVP delivery has bounded frame queues, not a durable media-time buffer | Queue-bound and diagnostics tests | Confirm no misleading buffered bar |
| Mute | `SetMuted`; audio session | Toggle canonical muted state; preserve volume | Implemented; button restores canonical state until helper confirmation | Audio-session mute and canonical projection tests | Toggle button and M key |
| Volume slider | `SetVolume`; audio session | 0–100%, keyboard ±5%, preserve across renderer handoff | Implemented | Audio volume and handoff tests; real GTK key path pending | Slider and Up/Down |
| Fullscreen | owner GTK window | Enter/exit, preserve HUD, F and double-click parity | Implemented; composited acceptance pending | Fullscreen runtime-spike path | Button, F, double-click |
| Playback speed | `SetSpeed`; helper/audio | 0.25× steps, visible non-1× badge, Backspace resets 1× | Implemented | Speed/A/V and handoff tests; real GTK key path pending | Compare 1.25× and reset |
| Fit/fill/original | canonical geometry + `set_display_mode` | Cycle Fit → Fill → Original; identical SAR/rotation crop | Implemented | Canonical geometry and display-mode tests | Compare all modes side-by-side |
| Subtitle toggle | subtitle track command | Off/on selected track without losing selection | Implemented for embedded tracks | Helper track-selection test | Toggle same subtitle |
| Subtitle track/language menu | GTK subtitle combo | List IDs/languages and current selection | Implemented | Track-list projection test required | Compare available tracks |
| Subtitle delay | `SetSubtitleDelay` | Range ±10 s, 250 ms step, canonical value | Implemented | Helper subtitle-delay test | Compare delayed cue |
| Whisper generation/progress/error | GTK `CC+`, Tauri whisper events | Generate, show progress, completion and recoverable error; never freeze playback | Partial: request/status plumbing exists; React-matching presentation incomplete | Whisper status-generation tests | Trigger success/error |
| Audio track selection | GTK audio combo → `SelectAudioTrack` | List tracks and preserve PTS/paused state while switching | Implemented | Helper lifecycle probe track switch | Switch while playing and paused |
| SVP toggle | GTK interpolation control → helper | Off/on without losing playhead, paused state, audio track, or A/V sync | Implemented | `probe_svp_lifecycle.py` | Toggle at visible timestamp |
| SVP preset/target | helper interpolation command | Show selected preset and target FPS | Partial: balanced/60 default only | Protocol serialization and cadence tests | Compare selected mode |
| SVP buffering/cancel/error | helper diagnostics/state | Distinct loading, buffering, cancellation, recoverable fallback and fatal failure | Missing in GTK HUD | State-projection tests required | Force missing plugin/cancel |
| Quality selector | player-specific | React/HLS shows quality choices; local native explicitly says “Original/local” and does not invent qualities | Implemented as `Original/local` badge | Renderer policy tests | Compare local and HLS media |
| Loading | first-frame generation state | Keep prior React image/frame until matching native first-frame draw completes | Implemented | Coordinator generation tests | Open slow media |
| Buffering | playback event state | Show spinner only while stalled; retain controls | Partial | Buffering transition test required | Induce SVP startup/stall |
| Seeking | seek transaction state | Show seeking state; do not report new time before canonical confirmation | Partial | Seek state-order test required | Seek repeatedly |
| EOF | helper `PlaybackEnded` | Stop at duration or follow loop/auto-advance exactly once | Partial | EOF coordinator test required | Play short fixture to end |
| Recoverable fallback | coordinator | Preserve canonical state and visibly identify effective fallback | Implemented for renderer failure; HUD message incomplete | Coordinator fallback tests | Force helper/SVP failure |
| Fatal error | runtime error event | Stop native resources, retain close/navigation, show actionable error | Partial | Runtime failure cleanup test | Kill helper during playback |
| Auto-hide/reveal | GTK pointer activity timer | Match React delay; reveal on pointer, key, focus and touch; never hide an open menu | Implemented for pointer, keyboard, focus-owned menus, scrubbing, and touch drag; manual timing pending | Timer/menu-focus tests still required | Idle, move, keyboard, menu |
| Pointer video zones | GTK event controller | Left/right thirds seek ±10 s; center toggles play/pause; show 600 ms indicator | Implemented; real hit-test pending | Compiled event path; real hit-test required | Click all thirds |
| Touch horizontal drag | GTK `GestureDrag` controller | >20 px, horizontal >1.5× vertical, 10 s per ~50 px, commit on release | Implemented; real touch-device acceptance pending | Compiled touch-only gesture path; hardware event required | Repeat React gesture |
| Focus and menus | GTK HUD | Visible focus ring, keyboard navigation, Escape closes menu first, restore trigger focus | Partial: CSS focus ring and native GTK traversal exist; menu precedence/restore require real focus testing | Focus traversal test required | Keyboard-only pass |

## Keyboard contract

| Shortcut | React behavior | Required native behavior | Current native status |
|---|---|---|---|
| Space | Play/pause | Same | Implemented |
| Left / Right | Seek −/+5 s | Same | Implemented |
| Shift+Left / Shift+Right | Seek −/+1 s | Same | Implemented |
| Ctrl/Cmd+Left / Ctrl/Cmd+Right | Seek −/+30 s | Same | Implemented for Control on Linux |
| Up / Down | Volume ±5% | Same | Implemented |
| M | Toggle mute | Same | Implemented, case-normalized |
| F | Toggle fullscreen | Same | Implemented, case-normalized |
| +, =, ] | Speed +0.25× | Same | Implemented |
| -, [ | Speed −0.25× | Same | Implemented |
| Backspace | Reset speed to 1× | Same | Implemented |
| E | Advance one frame while paused | Same | Implemented as a 1/60-second seek; exact source-frame stepping remains target-dependent |
| C | Toggle subtitles | Same | Implemented |
| I | Toggle diagnostics | Same native diagnostics overlay/state | Implemented |
| Escape | Close player; close open menu first | Same | Partial: closes player, menu precedence missing |
| Ctrl/Cmd+C | Copy current image only | Not applicable to video; do not intercept | Satisfied by scope |

## Visual acceptance

Task 11 must use SVG icons rather than text/emoji glyphs and match the React HUD’s control order, margins, timeline thickness, opacity, active/hover/focus states, loading language, and auto-hide timing. Acceptance requires screenshots at the same viewport for: playing, paused, timeline hover, non-1× speed, subtitle menu, SVP buffering, recoverable fallback, and fullscreen.

## Handoff acceptance

Task 12 must preserve this payload across React/native/SVP transitions: item and generation, position and duration, paused state, volume and mute, speed, audio/subtitle selection, subtitle delay, SVP engine/preset/target, and display mode. Unsupported target-specific state must produce an explicit fallback status rather than silently resetting.
