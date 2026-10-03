# DMC Android startup patch

Published Wry 0.54.1 source, licensed MIT OR Apache-2.0. Its upstream source
revision is recorded in `.cargo_vcs_info.json`.

An exact signed DMC APK cold-boot crash was resolved to Android `mod.rs:474:38`:
`platform_webview_version` unwrapped a ten-second main-thread reply timeout.
A plain error return is insufficient for this Tauri runtime: it caches probe
success and rejects subsequent WebView creation when the provider probe fails.

DMC keeps the original Android version request alive for a bounded 60-second
startup budget (10-second initial probe plus 50-second grace). It returns the
actual provider response and propagates genuine Java/channel errors. A late
version response after cancellation no longer panics on the Java main thread.
Desktop platform implementations and Kotlin templates are unchanged. One
upstream SECURITY.md trailing space is normalized for the repository diff gate.

Host tests include the exact response helper from `src-tauri/src/lib.rs`. They
exercise ready and delayed real replies, timeout, disconnection, Java errors and
canceled late replies. Signed APK cold-boot/UI acceptance is also required.

Upstream development source now propagates this query timeout with `?`:
https://github.com/tauri-apps/wry/blob/dev/src/android/mod.rs
