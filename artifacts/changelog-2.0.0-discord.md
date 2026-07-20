What's New in LocalBooru 2.0.0

🦀 **HEADLINE: A NATIVE RUST CORE AND TAURI 2 DESKTOP APP**
• LocalBooru’s desktop application has moved from Electron and a permanently coupled Python backend to **Tauri 2 with an embedded Rust/Axum server**
• Library queries, importing, directory watching, metadata extraction, task scheduling, sharing, transcoding, authentication, and most media services now run in the Rust core
• Python is isolated to optional add-ons instead of being required for ordinary library browsing and management
• The old Electron frontend and retired Tauri v1/video modules have been removed, leaving one current desktop architecture
• Existing databases are upgraded through the new migration system, with compatibility fixes for task attempts, watch history, directories, and mounted libraries

📚 **MULTI-LIBRARY WORKSPACES AND FASTER IMPORTS**
• Mount several independent LocalBooru libraries and switch between them without merging their databases or media folders
• Library tabs on the Directories page include the primary library, mounted libraries, and visibly dimmed offline libraries with mount controls
• Favorites, ratings, deletes, moves, pruning, searches, and directory operations now retain the correct library identity instead of silently targeting the primary library
• New scans use a **two-phase import**: files appear in the gallery quickly, while thumbnails, perceptual hashes, metadata, and video probes complete in the background
• Metadata extraction recognizes Automatic1111 generation parameters and configurable ComfyUI prompt/sampler nodes for searchable image provenance
• Parent-directory watching can discover subdirectories automatically, including explicitly watched symlinked folders
• Directory tools now cover verify, repair, relocate, refresh, prune, remove, and bulk operations across selected directories
• Repair can fix relocated paths, remove genuinely missing records, clean orphan thumbnails, and avoid re-importing files already moved to the dumpster
• The task queue gained atomic claims, deduplication, priorities, retry, pause/resume, cancellation, and safer recovery after restart

🧩 **OPTIONAL ADD-ONS WITH THEIR OWN LIFECYCLES**
• Auto Tagger, Age Detector, Whisper Subtitles, casting, and SVP-dependent services run as separately managed add-ons rather than one all-or-nothing Python environment
• Add-ons can be installed on demand into isolated environments, started, stopped, repaired/updated, and uninstalled from the app
• Installed configurable add-ons appear in one **Add-on Settings** area; unavailable tools no longer leave irrelevant settings behind
• Auto Tagger settings expose model download/status, CUDA/CPU/automatic device choice, general and character confidence thresholds, and the provider actually selected at runtime
• Age Detector can be run retrospectively over existing images as well as enabled for watched-directory processing
• Add-on startup and task execution recover more reliably after restarts, including deployment refreshes and status checks against the real sidecar
• Family Mode can hide non-family-safe directories, locks again after restart, and now rate-limits repeated PIN attempts per client

🎞️ **VIDEO SEEKING, TRANSCODING, AND SMOOTHER FALLBACKS**
• Local files support HTTP Range requests, so normal video playback can seek without downloading the whole file first
• HLS transcoding moved into the Rust service with quality presets, cleanup, hardware-encoder detection, and software fallback
• Direct, transcoded, and SVP playback can consistently attenuate loud sources toward the normalization target without automatically amplifying quiet sources
• Codec fallback preserves the current playhead instead of restarting the video from zero
• SVP HLS playback waits for usable buffered media before starting and no longer force-restarts an established stream on ordinary browser buffering events
• SVP seeking checks the media element's real buffered ranges, seeking locally when possible and restarting the stream only when necessary
• Auto-advance waits for the real end of playback instead of firing at an HLS segment boundary or while a video is seeking/buffering
• The previous Linux configuration that could force WebKit into degraded ~15 FPS playback without a working VA-API decoder has been removed
• NVOF startup failures now fall back instead of silently ending the SVP stream, even when the failed renderer returns no useful error text
• Video preview generation, stream cleanup on quit, pause-on-hide behavior, and “Continue Watching” state received reliability fixes
• Open a supported local image or video without importing it first using the title-bar folder button, `Ctrl+O`, or a media path passed to LocalBooru; subsequent open requests are forwarded to the existing window
• Fullscreen now targets the Lightbox itself instead of the entire application document
• The retired FFmpeg optical-flow UI has been removed; SVP is the supported interpolation path

✨ **SVP CONTROL PANEL INTEGRATION — LINUX**
• LocalBooru's original WebKit/GStreamer player can connect to the user's separately installed **SVP Control Panel** and apply its live script/preset output without replacing the WebView player
• Switching ordinary playback → SVP → ordinary playback keeps the same media session instead of handing playback to a separate player
• The SVP GStreamer filter uses a persistent bounded pipeline with managed worker cleanup rather than launching a new pipeline for every frame
• Source frame rate is retained as an exact rational value, avoiding drift from approximations such as treating 23.976 as 23976/1000
• SVP remains optional and proprietary SVP/SVPflow components are **not bundled**; LocalBooru does not force a GPU or replace the device selected in SVP Control Panel
• When SVP is off, the same packaged application keeps ordinary unfiltered WebKit/GStreamer playback available

📱 **TAURI MOBILE, QR PAIRING, AND REMOTE ACCESS**
• The mobile client moved from Capacitor to **Tauri Mobile**, with Android scaffolding and desktop-only features cleanly separated from mobile builds
• Pair a phone with a desktop server by QR code and connect through the authenticated remote proxy without Android WebView mixed-content failures
• Saved servers can use primary and fallback addresses, including fast Tailscale fallback when the main address is unreachable
• Remote URLs are normalized on save, health checks validate authenticated API access, and the active reachable address is remembered
• Android fullscreen uses real immersive mode, and live safe-area insets keep the title bar, Lightbox controls, counters, and cast sheets clear of system bars through rotation
• Mobile edge taps can navigate zoomed images without turning a real pan gesture into an accidental next/previous action
• Fixed Android WebView gamma adjustments and a portrait-layout bug that could slide the Lightbox sidebar in while leaving it transparent
• The iOS build pipeline can produce an unsigned device IPA intended for local AltStore/SideStore re-signing

📺 **CASTING AND SHARING**
• Chromecast and UPnP/DLNA discovery, status, playback, and stop controls are connected to the Rust backend and optional cast service
• The redesigned cast view shows a centered **Playing on…** card with device, media title, live state, and a clear stop action while local video is dimmed
• The device picker has explicit scanning/empty states, device counts, click-away dismissal, and a mobile bottom-sheet layout
• Chromecast automatically transcodes containers, codecs, bit depths, resolutions, and frame rates that are unsafe for direct playback, with stream cleanup when media changes or casting stops
• Chromecast can use the configured SVP stream when the SVP add-on is active, while DLNA keeps its existing playback path
• Shared HLS manifests keep segment access under the authenticated share token instead of exposing an unscoped image URL
• Share creation, cast media, file-serving, and path-based settings validate authorized library paths rather than trusting arbitrary client paths

🖼️ **GALLERY, LIGHTBOX, AND EVERYDAY WORKFLOW**
• Live page refreshes merge new or changed first-page records without throwing away images already loaded farther down the gallery
• Pagination retries unexpected empty middle pages with bounded backoff instead of silently advancing past missing results
• Selection mode supports batch retagging, age detection, moving, and deleting from the gallery
• Gallery and Lightbox items have right-click actions for copying images and pasting images from the clipboard
• Missing files and offline drives have distinct overlays instead of looking like ordinary thumbnail failures
• Unrated and partially populated database records no longer disappear from rating-filtered results
• Video cards keep lightweight thumbnail/preview behavior instead of loading full videos into the masonry grid
• Continue Watching can be shown or hidden from Settings, and window/tray transitions preserve cleaner media state

🔐 **SECURITY AND PRIVACY HARDENING**
• Passwords use Argon2, JWT secrets are generated per installation, and database queries/path operations received injection and traversal hardening
• Browser/mobile media URLs use short-lived, read-only media tokens rather than putting a full session token into every image URL
• Media tokens are accepted only for read-only image access and cannot authenticate API writes
• Tauri’s asset protocol is default-deny and grants access only to watched directories instead of exposing a global filesystem wildcard
• Referrer policy prevents query-string media tokens from leaking to navigated sites
• CORS now permits only the headers LocalBooru actually uses, while local-network, Tailscale, and localhost access retain their intended behavior
• Family Mode PIN attempts use exponential lockout, share creation is rate-limited, and top-level supply-chain-sensitive add-on dependencies are pinned

📦 **REPRODUCIBLE LINUX RELEASE DELIVERY**
• Linux releases are now built by one repository-owned Docker workflow instead of obsolete Electron-era CI packaging
• The same wrapper produces **AppImage, DEB, RPM, portable ZIP, SHA-256 manifest, and native-runtime source offer** from the 2.0.0 source tree
• The release image builds patched public WebKitGTK 2.52.3 and VapourSynth R75; it never copies proprietary SVPflow libraries from the build machine
• Packages carry the patched WebKit runtime, GStreamer/VapourSynth integration, required Python and JPEG XL dependency closure, and matching third-party notices
• Release verification extracts every format, compares the runtime payload across packages, checks required RPATH/dependency metadata, rejects private host/build paths, and rejects bundled proprietary SVP components
• Package metadata and all JavaScript, Rust, and Tauri manifests now agree on version **2.0.0**

🛠️ **RELIABILITY FIXES SINCE 0.3.33**
• Fixed directory-watcher runtime/thread failures, rename handling, recursive discovery, and startup reconciliation
• Fixed task workers racing to claim the same item, crashing on old NULL attempt counts, or processing video files as still-image tagging jobs
• Fixed duplicate imports and mounted-library operations that could target the wrong database
• Fixed pruning so dumpster contents are excluded from scans, watchers, repair, and re-import
• Fixed API caching that could leave stale gallery state in WebKitGTK
• Fixed large operations timing out too early and improved retry behavior for imports, thumbnails, metadata, and add-on tasks
• Fixed remote auto-connect messages that blamed expired authentication when the real cause could be an unreachable or busy server
• Fixed cast, saved-search, Whisper, migration, collection, directory, and watch-history request/response mismatches uncovered during the v2 audit
