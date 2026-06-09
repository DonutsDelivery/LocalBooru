# LocalBooru Security Fixes — Working Doc

Status legend: ☐ todo · ◐ in progress · ☑ done · ⏸ blocked on decision

**Hard constraints (must NOT break):**
1. Localhost access from the same machine — full access, no auth.
2. LAN access via the Android app — works with JWT.
3. QR-code pairing → JWT token auth flow — unchanged.
4. **HTTP is a first-class, supported transport. No HTTPS certs exist.** Security comes from the **JWT/QR auth layer + a trusted network (Tailscale/LAN)**, NOT from TLS. **No fix may force HTTPS, reject HTTP, require valid certs, add HSTS, or otherwise break plaintext-http connections.**

Every fix below is checked against these four. Where a fix has a real tradeoff, it's marked **⏸ DECISION** and we discuss before touching code.

**Accepted residual risk (by design, NOT a bug):** over an untrusted plaintext-http network with no Tailscale/VPN, traffic — including the JWT — is eavesdroppable. The product assumes the transport is trusted (loopback, LAN, or Tailscale's WireGuard encryption). The fixes below harden the *auth/authorization logic*, which is what actually protects you on a trusted-but-shared network; they do not try to add transport encryption.

---

## How the access model works today (so we don't break it)

`src-tauri/src/server/middleware/access_control.rs` decides every request in this order:

1. `OPTIONS` → allow (CORS preflight)
2. `EXEMPT_PREFIXES` → allow with **no auth at all**: `/health`, `/docs`, `/assets`, `/thumbnails`, `/icon.png`, `/api/share/`, `/api/cast-media/`, `/watch/`
3. localhost IP → allow (full access)
4. **valid JWT → allow (full access) ← the bug in C2**
5. otherwise: block localhost-only prefixes (403) and everything else (401)

There's already a curated `LOCALHOST_EXEMPTIONS` list — the exact endpoints the LAN/Android client legitimately needs (pairing handshake, qr-data, login, verify, and user-preference settings like video-playback/optical-flow/svp). **This is the key insight: the design already separates "network-OK" endpoints from "localhost-only" ones. Most fixes just make the enforcement match that intent.**

---

## CRITICAL

### C2 — JWT bypasses localhost-only endpoints  ☑ DONE (compiles, runtime-verified: connect/video/images still work)
**File:** `access_control.rs:230`
**Problem:** The `if has_valid_jwt { return ... }` early-return runs *before* the localhost-only check. So any paired device's token can hit `POST /api/settings` (incl. set `jwt_secret`), `/api/network` (flip server to public), and `/api/users` (user management) — not just the curated `LOCALHOST_EXEMPTIONS`.
**Fix:** Reorder so `LOCALHOST_ONLY_PREFIXES` is enforced regardless of JWT. A JWT still grants: all normal app endpoints + everything in `LOCALHOST_EXEMPTIONS`. It no longer grants the sensitive roots.
**Constraint check:** ✅ Localhost unchanged. ✅ Android keeps every endpoint it uses (all in LOCALHOST_EXEMPTIONS). ✅ QR/JWT unchanged.
**Extra (defense-in-depth):** in the `POST /api/settings` deep-merge handler, refuse to overwrite `jwt_secret` / `network.*` keys even from localhost, so a bug can't silently clobber them.
**Risk:** Low. Pure reordering + a key guard.

### C1 — TLS "accept invalid certs"  →  BY DESIGN, no change
**File:** `src-tauri/src/commands.rs:282, 316`
**Re-assessed:** Per constraint 4, there are no HTTPS certs and HTTP is the supported transport. For `http://` URLs this flag is a no-op (no TLS at all). It only matters if a user points at an `https://` self-signed endpoint (e.g. Tailscale-served), where accepting the self-signed cert is *required* for it to work — and Tailscale already encrypts at the network layer.
**Decision:** **Leave as-is.** Forcing cert validation would break self-signed/no-cert connections (constraint 2/4). Security on the wire is delegated to the trusted network, per the accepted-residual-risk note above.
**Optional, only if you ever standardize on https with a pinned fingerprint:** verify the presented cert against the QR fingerprint instead of `accept_invalid_certs`. Not needed for the current http model. Parked.

### C3 — Unauthenticated media serving + unvalidated `file_path`  ☑ DONE (a + b)

**Re-assessment of (a):** Not the "no token at all" the original audit implied. Both exempt routes are already capability-URLs:
- `/api/share/{token}` — full `Uuid::new_v4()` (122-bit, unguessable). The token *is* the auth. **No change needed.**
- `/api/cast-media/{media_id}` — also a UUID, but was **truncated to 8 chars (32 bits)** and removed after the session. **Fixed:** stopped truncating (`cast.rs:144`), so it's now a full 122-bit token. One line, zero downside, Chromecast unaffected (id is opaque end-to-end; only the direct-play branch uses it; SVP/transcode cast paths untouched).

Rejected the doc's heavier "HMAC-signed expiring URL" proposal as over-engineering — the capability-URL design is sound; it just needed a non-truncated token. Optional future defense-in-depth: add a TTL to share sessions (currently live until deleted/restart). Parked, low priority.

---
<!-- original C3 notes retained below for reference -->
### C3 (original notes)  ☑ superseded by the re-assessment above
**Files:** `access_control.rs:29-30` (exempt list), `api/routers/cast.py:137`, `api/services/cast_session.py:542`
**Problem (a):** `/api/cast-media/` and `/api/share/` are fully exempt — anyone on the network downloads media by id, no token.
**Problem (b):** `/api/cast/play` takes a client `file_path` and serves/transcodes it with no check that it's inside a watched directory → arbitrary file read.
**Fix (b) ☑ DONE:** validates every client-supplied `file_path` with `canonicalize()` + "is inside a watch dir" containment check via the shared `validate_path_in_watch_dir` helper (see H6). Reject otherwise.
**Fix (a) — DECISION:** Chromecast/DLNA devices fetch the media URL directly and *can't* send a JWT header, which is why these are exempt. Options:
- Per-cast-session signed, expiring URL token (path carries an HMAC of media-id+expiry). Device needs no login; link dies after the session. Recommended.
- IP-allowlist cast-media to the LAN only + short TTL.
- Leave share/cast-media exempt but gate the *control* endpoints (`/api/cast/play`) behind the normal auth (they're not exempt today, so a JWT is already required for control — only the media fetch is open).
**Constraint check:** Casting still works (device fetches a signed URL); LAN/Android unaffected.

---

## HIGH

### H4 — Path traversal in SVP sidecar stream endpoint  ☑ DONE
**File:** `addons/svp/app.py:930` (`stream_file`)
**Problem:** `file_path = stream.hls_dir / filename` with no containment check → `..%2f..%2fetc/passwd`.
**Fix:** resolve and assert the path stays under `stream.hls_dir`; reject `..` and absolute. ~4 lines.
**Constraint check:** ✅ none of the three affected. Pure hardening.

### H5 — Login rate-limit bypass via X-Forwarded-For  ☑ DONE
**File:** `src-tauri/src/routes/users.rs:107` (`extract_client_ip`)
**Problem:** rate-limit key trusts spoofable `X-Forwarded-For`; rotate header → unlimited brute force. (Access-control itself correctly uses `ConnectInfo` — only the limiter is fooled.)
**Fix:** only honor `X-Forwarded-For` when the direct `ConnectInfo` IP is loopback (i.e. behind the app's own local proxy); otherwise use `ConnectInfo`. Safe default for the no-proxy case.
**Constraint check:** ✅ none affected. Localhost still trusted; LAN clients rate-limited by real IP.

### H6 — Unvalidated `file_path` → ffmpeg/ffprobe  ☑ DONE (+ C3b)
**Files:** `src-tauri/src/server/utils.rs` (new shared helper), `routes/cast.rs`, `routes/settings.rs`
**Problem:** Endpoints take a client path and hand it to ffmpeg/ffprobe (or serve it) with no watch-dir check → arbitrary file probe/decode/read.
**Fix (done):** Added `validate_path_in_watch_dir(conn, client_path)` to `server/utils.rs` (canonicalize + `starts_with` a watch dir; reuses the exact check from `images/single.rs`). Wired into **all 7 sinks**: `cast/play`, `video-info`, `dimensions`, `audio-gain`, transcode `play`, optical-flow interpolated stream, whisper generate. Three handlers (`video-info`/`dimensions`/`audio-gain`) gained a `State<AppState>` param for DB access. Validation runs *after* existing availability/exists checks so an offline drive still reports its own error. Compiles clean, no warnings.
**Constraint check:** ✅ none affected (legit calls always pass in-library paths). C3b (cast `file_path` validation) is covered by the same helper.

### H2 — Android WebView hardening  ☐
**File:** `MainActivity.kt:20, 27, 33`
- **H2a ☑ DONE:** `setWebContentsDebuggingEnabled(true)` now gated behind `if (BuildConfig.DEBUG)` (`buildConfig = true` already set in app/build.gradle.kts; namespace matches package so no import needed). ✅ no constraint impact.
- **H2b — BY DESIGN, keep:** `mixedContentMode = MIXED_CONTENT_ALWAYS_ALLOW` is **required** by constraint 4 — the WebView (`https://tauri.localhost`) talks to the local `http://127.0.0.1:8790` server and to remote LAN servers over http. Changing it breaks http connections. No change.
- **H2c (leave):** `mediaPlaybackRequiresUserGesture=false` is needed for autoplay; low risk. No change.

### H1 — JWT token in URL query string  ☑ DONE
**Implemented (short-lived media-scoped token):**
- `auth.rs` — added `scope: Option<String>` to `Claims` (`#[serde(default)]` keeps old tokens valid), `MEDIA_SCOPE`, `Claims::is_media_scoped()`, `create_media_jwt()` (24h, `can_write=false`, `scope="media"`), `MEDIA_TOKEN_TTL_SECS`. `AuthUser` now rejects media-scoped tokens, so they can never authenticate a real handler.
- `routes/users.rs` — new `GET /api/users/media-token` (guarded by `AuthUser`, i.e. requires a full session JWT) returns `{token, expires_in}`. Added to `LOCALHOST_EXEMPTIONS` so JWT-authenticated LAN devices can mint one (handler's `AuthUser` still enforces auth).
- `access_control.rs` — `?token=` query tokens: full tokens grant access as before; **media-scoped tokens only honored for `GET /api/images/...`** (the one media path that needs a token — streams/cast-media/share/`/thumbnails`/`/watch` are already exempt). Bearer header accepts full tokens only.
- `server/mod.rs` — added `Referrer-Policy: no-referrer` so a token in a URL can't leak via `Referer`.
- `frontend/src/api.js` — `fetchMediaToken()` mints a media token after pairing; `getMediaUrl()` (non-Tauri browser path) uses it in `?token=` instead of the 30-day session JWT, refreshes in background near expiry, falls back to the session token only in the brief pre-mint window. QR/session flow untouched — refresh is invisible (no rescanning).

_Original analysis:_
**Files:** `frontend/src/api.js:880`, `access_control.rs:219`
**Problem:** `?token=` appended to `<img>`/`<video>` src so the browser can authenticate media loads. **Note:** under the http transport, the token already crosses the wire in cleartext, so the network-sniffing angle is already covered by the accepted residual risk. What remains is the *local* leak surface: server logs, browser history, and `Referer` headers retaining the full token.
**Why it's hard:** `<img>`/`<video>` can't send `Authorization` headers, so *some* in-URL credential is needed for media.
**Fix options (lower priority now):**
- **Cheap win:** set `Referrer-Policy: no-referrer` (already partly mitigated since requests are same-origin to the local proxy) and keep media-token URLs out of access logs. Low effort.
- Issue a **short-lived, media-scoped** token for src URLs (minutes TTL, read-only) instead of the main session JWT — limits blast radius if a URL leaks. Moderate work.
- `HttpOnly` cookie media auth — cleanest, larger change.
**DECISION NEEDED:** whether this is worth doing now given the trusted-network model. Recommend deferring to the M/L batch unless you want the short-lived media token.

### H3 — Tauri CSP + asset-protocol scope  ☑ DONE (CSP + asset-scope)

**Asset scope — DONE (dynamic, grant-only):** `tauri.conf.json` scope changed `["**"]` → `[]`. `AppState` holds a clone of `app.asset_protocol_scope()` (Tauri 2.10.1 `Scope` is `Clone` with shared `Arc<Mutex>` state) + `allow_asset_dir()`. On startup (`lib.rs` setup) every existing watch dir is granted; on add — `routes/directories/mod.rs` (both INSERT sites) and `services/directory_watcher.rs` (auto-discovered subdirs) — the new dir is granted (`recursive`). **No runtime revoke on delete**: Tauri's scope API is append-only and `forbid` is permanent (would poison re-adds), so a removed dir stays `asset://`-readable until next launch (config scope `[]` clears it on relaunch). Surface is our own webview only, not a network endpoint.

_Previous status:_ ☑ CSP DONE (smoke-test passed) · asset-scope DEFERRED
**CSP (done, smoke-test passed: app loads, images/video render, cast works):** `tauri.conf.json:28` — removed `script-src 'unsafe-inline'` (now `script-src 'self'`), added `object-src 'none'`, `base-uri 'self'`, `frame-ancestors 'none'`. Kept `style-src 'unsafe-inline'` (React) and all `http:`/`asset:` sources (constraint 4). Tauri v2 auto-nonces its bootstrap + hashes the frontend's inline scripts, so this should hold — **must verify with a build+launch smoke-test** (app loads, image + video render, cast works, no CSP errors in console). Revert script-src if it white-screens.
**Asset scope `["**"]` — DEFERRED (tracked):** `asset://` is actively used to serve local media (`getAssetUrl` → `Lightbox.jsx:962`, `useVideoStreaming.js:1217`) from user-configured watch dirs. Narrowing requires dynamic per-watch-dir scoping tied to the watch-dir add/remove lifecycle — feature-sized, separate task. Left at `["**"]` for now.
**File:** `src-tauri/tauri.conf.json:28-32`
**Keep `http:` in `connect-src`/`img-src`/`media-src`** — required by constraint 4 (talking to http servers). The only safe-to-tighten parts:
- **asset scope `["**"]`:** whole-filesystem read via `asset://`. Narrow to watch dirs — but they're user-configured/dynamic, and we must confirm media even goes through `asset:` vs the http server. **DECISION:** I'll trace whether `asset:` is actually used; if media is all via `http://127.0.0.1:8790`, drop `asset:` from CSP and tighten scope.
- **CSP `script-src 'unsafe-inline'`:** weakens XSS defense, unrelated to transport. Removing may break the Vite/React bundle. **DECISION:** test a build with it removed from `script-src` (keep for `style-src` if needed); adopt only if the app still loads.
**Constraint check:** transport untouched; needs a build smoke-test before committing.

---

## MEDIUM / LOW (batch later)

- **M1 — Addon Python deps unpinned**  ☐ `src-tauri/src/addons/manifest.rs:20` — pin versions / add hashes. Supply-chain. No functional impact.
- **M2 — No sidecar resource limits / no `/api/share/create` rate limit**  ☐ — DoS hardening.
- **M3 — Family-mode PIN brute force**  ☐ `routes/settings.rs` — add backoff/lockout on `unlockFamilyMode()`.
- **M4 — CORS `.allow_headers(Any)`**  ☐ `server/mod.rs:49` — explicit header allowlist.
- **L1 — VapourSynth path string-interpolation**  ☐ `svp_integration.py:190` — use `repr()`; current escaping holds but brittle.

---

## Corrections to the original audit (not bugs)
- "Tailscale private key committed to repo" — **false**. `archlinux.tailb63b0e.ts.net.key/.crt` are gitignored and untracked. No action.
- Several endpoints described as reachable "without authentication" — they require a JWT; the real issue is C2 (a normal token is over-privileged), not missing auth.

---

## Suggested order
1. **C2** (clean, highest impact, zero tradeoff) →
2. **H4, H5, H6, C3(b), H2a** (auth/authorization + path hardening, no tradeoffs, all work fine over http) →
3. Decisions: **C3(a)** (cast tokens), **H3** (CSP/asset — needs build smoke-test), **H1** (media token, optional) →
4. **M/L batch.**

**Closed as by-design (constraint 4):** C1 (accept-invalid-certs), H2b (mixed-content ALWAYS_ALLOW), and any "force HTTPS/HSTS/reject http" items from the original audit. Transport is trusted-network, not TLS.
