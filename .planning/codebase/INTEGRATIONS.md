# External Integrations

**Analysis Date:** 2026-01-15

## APIs & External Services

**Model Downloads (Hugging Face):**
- Hugging Face Hub - AI model downloads for tagging
  - Integration: HTTP via httpx (`api/services/model_downloader.py`)
  - Auth: None required (public models)
  - Models downloaded:
    - `https://huggingface.co/SmilingWolf/wd-vit-tagger-v3/`
    - `https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3/`
    - `https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3/`

**GitHub Releases:**
- GitHub API - Auto-updates for desktop app
  - Integration: electron-updater (`electron/updater.js`)
  - Auth: None (public releases)
  - Repository: DonutsDelivery/LocalBooru
  - Config: `package.json` build.publish.provider = "github"

## Data Storage

**Databases:**
- SQLite - Local single-user database
  - Connection: File path from `api/config.py`
  - Client: SQLAlchemy 2.0 async with aiosqlite
  - Location: `~/.localbooru/library.db` or portable `data/` folder
  - Mode: WAL (Write-Ahead Logging) for concurrent reads

**File Storage:**
- Local filesystem - Image storage
  - Watch directories configured via API
  - Thumbnails: `~/.localbooru/thumbnails/`
  - Portable mode: `data/` folder next to executable

**Caching:**
- In-memory model cache (`api/services/tagger.py`)
- LRU cache for settings (`api/config.py`)
- No Redis or external cache

## Authentication & Identity

**Auth Provider:**
- Custom password auth (`api/routers/users.py`)
  - Implementation: PBKDF2 password hashing with salt
  - Storage: password_hash in SQLite users table
  - Session: No token-based sessions (TODO)

**Access Control:**
- IP-based access levels (`api/middleware/access_control.py`)
  - Levels: localhost, local_network, public
  - Enforcement: Middleware checks request IP

**OAuth Integrations:**
- None configured

## Monitoring & Observability

**Error Tracking:**
- None (no Sentry or similar)

**Analytics:**
- None

**Logs:**
- stdout/stderr only
- Electron: File log at `app.getPath('userData')/debug.log`

## CI/CD & Deployment

**Hosting:**
- Local desktop application (Electron)
- No server deployment

**Distribution:**
- GitHub Releases for desktop builds
- Capacitor for mobile builds (Android/iOS)

**CI Pipeline:**
- GitHub Actions (`.github/workflows/`)

## Environment Configuration

**Development:**
- Required env vars: None strictly required (has defaults)
- Optional: `LOCALBOORU_*` prefix vars (`api/config.py`)
- No `.env.example` file

**Production:**
- Portable mode: Data folder next to executable
- AppData mode: Standard OS user data location
- Config via Pydantic Settings with env var support

## Network Services

**UPnP Port Forwarding:**
- miniupnpc - Automatic router port mapping
  - Integration: `api/services/network.py`
  - Purpose: Allow internet access to local booru
  - Optional feature

**Local Network:**
- LAN access configurable (`api/routers/network.py`)
- QR code generation for mobile app connection

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- None

## Internal IPC

**Electron to Backend:**
- Backend spawned as subprocess (`electron/backendManager.js`)
- Health checks via HTTP (`/health` endpoint)
- IPC channels for:
  - `updater:check`, `updater:download`, `updater:install`
  - Directory picker, clipboard, window controls

**Frontend to Backend:**
- REST API via axios (`frontend/src/api.js`)
- SSE for real-time events (`/api/library/events`)
- Base URL: `/api` or `http://127.0.0.1:8790/api`

---

*Integration audit: 2026-01-15*
*Update when adding/removing external services*
