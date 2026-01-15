# Technology Stack

**Analysis Date:** 2026-01-15

## Languages

**Primary:**
- TypeScript/JavaScript - Frontend, Electron desktop app (`frontend/`, `electron/`)
- Python 3.10+ - Backend API (`api/`)

**Secondary:**
- CSS - Component styling (`frontend/src/*.css`)
- HTML - Entry template (`frontend/index.html`)

## Runtime

**Environment:**
- Node.js 20+ - Electron and frontend build tools
- Python 3.10+ - FastAPI backend server
- Electron 33.x - Desktop application runtime

**Package Manager:**
- npm - Node packages (`package.json`, `frontend/package.json`)
- pip - Python packages (`requirements.txt`)
- Lockfile: `package-lock.json` present

## Frameworks

**Core:**
- React 19.2 - Frontend UI framework (`frontend/package.json`)
- FastAPI 0.115.0+ - Python async web API (`api/main.py`)
- Electron 33.0 - Desktop wrapper (`electron/main.js`)
- Capacitor 8.0.1 - Mobile app framework (`frontend/capacitor.config.ts`)

**Testing:**
- No formal test framework configured
- ESLint 9.39.1 - Frontend linting (`frontend/eslint.config.js`)

**Build/Dev:**
- Vite 7.2.4 - Frontend bundling (`frontend/vite.config.js`)
- electron-builder 25.0 - Desktop packaging (`package.json`)

## Key Dependencies

**Critical:**
- SQLAlchemy 2.0+ - Async ORM for database (`api/database.py`)
- ONNX Runtime 1.18+ - ML model inference for tagging (`api/services/tagger.py`)
- axios 1.13+ - HTTP client (`frontend/src/api.js`)
- react-router-dom 7.10+ - Client routing (`frontend/package.json`)

**Infrastructure:**
- Uvicorn 0.32+ - ASGI server (`requirements.txt`)
- aiosqlite 0.20+ - Async SQLite driver (`api/database.py`)
- Pillow 10.0+ - Image processing (`requirements.txt`)
- watchdog 4.0+ - File system monitoring (`api/services/directory_watcher.py`)

**AI/ML:**
- insightface 0.7.3+ - Face detection, age estimation (`api/services/age_detector.py`)
- opencv-python-headless 4.8+ - Computer vision fallback (`api/services/age_detector.py`)
- numpy 1.24+ - Numerical computing (`requirements.txt`)

**Desktop:**
- electron-updater 6.6.2 - Auto-updates via GitHub (`electron/updater.js`)
- chokidar 3.6+ - File watcher for Electron (`electron/directoryWatcher.js`)

## Configuration

**Environment:**
- Pydantic BaseSettings with `LOCALBOORU_` prefix (`api/config.py`)
- `.env` file support at project root
- No `.env.example` file present

**Build:**
- `frontend/vite.config.js` - Vite build configuration
- `frontend/capacitor.config.ts` - Mobile app configuration
- `package.json` - Electron-builder settings

## Platform Requirements

**Development:**
- macOS/Linux/Windows with Node.js 20+ and Python 3.10+
- No Docker required for development

**Production:**
- Distributed as Electron app via GitHub Releases
- Portable mode (data folder next to exe) or AppData mode
- Mobile: Capacitor builds for Android/iOS

---

*Stack analysis: 2026-01-15*
*Update after major dependency changes*
