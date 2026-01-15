# Architecture

**Analysis Date:** 2026-01-15

## Pattern Overview

**Overall:** Three-Tier Desktop Application with Mobile Support

**Key Characteristics:**
- Desktop app (Electron) wrapping a Python backend
- React SPA frontend communicating via REST API
- SQLite database for single-user local storage
- Background task queue for ML processing
- Multi-platform: Desktop, Web, Mobile (Capacitor)

## Layers

**Presentation Layer (React):**
- Purpose: User interface and client-side routing
- Contains: Components, pages, API client (`frontend/src/`)
- Depends on: API layer via HTTP/REST
- Used by: End users via browser or Electron webview

**API Layer (FastAPI Routers):**
- Purpose: HTTP endpoint handlers, request validation
- Contains: Route handlers with Pydantic validation (`api/routers/`)
- Depends on: Service layer for business logic
- Used by: Frontend, mobile app, external clients

**Service Layer:**
- Purpose: Core business logic, ML processing
- Contains: Task queue, tagger, importer, file tracker (`api/services/`)
- Depends on: Data layer, external ML models
- Used by: API routers

**Data Layer (SQLAlchemy):**
- Purpose: Database operations, ORM models
- Contains: Models, database connection (`api/models.py`, `api/database.py`)
- Depends on: SQLite database
- Used by: Service layer, some routers directly

**Desktop Layer (Electron):**
- Purpose: Native desktop integration, backend lifecycle
- Contains: Main process, backend manager, IPC (`electron/`)
- Depends on: Node.js APIs, spawned Python process
- Used by: Desktop users

## Data Flow

**Image Import Flow:**
1. File detected by DirectoryWatcher (`api/services/directory_watcher.py`)
2. Importer validates, hashes, creates DB record (`api/services/importer.py`)
3. TaskQueue enqueues "tag" task (`api/services/task_queue.py`)
4. Tagger processes image with ONNX model (`api/services/tagger.py`)
5. Tags and rating stored in database
6. Frontend refreshes via polling or SSE

**User Search Flow:**
1. User types in SearchBar (`frontend/src/components/SearchBar.jsx`)
2. API call via axios (`frontend/src/api.js`)
3. Router handles request (`api/routers/images.py`)
4. SQLAlchemy query with tag joins
5. JSON response with image metadata
6. MasonryGrid renders results

**State Management:**
- Database: SQLite with WAL mode for concurrent reads
- Frontend: React useState/useEffect, no global state manager
- Settings: JSON file in data directory

## Key Abstractions

**BackgroundTaskQueue:**
- Purpose: Async job processing for ML tasks
- Location: `api/services/task_queue.py`
- Pattern: In-process worker pool with retry logic

**WatchDirectory:**
- Purpose: Monitor filesystem for new images
- Location: `api/services/directory_watcher.py`
- Pattern: Event-driven with debouncing

**Tagger Service:**
- Purpose: AI-powered image tagging
- Location: `api/services/tagger.py`
- Pattern: ONNX model inference with caching

**BackendManager:**
- Purpose: Spawn and manage Python backend
- Location: `electron/backendManager.js`
- Pattern: Process supervisor with health checks

## Entry Points

**Desktop App:**
- Location: `electron/main.js`
- Triggers: User launches app
- Responsibilities: Create window, spawn backend, setup IPC, tray

**API Server:**
- Location: `api/main.py`
- Triggers: Uvicorn startup or BackendManager spawn
- Responsibilities: Register routes, start services, handle lifespan

**Frontend SPA:**
- Location: `frontend/src/main.jsx`
- Triggers: Browser/webview loads page
- Responsibilities: Mount React app, setup routing

## Error Handling

**Strategy:** Exception bubbling to route handlers with HTTP responses

**Patterns:**
- FastAPI HTTPException for client errors
- Generic exception handlers in routers with logging
- Silent exception handlers in database migrations (needs improvement)
- Frontend: try/catch with user-friendly error display

## Cross-Cutting Concerns

**Logging:**
- Backend: Mixed print() and Python logging
- Frontend: console.log for debugging
- Electron: File-based debug.log

**Validation:**
- API: Pydantic models at request boundary
- Frontend: Basic form validation

**Authentication:**
- Password auth with PBKDF2 hashing (`api/routers/users.py`)
- IP-based access control middleware (`api/middleware/access_control.py`)
- No session tokens implemented yet (TODO)

---

*Architecture analysis: 2026-01-15*
*Update when major patterns change*
