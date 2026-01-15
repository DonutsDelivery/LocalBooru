# Coding Conventions

**Analysis Date:** 2026-01-15

## Naming Patterns

**Files:**
- Python: snake_case (`task_queue.py`, `directory_watcher.py`, `age_detector.py`)
- React components: PascalCase with .jsx (`Header.jsx`, `Lightbox.jsx`, `MediaItem.jsx`)
- JS utilities: camelCase with .js (`api.js`, `serverManager.js`)
- CSS: Match component name (`Header.css`, `Sidebar.css`)

**Functions:**
- Python: snake_case (`get_settings()`, `list_directories()`, `init_db()`)
- JavaScript: camelCase (`fetchImages()`, `handleSubmit()`, `getApiUrl()`)
- React handlers: camelCase with "handle" prefix (`handleAddDirectory`, `handleClear`)

**Variables:**
- Python: snake_case (`current_server_url`, `watch_directory_id`)
- JavaScript: camelCase (`menuOpen`, `isLoading`, `scanning`)
- Constants: UPPER_SNAKE_CASE (`DEFAULT_MODEL`, `STARTUP_GRACE_PERIOD`, `ALL_RATINGS`)

**Types:**
- Python classes: PascalCase (`Settings`, `WatchDirectory`, `ImageFile`)
- Python enums: PascalCase class, lowercase values (`TaskStatus.pending`, `Rating.pg13`)
- Private members: underscore prefix (`_models`, `_tags_data_cache`)

## Code Style

**Formatting:**
- Python: 4 spaces indentation (PEP 8)
- JavaScript: 2 spaces indentation
- No Prettier config (consistent 2-space indent in JS)
- Single quotes in JavaScript
- Double quotes in Python strings

**Linting:**
- ESLint 9.39.1 for frontend (`frontend/eslint.config.js`)
- Plugins: react-hooks, react-refresh
- Rule: `no-unused-vars` with varsIgnorePattern for uppercase
- No Python linting config (flake8, pylint not configured)

**Line Length:**
- Generally 100-120 characters
- No strict enforcement

## Import Organization

**Python:**
1. Standard library imports
2. Third-party imports (fastapi, sqlalchemy, pydantic)
3. Local imports (from ..config, from .models)

**JavaScript:**
1. React imports
2. Third-party imports (axios, react-router-dom)
3. Local imports (./api, ./components/*)

**Path Aliases:**
- None configured (relative imports used)

## Error Handling

**Python Patterns:**
- FastAPI HTTPException for client errors
- Generic `except Exception as e` in batch operations
- Some bare `except:` clauses (needs improvement)
- Database migrations use silent exception handlers

**JavaScript Patterns:**
- try/catch with error display
- Axios interceptors for API errors
- Startup grace period for backend connection

**Error Types:**
- Throw HTTPException with status code and detail message
- Log errors before re-raising in critical paths

## Logging

**Framework:**
- Python: Mixed print() and logging module
- JavaScript: console.log throughout
- Electron: File-based debug.log (`app.getPath('userData')/debug.log`)

**Patterns:**
- `print()` statements for debugging (should use logger)
- Logging at service boundaries
- No structured logging framework

## Comments

**When to Comment:**
- Module-level docstrings in Python files
- Section comments with equals (`# =============`)
- Explain business logic (rating adjustment rules in tagger.py)
- Inline comments for non-obvious code

**Docstrings:**
- Triple-quoted strings for Python functions/classes
- JSDoc not consistently used in JavaScript

**TODO Comments:**
- Format: `# TODO: description` or `// TODO: description`
- Found in: `api/routers/users.py`, `api/services/external_upload.py`

## Function Design

**Size:**
- Large files exist (images.py 1162 lines, tagger.py 655 lines)
- Functions sometimes exceed 50 lines

**Parameters:**
- Python: Type hints with Pydantic models for validation
- JavaScript: Destructured props in components

**Return Values:**
- Explicit returns in Python
- Early returns for guard clauses

## Module Design

**Exports:**
- Python: No __all__ defined, import specific items
- JavaScript: Named exports and default exports mixed
- React components: Default export

**Barrel Files:**
- Not used (no index.js re-exports)

## CSS Patterns

**Variables:**
- CSS custom properties in `:root` (`frontend/src/index.css`)
- Naming: `--bg-primary`, `--text-secondary`, `--accent`
- Color scheme: Dark theme (ComfyUI style)

**Organization:**
- Global styles in `index.css`
- Component styles co-located with components

---

*Convention analysis: 2026-01-15*
*Update when patterns change*
