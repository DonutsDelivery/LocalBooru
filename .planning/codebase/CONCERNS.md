# Codebase Concerns

**Analysis Date:** 2026-01-15

## Tech Debt

**Silent Exception Handlers in Database Migrations:**
- Issue: Multiple bare `except Exception: pass` blocks in migration code
- Files: `api/database.py` (lines 82-138)
- Why: Quick fix for "column already exists" errors during development
- Impact: Masks real errors, makes debugging difficult
- Fix approach: Catch specific `sqlalchemy.exc.OperationalError`, add logging

**TODO: Session Token Implementation:**
- Issue: Login returns user data but no secure session token
- File: `api/routers/users.py:269`
- Why: MVP implementation, never completed
- Impact: No proper authentication for subsequent requests
- Fix approach: Implement JWT or session tokens

**TODO: External Upload Feature Incomplete:**
- Issue: Upload logic only creates placeholder records
- File: `api/services/external_upload.py:38`
- Why: Feature not fully implemented
- Impact: External booru upload doesn't work
- Fix approach: Implement upload logic per booru type

## Known Bugs

**No critical bugs identified during analysis**

## Security Considerations

**Weak Password Requirements:**
- Risk: Minimum password length is only 4 characters
- Files: `api/routers/users.py:129-130, 192-193`
- Current mitigation: None
- Recommendations: Increase to 8+ characters, add complexity requirements

**Missing Session Token:**
- Risk: No secure auth mechanism after login
- File: `api/routers/users.py:269`
- Current mitigation: IP-based access control only
- Recommendations: Implement JWT with expiration

**CORS Allow All Origins:**
- Risk: CORS set to `allow_origins=["*"]`
- File: `api/middleware/access_control.py:77`
- Current mitigation: Access control middleware checks IP
- Recommendations: Restrict origins in production if exposed to internet

## Performance Bottlenecks

**No significant bottlenecks identified**

- Database uses WAL mode for concurrent reads
- Task queue has configurable concurrency
- Model caching in tagger service

## Fragile Areas

**Database Migration Logic:**
- File: `api/database.py:82-138`
- Why fragile: Silent exception handling hides failures
- Common failures: New migrations may silently fail
- Safe modification: Add explicit error handling with logging
- Test coverage: None

**Rating Adjustment Logic:**
- File: `api/services/tagger.py:39-132`
- Why fragile: Complex tag-based rules with unclear precedence
- Common failures: Unexpected rating assignments
- Safe modification: Add comprehensive comments, consider config file
- Test coverage: None

## Scaling Limits

**SQLite Single-User:**
- Current capacity: Single user, local storage only
- Limit: Not designed for multi-user concurrent writes
- Symptoms at limit: Database lock errors
- Scaling path: Would need PostgreSQL for multi-user

**Task Queue In-Process:**
- Current capacity: Configurable workers (default 2)
- Limit: CPU-bound by ONNX inference
- Symptoms at limit: Task backlog grows
- Scaling path: Consider Celery/Redis for distributed processing

## Dependencies at Risk

**No immediate concerns identified**

- Core dependencies (FastAPI, SQLAlchemy, React) actively maintained
- ONNX Runtime stable
- Electron well-supported

## Missing Critical Features

**Automated Testing:**
- Problem: No unit, integration, or e2e tests
- Current workaround: Manual testing
- Blocks: Confident refactoring, CI/CD quality gates
- Implementation complexity: Medium (pytest setup, test fixtures)

**Environment Documentation:**
- Problem: No `.env.example` file
- Current workaround: Read `api/config.py` for available options
- Blocks: Easy developer onboarding
- Implementation complexity: Low

## Test Coverage Gaps

**Authentication System:**
- What's not tested: Password hashing, verification, access control
- Risk: Security regressions undetected
- Priority: High
- Difficulty: Low (pure functions, easy to unit test)

**Task Queue Processing:**
- What's not tested: Background job execution, retry logic
- Risk: Silent failures in image processing
- Priority: High
- Difficulty: Medium (need async test setup)

**Image Import Pipeline:**
- What's not tested: File hashing, database insertion, tag assignment
- Risk: Data corruption or loss
- Priority: High
- Difficulty: Medium (need file fixtures)

**Rating Adjustment Logic:**
- What's not tested: Tag-based rating elevation rules
- Risk: Incorrect content ratings
- Priority: Medium
- Difficulty: Low (input/output tests on tag sets)

## Code Quality Issues

**Mixed Logging Approaches:**
- Issue: print() statements mixed with logging module
- Files: Throughout `api/` directory
- Impact: Can't control log levels, no structured logging
- Fix approach: Standardize on Python logging with consistent format

**Large Files:**
- Issue: Several files exceed 500 lines
- Files: `api/routers/images.py` (1162), `api/services/tagger.py` (655), `electron/backendManager.js` (603)
- Impact: Harder to understand and maintain
- Fix approach: Extract related functions into separate modules

**Bare Exception Handlers:**
- Issue: Some `except:` clauses catch all exceptions
- Files: `api/services/importer.py:45,52`, `api/routers/settings.py:54`
- Impact: Hides errors, makes debugging difficult
- Fix approach: Catch specific exceptions

---

*Concerns audit: 2026-01-15*
*Update as issues are fixed or new ones discovered*
