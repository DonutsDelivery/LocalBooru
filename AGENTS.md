# LocalBooru Agent Rules

## Repository lifecycle

- Treat `main` as the integration branch. A worktree task is not complete merely because it is committed on its own branch.
- After verification, integrate the product commit into `main` or report an explicit blocker. Do not leave a clean, approved candidate stranded indefinitely.
- Remove a task worktree only after its commits are ancestors of `main` and its working tree is clean. Never delete dirty or unmerged worktrees.
- Preserve unrelated changes. Stage explicit paths only.
- Do not push, publish, or rewrite history unless the user explicitly asks.

## Build and runtime locks

- Every compiler and Docker build participates in the host-wide heavy-build gate shared with DonutStudio and Jak X. The LocalBooru project lock is a compatibility alias to that host token.
- Use `scripts/run-cargo.sh` for one-shot Cargo builds, checks, and tests. Do not invoke heavy `cargo` commands directly.
- Use `./run-dev.sh` for the long-lived development app. It owns only the duplicate-dev lock; merely leaving the app open must not block release builds.
- Development `rustc` invocations acquire the host token individually, so an idle app owns no build lock while hot recompilation cannot overlap another project's heavy build.
- Use the project release wrappers for Docker builds. Do not call `docker build`, `cargo tauri build`, or container build scripts directly.
- Local ad-hoc Cargo commands are capped at two jobs by `.cargo/config.toml`; dev hot rebuilds are capped at one. Do not raise build jobs without checking active builds and available memory.
- Never run a regular Cargo build and a Docker release build concurrently. If a build gate is occupied, wait or stop the conflicting build; do not bypass or delete lock files.

## Verification

- Run focused tests for the changed surface. Do not start a broad build merely to verify documentation or workflow changes.
- Before reporting build completion or starting another build, inspect live Cargo/Rust/Docker processes and confirm the previous writer exited.
