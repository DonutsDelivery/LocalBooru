# LocalBooru Agent Rules

## OpenAI/Codex harness reasoning (including T3 Code)

These rules apply to OpenAI agents in the Codex harness. Other harnesses must
use their own supported controls; do not send them OpenAI reasoning parameters.

- Use `high` reasoning for new delegated OpenAI agents. Do not select `xhigh`, `max`, or `ultra`, or escalate to another model, without explicit user authorization for that work.
- With `collaboration.spawn_agent`, set `reasoning_effort: "high"` explicitly when overrides are supported. Use `fork_turns: "none"` or a bounded history fork with a self-contained task; full-history forks inherit the parent setting and reject overrides. Inherit only when the parent is confirmed to use `high`.
- Preserve already running agents. Do not kill or restart them merely to correct their reasoning setting.
- Main-session reasoning is controlled by the T3/Codex runtime. These Markdown instructions do not change the runtime setting. Report a mismatch and use the supported spawn controls for new agents; do not claim runtime enforcement from a prompt rule.
- For kspec daemon dispatch, use the daemon's worker configuration rather than substituting `collaboration.spawn_agent`. Verify that harness's supported setting before claiming it is applied.

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
