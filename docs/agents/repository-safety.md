# Repository Safety and User-Data Boundary

This repository contains source code, documentation, tests, and deliberately
synthetic fixtures. It must not contain private data from a user's LocalBooru
installation or library.

## Prohibited repository contents

Never copy, back up, export, snapshot, or commit a user's library or private
files into the repository or any worktree. The boundary includes, but is not
limited to:

- SQLite databases and sidecars (`*.db`, `*.db-wal`, `*.db-shm`, and
  `*.db-journal`);
- thumbnails, source media, imported files, and generated previews;
- library metadata, settings, logs, caches, and recovery/backup directories;
- credentials, tokens, keys, personal paths, and generated personal data.

This applies to agents, scripts, tests, fixtures, build acceptance, release
validation, and debugging. Do not use the application's real data directory as
a test fixture or copy its contents into a worktree.

## Safe testing and staging

- Use synthetic fixtures or disposable test data in a temporary directory
  outside the repository.
- Before staging or committing, inspect `git status --short` and the staged path
  list for private data and user-specific paths.
- `.gitignore` is defense in depth; an ignored path is not permission to copy
  private data into the repository.
- If private data appears in the working tree or index, stop and remove it from
  the index before committing. If it was committed, stop and report the exact
  commit/path so repository history can be handled deliberately.

## Isolated local-app testing

Use the disposable isolated launcher for local desktop testing:

```bash
npm run app:test-isolated
```

It assigns a temporary `HOME`, XDG config/data/state roots, `TMPDIR`, Cargo
target directory, `LOCALBOORU_PORTABLE_DATA`, and an unused embedded HTTP port.
It also disables only that test process's single-instance forwarding, so it
cannot hand its launch or files to the user's normal LocalBooru instance. The
temporary root is removed when the launcher exits. Set
`LOCALBOORU_TEST_PORT` when the default test port is occupied.

For acceptance of an already-built candidate, keep the canonical isolated
launcher and set an absolute executable path:

```bash
LOCALBOORU_TEST_BINARY=/absolute/path/to/candidate npm run app:test-isolated
```

This runs the candidate directly without invoking the development launcher or
starting a build. It applies the same disposable home, XDG config/data/cache/state,
temporary files, portable library, target directory, port, and disabled
single-instance forwarding. The candidate receives no application file arguments.
The launcher also exports `LOCALBOORU_ISOLATED_TEMP_DIR`; isolated media
transcodes, upload/download staging and snapshots use that explicit directory
and reject a missing or redirected root.
Candidate adapters must preserve this variable rather than rely on an SDK's
system temporary-directory selection. It is separate from the normal app's
temporary cache and must remain inside the disposable profile.
Its output stays in the temporary test log, and the launcher removes the profile
and stops its child if readiness fails. Without this option the launcher retains
its normal development behavior and host build-gate requirement. Never point the
portable data override at a user profile, use the normal installed launcher, or
fall back to the user's running instance for acceptance.

Synthetic launcher checks can be run without a build:

```bash
bash scripts/test-run-dev.sh
bash scripts/test-run-isolated-test.sh
```

On Linux, the launcher requires `dbus-run-session` and `setsid`. It starts both
candidate and development acceptance under a private D-Bus session while
preserving display connection variables. This prevents GTK's application identity
from registering in the user's desktop session. Cleanup targets only the launcher's
owned process group, including the candidate and private bus; missing session tools
fail closed. Portable and isolated credential storage also bypasses shared OS
keyring reads and writes, using only the owned profile's protected credential file.
An isolated process without an explicit portable data directory refuses credential
access before resolving any normal profile path.

Native isolated candidates also disable autostart and the normal window-state
plugin and send application logs only to the launcher's stdout file. macOS and
Windows use an ephemeral WebView store; Linux retains its private XDG store so
GTK automation remains attached to the correct WebContext. This is essential on macOS, where system-managed log,
window-state and WKWebView paths do not follow XDG/HOME overrides. These test
runs do not establish persistence of WebView localStorage or normal window
geometry across launches. Backend persistence remains under the owned portable
profile. Do not launch a sandboxed Mac candidate until its owned data-directory
access has been verified; sandbox denial is not permission to fall back to the
normal app profile or disable the sandbox.

For a prebuilt sandboxed Mac Lite candidate, set both
`LOCALBOORU_TEST_MACOS_APP` to the verified `.app` and
`LOCALBOORU_TEST_MACOS_CONTAINER_PARENT` to its Foundation-resolved container
Application Support directory, after native preflight has verified access.
Keep using `npm run app:test-isolated` with `LOCALBOORU_TEST_BINARY`.
The launcher creates a fresh private directory there, keeps all test data beneath
it, and checks its owner, permissions and filesystem identity before cleanup.
It never opens the ordinary library. A wrong edition, parent path, symlink,
changed identity or unsafe permissions fail closed. Do not create the parent by
guessing another app's container, grant broad sandbox exceptions, or use this
option as evidence of Powerbox/bookmark/media-helper acceptance.
