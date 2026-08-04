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
