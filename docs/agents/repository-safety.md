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
