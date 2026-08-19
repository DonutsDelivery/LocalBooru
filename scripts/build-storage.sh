#!/usr/bin/env bash
# Shared host-side guard for LocalBooru's high-I/O build paths.

localbooru_assert_ssd_path() {
  local path="$1"
  local label="$2"
  local probe source rotational

  command -v findmnt >/dev/null 2>&1 || {
    echo "ERROR: findmnt is required to verify $label storage" >&2
    return 1
  }
  command -v lsblk >/dev/null 2>&1 || {
    echo "ERROR: lsblk is required to verify $label storage" >&2
    return 1
  }

  probe="$(realpath -m -- "$path")"
  while [[ ! -e "$probe" ]]; do
    [[ "$probe" != "/" ]] || break
    probe="$(dirname "$probe")"
  done

  source="$(findmnt -n -o SOURCE -T "$probe")"
  rotational="$(lsblk -s -n -o ROTA "$source" | tr -d '[:space:]')"
  if [[ -z "$rotational" || "$rotational" == *1* ]]; then
    echo "ERROR: $label must be on non-rotational storage: $path (source: $source)" >&2
    return 1
  fi
}
