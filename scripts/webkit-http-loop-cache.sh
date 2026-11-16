#!/usr/bin/env bash
# Shared read-only recognition and exact-source upgrade for one reviewed cache.

localbooru_webkit_http_loop_stack_matches() {
  local source="$1" primary_hash="$2" upgrade_patch="$3"
  local expected previous
  expected="$(awk '$2 == "primary-patch" {print $3}' "$upgrade_patch")"
  previous="$(awk '$2 == "previous-stack" {print $3}' "$upgrade_patch")"
  [[ "$primary_hash" == "$expected" && -f "$source/.localbooru-patch-$previous" ]]
}

localbooru_webkit_http_loop_upgrade_available() {
  local source="$1" primary_hash="$2" upgrade_patch="$3" helper="$4"
  localbooru_webkit_http_loop_stack_matches "$source" "$primary_hash" "$upgrade_patch" || return 1
  python3 "$helper" --source "$source" --patch "$upgrade_patch" >/dev/null
}

localbooru_webkit_prepare_http_loop() {
  local source="$1" primary_hash="$2" upgrade_patch="$3" helper="$4" runtime_stamp="$5"
  local expected status
  expected="$(awk '$2 == "primary-patch" {print $3}' "$upgrade_patch")"
  [[ "$primary_hash" == "$expected" ]] || return 0
  status="$(python3 "$helper" --source "$source" --patch "$upgrade_patch")" || return 1
  if [[ "$status" != "already upgraded" ]]; then
    # Invalidate completion before mutation; an interrupted upgrade must rebuild.
    rm -f "$runtime_stamp"
    python3 "$helper" --source "$source" --patch "$upgrade_patch" --apply
  fi
}
