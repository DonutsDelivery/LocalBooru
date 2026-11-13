# Media helper folder grants

`media-folder-grants.c` is linked directly into each bundled ffmpeg and ffprobe
executable with CoreFoundation, for each architecture. Native initialization
records the private folder-grants store in the app container. Media command
creation passes that store as `DMC_MEDIA_BOOKMARK_DIRECTORY` only for macOS Lite;
all other editions explicitly remove the transfer variable. The parent remains
responsible for restoring grants and reporting failed grants to the user. The
variable is removed before helper main; absence is a no-op for standalone and
full-edition helpers. The directory must already exist and must not be a final
symlink. It is an input channel for existing bookmarks, not authorization to
select a folder. Sandbox inheritance transfers static entitlements, not later
Powerbox access; the bookmark itself must be transferred to the child. See
[Apple's App Sandbox inheritance documentation](https://developer.apple.com/library/archive/documentation/Miscellaneous/Reference/EntitlementKeyReference/Chapters/EnablingAppSandbox.html).

The constructor reads valid `[A-Za-z0-9-]+.bookmark` entries without following
final symlinks, with limits of 1024 entries, 1 MiB per entry and 16 MiB total.
Unreadable or unsafe matching entries, directory errors, limit violations and
resource failures terminate the helper before main with a generic diagnostic and
exit status 78. Unresolvable, stale or denied individual bookmarks are skipped;
the parent owns recovery. Successful scopes remain active until normal process
exit; bookmarks are never modified. Abrupt process termination relies on OS
process cleanup.

Native owned-fixture validation remains required for both helpers and
architectures. Validate access to an explicitly granted folder, denial outside
it, independent invalid/stale references, malformed stores and limits, variable
removal, and standalone no-variable behavior under the shipping sandbox and
signing configuration. This component alone does not establish containment or
change entitlements.
