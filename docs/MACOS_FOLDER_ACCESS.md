# macOS Lite folder access

The lite build selects App Sandbox entitlements for user-selected read/write
folders, app-scoped bookmarks, and client/server networking. Full builds retain
their existing folder picker and storage. Lite does not install a LaunchAgent.

The lite native picker creates a security-scoped bookmark from its selection,
resolves it, starts access, then atomically writes it under the lite application's
`folder-grants` data directory before returning success. There is no command
accepting a browser-provided path or bookmark. Bookmarks are not cached media and
must survive cache eviction. The command retains active scopes for library
watchers, HTTP media serving, and background library operations.

On startup, restoration precedes opening auxiliary libraries and starting
watchers. Resolution uses security scope without prompting or mounting drives.
Stale bookmarks refresh while access is active. Failed resolution, refresh, or
write preserves references and reports recovery in Watch Directories. Select the
folder again using **Restore folder access**; the existing watcher refreshes.
If the actual folder moved, use its existing Relocate action to update library
paths. Scope replacement and shutdown balance every successful access start with
a stop. No selected path is written to grant diagnostic logs.

The adapter follows Apple's [bookmark creation API](https://developer.apple.com/documentation/corefoundation/cfurlcreatebookmarkdata(_:_:_:_:_:_:)),
[resolution options](https://developer.apple.com/documentation/corefoundation/bookmark-data-resolution-options),
and [start/stop access contract](https://developer.apple.com/documentation/corefoundation/cfurlstartaccessingsecurityscopedresource(_:)).
Linux lifecycle tests use a fake backend; adapter type checking on Linux cannot
establish Apple ABI, signing, sandbox, or Powerbox behavior.

Native acceptance must use an isolated fixture and the exact signed package:

- Select a folder with playable synthetic image, audio, and video files; import,
  restart, and confirm watching, playback and saving beneath that folder.
- Check both Intel and Apple Silicon builds with the intended signing identity.
  Moving from ad-hoc to a different signing identity can invalidate old bookmarks;
  reauthorize rather than copying full-edition settings into the lite container.
- Rename/move the selected fixture, disconnect its volume, revoke its grant, and
  restart. Confirm clear recovery, retained metadata/references, and successful
  reselection/relocation. Failed bookmark writes must remain visible.
- Evict only fixture caches/transcodes; verify bookmarks, saved library assets
  and references remain. Verify bundled helpers inherit appropriate sandbox
  access and can decode/transcode the selected media.

This is source implementation and a native inspection-build configuration.
Store signing, bundled helper validation and real-Mac acceptance are outstanding;
the configuration does not submit or publish the application.
