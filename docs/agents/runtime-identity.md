# DonutMediaCenter runtime identity

The desktop executable is `donutmediacenter` (`donutmediacenter.exe` on
Windows), including the executable inside `DonutMediaCenter.app` on macOS.
Cargo declares it explicitly so development and release builds use the same
name. Linux desktop entries use that executable and window class. Android uses
`com.donutmediacenter.app` as its process name.

## Upgrade compatibility

These identifiers deliberately remain stable:

- `com.localbooru.app`: Android/iOS application ID and macOS bundle ID. Changing
  these would create a separate installation and lose access to its sandbox,
  permissions, saved settings and pairing credentials.
- Cargo package `localbooru` and native library `localbooru_lib`: mobile native
  bindings and existing packaging metadata.
- Existing library/configuration paths, environment variables, Linux package
  names, native plugin ABI and shared build locks.
- Linux single-instance D-Bus address and desktop file/icon identity.

Linux packages and the local installer provide a `localbooru` compatibility
launcher that executes `donutmediacenter` for existing shortcuts. The real
executable and process use the current name. Bundled native runtime resources
remain under `/usr/lib/localbooru`, independent of the executable name.

A running installation keeps its previous process name until it is rebuilt,
installed and restarted. Renaming these source files does not alter an already
running process.
