//! Native helper discovery independent of a desktop user's shell configuration.
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

static RESOURCES: OnceLock<PathBuf> = OnceLock::new();

#[cfg(any(desktop, test))]
pub fn desktop_data_dir(
    os: &str,
    base: Option<PathBuf>,
    app_data: Result<PathBuf, String>,
    contained: bool,
) -> Result<PathBuf, String> {
    // Contained editions must use their own app data, even when host HOME still
    // names an existing legacy library. Full desktop editions retain migration.
    if contained && matches!(os, "linux" | "macos") {
        return app_data;
    }
    let legacy = base
        .ok_or("User storage directory is unavailable")?
        .join(".localbooru");
    if os == "macos" && !legacy.is_dir() {
        app_data
    } else {
        Ok(legacy)
    }
}

pub fn init(resources: Option<PathBuf>) {
    if let Some(resources) = resources {
        let _ = RESOURCES.set(resources);
    }
}

/// App Sandbox can replace TMPDIR when launching the native process. Isolated
/// media work must use the launcher's explicit owned directory instead.
pub fn temporary_dir() -> Result<PathBuf, String> {
    if std::env::var_os("LOCALBOORU_ISOLATED_INSTANCE_ID").is_some() {
        isolated_temporary_dir(std::env::var_os("LOCALBOORU_ISOLATED_TEMP_DIR").map(PathBuf::from))
    } else {
        Ok(std::env::temp_dir())
    }
}

fn isolated_temporary_dir(directory: Option<PathBuf>) -> Result<PathBuf, String> {
    let directory = directory.ok_or("Isolated temporary directory is required")?;
    if !directory.is_absolute()
        || directory
            .components()
            .any(|part| part == std::path::Component::ParentDir)
        || !std::fs::symlink_metadata(&directory)
            .is_ok_and(|metadata| metadata.is_dir() && !metadata.file_type().is_symlink())
    {
        return Err("Isolated temporary directory is unavailable".into());
    }
    Ok(directory)
}

fn candidates(
    name: &str,
    os: &str,
    resources: Option<&Path>,
    executable: Option<&Path>,
) -> Vec<PathBuf> {
    let filename = if os == "windows" {
        format!("{name}.exe")
    } else {
        name.into()
    };
    let mut paths = Vec::new();
    if let Some(resources) = resources {
        paths.push(resources.join("bin").join(&filename));
        paths.push(resources.join(&filename));
    }
    if let Some(executable) = executable {
        paths.push(executable.join(&filename));
    }
    if os == "macos" {
        paths.push(PathBuf::from("/opt/homebrew/bin").join(&filename));
        paths.push(PathBuf::from("/usr/local/bin").join(&filename));
    }
    paths
}

pub fn helper(name: &str) -> PathBuf {
    let executable = std::env::current_exe().ok();
    // The sandbox edition must never mask a missing bundled helper by running
    // an unrelated Homebrew/PATH executable without our bookmark activation code.
    #[cfg(all(target_os = "macos", feature = "store-lite"))]
    return contained_helper(
        name,
        RESOURCES.get().map(PathBuf::as_path),
        executable.as_deref().and_then(Path::parent),
    );
    #[cfg(not(all(target_os = "macos", feature = "store-lite")))]
    candidates(
        name,
        std::env::consts::OS,
        RESOURCES.get().map(PathBuf::as_path),
        executable.as_deref().and_then(Path::parent),
    )
    .into_iter()
    .find(|path| path.is_file())
    .unwrap_or_else(|| PathBuf::from(name))
}

#[cfg(any(all(target_os = "macos", feature = "store-lite"), test))]
fn contained_helper(name: &str, resources: Option<&Path>, executable: Option<&Path>) -> PathBuf {
    let resources = resources.map(Path::to_path_buf).or_else(|| {
        executable
            .and_then(Path::parent)
            .map(|contents| contents.join("Resources"))
    });
    resources
        .unwrap_or_else(|| PathBuf::from("/DMC-missing-resources"))
        .join("bin")
        .join(name)
}

fn command_with_bookmarks(
    program: impl AsRef<std::ffi::OsStr>,
    directory: Option<&Path>,
) -> std::process::Command {
    let mut command = std::process::Command::new(program);
    command.env_remove("DMC_MEDIA_BOOKMARK_DIRECTORY");
    if let Some(directory) = directory {
        command.env("DMC_MEDIA_BOOKMARK_DIRECTORY", directory);
    }
    command
}

/// Media helpers activate the parent's persisted grants inside their own process.
/// Full editions clear this private transfer variable rather than inherit it.
pub fn media_command(program: impl AsRef<std::ffi::OsStr>) -> std::process::Command {
    let program = program.as_ref();
    media_command_with_helper(program, program)
}

/// Priority wrappers pass on the media helper's library environment unchanged.
/// Select that environment using the actual helper, rather than the wrapper.
pub fn media_command_with_helper(
    program: impl AsRef<std::ffi::OsStr>,
    helper: impl AsRef<std::ffi::OsStr>,
) -> std::process::Command {
    #[cfg(all(target_os = "macos", feature = "store-lite"))]
    let directory = crate::macos_folder_access::helper_bookmark_directory();
    #[cfg(not(all(target_os = "macos", feature = "store-lite")))]
    let directory = None;
    let command = command_with_bookmarks(program, directory);
    #[cfg(target_os = "linux")]
    {
        let mut command = command;
        restore_appimage_helper_library_path(&mut command, helper.as_ref());
        command
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = helper;
        command
    }
}

#[cfg(target_os = "linux")]
fn restore_appimage_helper_library_path(
    command: &mut std::process::Command,
    program: &std::ffi::OsStr,
) {
    apply_appimage_helper_library_path(
        command,
        program,
        std::env::var_os("APPDIR"),
        std::env::var_os("DMC_APPIMAGE_LD_LIBRARY_PATH_STATE"),
        std::env::var_os("DMC_APPIMAGE_ORIGINAL_LD_LIBRARY_PATH"),
        std::env::var_os("PATH"),
    );
}

#[cfg(target_os = "linux")]
fn apply_appimage_helper_library_path(
    command: &mut std::process::Command,
    program: &std::ffi::OsStr,
    appdir: Option<std::ffi::OsString>,
    state: Option<std::ffi::OsString>,
    original: Option<std::ffi::OsString>,
    search_path: Option<std::ffi::OsString>,
) {
    use std::os::unix::fs::PermissionsExt;
    let Some(root) = appdir.map(PathBuf::from).filter(|root| root.is_absolute()) else {
        return;
    };
    let root = root.canonicalize().unwrap_or(root);
    let path = Path::new(program);
    let resolved = if path.components().count() == 1 {
        search_path.and_then(|paths| {
            std::env::split_paths(&paths)
                .map(|dir| dir.join(path))
                .find(|candidate| {
                    candidate.metadata().is_ok_and(|metadata| {
                        metadata.is_file() && metadata.permissions().mode() & 0o111 != 0
                    })
                })
        })
    } else {
        Some(path.to_path_buf())
    };
    // Resolve aliases too: a bundle entry can point to a host tool or vice versa.
    let bundled =
        resolved.is_some_and(|path| path.canonicalize().unwrap_or(path).starts_with(&root));
    if !bundled {
        match state.as_deref() {
            Some(state) if state == "set" => {
                if let Some(original) = original {
                    command.env("LD_LIBRARY_PATH", original);
                }
            }
            Some(state) if state == "unset" => {
                command.env_remove("LD_LIBRARY_PATH");
            }
            _ => {}
        }
    }
}

pub fn media_command_async(program: impl AsRef<std::ffi::OsStr>) -> tokio::process::Command {
    tokio::process::Command::from(media_command(program))
}

/// Directory of a helper found outside PATH, so child processes that look it up
/// on PATH (such as addon sidecars) find the bundled copy too.
#[cfg(target_os = "windows")]
pub fn helper_dir(name: &str) -> Option<PathBuf> {
    let path = helper(name);
    if path.is_absolute() {
        path.parent().map(Path::to_path_buf)
    } else {
        None
    }
}

/// Background helpers must not flash a console window on Windows.
pub trait HideConsoleWindow {
    fn hide_console_window(&mut self) -> &mut Self;
}

impl HideConsoleWindow for std::process::Command {
    fn hide_console_window(&mut self) -> &mut Self {
        #[cfg(target_os = "windows")]
        std::os::windows::process::CommandExt::creation_flags(self, 0x08000000); // CREATE_NO_WINDOW
        self
    }
}

impl HideConsoleWindow for tokio::process::Command {
    fn hide_console_window(&mut self) -> &mut Self {
        #[cfg(target_os = "windows")]
        self.creation_flags(0x08000000); // CREATE_NO_WINDOW
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn isolated_temporary_work_never_falls_back_to_system_storage() {
        let fixture = tempfile::tempdir().unwrap();
        let owned = fixture.path().join("owned temp");
        std::fs::create_dir(&owned).unwrap();
        assert_eq!(isolated_temporary_dir(Some(owned.clone())).unwrap(), owned);
        for invalid in [
            None,
            Some(PathBuf::from("relative")),
            Some(fixture.path().join("missing")),
        ] {
            assert!(isolated_temporary_dir(invalid).is_err());
        }
        let file = fixture.path().join("file");
        std::fs::write(&file, b"synthetic fixture").unwrap();
        assert!(isolated_temporary_dir(Some(file)).is_err());
        assert!(isolated_temporary_dir(Some(owned.join(".."))).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn isolated_temporary_work_rejects_a_redirected_directory() {
        let fixture = tempfile::tempdir().unwrap();
        let owned = fixture.path().join("owned");
        std::fs::create_dir(&owned).unwrap();
        let redirected = fixture.path().join("redirected");
        std::os::unix::fs::symlink(&owned, &redirected).unwrap();
        assert!(isolated_temporary_dir(Some(redirected)).is_err());
    }

    #[test]
    fn mac_lite_uses_its_own_data_without_opening_an_existing_full_library() {
        let fixture = tempfile::tempdir().unwrap();
        let home = fixture.path().join("synthetic home");
        let legacy = home.join(".localbooru");
        let app_data = fixture.path().join("lite-container/app-data");
        std::fs::create_dir_all(&legacy).unwrap();
        std::fs::write(legacy.join("synthetic-media.txt"), "preserve").unwrap();
        assert_eq!(
            desktop_data_dir("macos", Some(home), Ok(app_data.clone()), true).unwrap(),
            app_data
        );
        assert_eq!(
            std::fs::read_to_string(legacy.join("synthetic-media.txt")).unwrap(),
            "preserve"
        );
        assert!(!legacy.join("settings.json").exists());
    }
    #[test]
    fn new_mac_data_uses_application_support_but_preserves_existing_libraries() {
        let temporary = tempfile::Builder::new()
            .prefix("DMC User æ space ")
            .tempdir()
            .unwrap();
        let home = temporary.path().join("Home æ");
        let app_data = home.join("Library/Application Support/com.localbooru.app");
        assert_eq!(
            desktop_data_dir("macos", Some(home.clone()), Ok(app_data.clone()), false).unwrap(),
            app_data
        );
        let legacy = home.join(".localbooru");
        std::fs::create_dir_all(&legacy).unwrap();
        assert_eq!(
            desktop_data_dir("macos", Some(home.clone()), Ok(app_data), false).unwrap(),
            legacy
        );
        assert!(desktop_data_dir("macos", None, Ok(home), false).is_err());
    }
    #[test]
    fn other_desktop_libraries_keep_their_existing_layout() {
        for os in ["linux", "windows"] {
            assert_eq!(
                desktop_data_dir(
                    os,
                    Some(PathBuf::from("User æ space")),
                    Err("unavailable".into()),
                    false
                )
                .unwrap(),
                PathBuf::from("User æ space/.localbooru")
            );
        }
    }
    #[test]
    fn flatpak_uses_container_app_data_even_with_legacy_host_home() {
        let temporary = tempfile::tempdir().unwrap();
        let host = temporary.path().join("host-home");
        let legacy = host.join(".localbooru");
        let app_data = temporary.path().join("sandbox/data/com.localbooru.app");
        std::fs::create_dir_all(&legacy).unwrap();
        let selected =
            desktop_data_dir("linux", Some(host.clone()), Ok(app_data.clone()), true).unwrap();
        assert_eq!(selected, app_data);
        std::fs::create_dir_all(&selected).unwrap();
        std::fs::write(
            selected.join("settings.json"),
            br#"{"onboarding_complete":true}"#,
        )
        .unwrap();
        // Resolve again as a subsequent process would, with host and XDG separated.
        let reopened =
            desktop_data_dir("linux", Some(host.clone()), Ok(app_data.clone()), true).unwrap();
        assert_eq!(
            std::fs::read(reopened.join("settings.json"))
                .unwrap()
                .as_slice(),
            br#"{"onboarding_complete":true}"#
        );
        assert!(!legacy.join("settings.json").exists());
        assert_eq!(
            desktop_data_dir("linux", Some(host), Err("no XDG data".into()), true).unwrap_err(),
            "no XDG data"
        );
        assert_eq!(
            desktop_data_dir("linux", None, Ok(app_data.clone()), true).unwrap(),
            app_data
        );
    }
    #[test]
    fn finder_and_windows_helpers_use_native_resource_paths() {
        let mac = candidates(
            "ffprobe",
            "macos",
            Some(Path::new("/Applications/Donut æ.app/Contents/Resources")),
            Some(Path::new("/Applications/Donut æ.app/Contents/MacOS")),
        );
        assert_eq!(
            mac[0],
            PathBuf::from("/Applications/Donut æ.app/Contents/Resources/bin/ffprobe")
        );
        assert!(mac.contains(&PathBuf::from("/opt/homebrew/bin/ffprobe")));
        assert!(mac.contains(&PathBuf::from("/usr/local/bin/ffprobe")));
        let windows = candidates("ffmpeg", "windows", Some(Path::new("DMC æ space")), None);
        assert_eq!(windows[0], PathBuf::from("DMC æ space/bin/ffmpeg.exe"));
        assert_eq!(windows[1], PathBuf::from("DMC æ space/ffmpeg.exe"));
    }

    #[test]
    fn contained_helpers_never_fall_back_to_host_search_paths() {
        assert_eq!(
            contained_helper(
                "ffmpeg",
                Some(Path::new("/fixture/Donut æ.app/Contents/Resources")),
                None
            ),
            PathBuf::from("/fixture/Donut æ.app/Contents/Resources/bin/ffmpeg")
        );
        assert_eq!(
            contained_helper(
                "ffprobe",
                None,
                Some(Path::new("/fixture/App.app/Contents/MacOS"))
            ),
            PathBuf::from("/fixture/App.app/Contents/Resources/bin/ffprobe")
        );
        assert!(contained_helper("ffmpeg", None, None).is_absolute());
    }

    #[cfg(target_os = "linux")]
    fn library_path_child(
        program: &Path,
        appdir: Option<&Path>,
        state: Option<&str>,
        original: Option<&str>,
        search_path: Option<&std::ffi::OsStr>,
    ) -> Vec<u8> {
        let mut command = command_with_bookmarks(program, None);
        command.env("LD_LIBRARY_PATH", "/synthetic-inherited/lib");
        command.env("DMC_UNRELATED_FIXTURE", "retained");
        if let Some(path) = search_path {
            command.env("PATH", path);
        }
        apply_appimage_helper_library_path(
            &mut command,
            program.as_os_str(),
            appdir.map(|p| p.as_os_str().to_owned()),
            state.map(Into::into),
            original.map(Into::into),
            search_path.map(ToOwned::to_owned),
        );
        let output = command.output().unwrap();
        assert!(output.status.success(), "{:?}", output);
        output.stdout
    }

    #[cfg(target_os = "linux")]
    fn library_path_script(path: &Path) {
        use std::os::unix::fs::PermissionsExt;
        std::fs::write(
            path,
            "#!/bin/sh\nprintf '%s|%s' \"${LD_LIBRARY_PATH-missing}\" \"$DMC_UNRELATED_FIXTURE\"\n",
        )
        .unwrap();
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn appimage_host_helper_receives_original_library_path() {
        let fixture = tempfile::tempdir().unwrap();
        let appdir = fixture.path().join("App Dir");
        std::fs::create_dir(&appdir).unwrap();
        let helper = fixture.path().join("host helper");
        library_path_script(&helper);
        for (state, original, expected) in [
            ("unset", None, "missing|retained"),
            ("set", Some(""), "|retained"),
            ("set", Some("/caller/lib"), "/caller/lib|retained"),
        ] {
            assert_eq!(
                library_path_child(&helper, Some(&appdir), Some(state), original, None),
                expected.as_bytes()
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn appimage_bundled_helpers_retain_library_path_including_path_lookup() {
        let fixture = tempfile::tempdir().unwrap();
        let helper = fixture.path().join("ffprobe");
        library_path_script(&helper);
        for program in [helper.as_path(), Path::new("ffprobe")] {
            assert_eq!(
                library_path_child(
                    program,
                    Some(fixture.path()),
                    Some("unset"),
                    None,
                    Some(fixture.path().as_os_str())
                ),
                b"/synthetic-inherited/lib|retained"
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn appimage_host_helper_requires_complete_launcher_context() {
        let fixture = tempfile::tempdir().unwrap();
        let appdir = fixture.path().join("bundle");
        std::fs::create_dir(&appdir).unwrap();
        let helper = fixture.path().join("host-helper");
        library_path_script(&helper);
        for (root, state, original) in [
            (None, Some("unset"), None),
            (Some(Path::new("relative")), Some("unset"), None),
            (Some(appdir.as_path()), None, None),
            (Some(appdir.as_path()), Some("invalid"), None),
            (Some(appdir.as_path()), Some("set"), None),
        ] {
            assert_eq!(
                library_path_child(&helper, root, state, original, None),
                b"/synthetic-inherited/lib|retained"
            );
        }
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn appimage_helper_aliases_follow_the_actual_helper_location() {
        use std::os::unix::fs::symlink;
        let fixture = tempfile::tempdir().unwrap();
        let appdir = fixture.path().join("bundle");
        std::fs::create_dir(&appdir).unwrap();
        let host = fixture.path().join("host-helper");
        let bundled = appdir.join("bundled-helper");
        library_path_script(&host);
        library_path_script(&bundled);
        let host_alias = appdir.join("host-alias");
        let bundled_alias = fixture.path().join("bundled-alias");
        symlink(&host, &host_alias).unwrap();
        symlink(&bundled, &bundled_alias).unwrap();
        assert_eq!(
            library_path_child(&host_alias, Some(&appdir), Some("unset"), None, None),
            b"missing|retained"
        );
        assert_eq!(
            library_path_child(&bundled_alias, Some(&appdir), Some("unset"), None, None),
            b"/synthetic-inherited/lib|retained"
        );
        let appdir_alias = fixture.path().join("bundle-alias");
        symlink(&appdir, &appdir_alias).unwrap();
        assert_eq!(
            library_path_child(&bundled, Some(&appdir_alias), Some("unset"), None, None),
            b"/synthetic-inherited/lib|retained"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn appimage_path_lookup_skips_nonexecutable_bundle_entries() {
        let fixture = tempfile::tempdir().unwrap();
        let appdir = fixture.path().join("bundle");
        let host = fixture.path().join("host");
        std::fs::create_dir(&appdir).unwrap();
        std::fs::create_dir(&host).unwrap();
        std::fs::write(appdir.join("ffprobe"), "not executable").unwrap();
        library_path_script(&host.join("ffprobe"));
        let paths = std::env::join_paths([&appdir, &host]).unwrap();
        assert_eq!(
            library_path_child(
                Path::new("ffprobe"),
                Some(&appdir),
                Some("set"),
                Some("/caller/lib"),
                Some(&paths)
            ),
            b"/caller/lib|retained"
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn appimage_wrapped_helper_child() {
        let Some(helper) = std::env::var_os("DMC_TEST_WRAPPED_HELPER") else {
            return;
        };
        let wrapper = std::env::var_os("DMC_TEST_PRIORITY_WRAPPER").unwrap();
        let second_wrapper = std::env::var_os("DMC_TEST_SECOND_WRAPPER").unwrap();
        let output = media_command_with_helper(&wrapper, &helper)
            .arg("-c3")
            .arg(second_wrapper)
            .args(["-n", "10"])
            .arg(helper)
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        assert_eq!(
            output.stdout,
            std::env::var("DMC_TEST_EXPECTED_LIBRARY_PATH")
                .unwrap()
                .as_bytes()
        );
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn appimage_wrapper_chain_uses_the_actual_media_helper_environment() {
        use std::os::unix::fs::PermissionsExt;
        let fixture = tempfile::tempdir().unwrap();
        let appdir = fixture.path().join("bundle");
        std::fs::create_dir(&appdir).unwrap();
        let first = fixture.path().join("ionice");
        let second = fixture.path().join("nice");
        std::fs::write(
            &first,
            "#!/bin/sh\n[ \"$1\" = -c3 ] || exit 40\nshift\nexec \"$@\"\n",
        )
        .unwrap();
        std::fs::write(
            &second,
            "#!/bin/sh\n[ \"$1 $2\" = '-n 10' ] || exit 41\nshift 2\nexec \"$@\"\n",
        )
        .unwrap();
        for path in [&first, &second] {
            std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        for (location, bundled) in [(appdir.as_path(), true), (fixture.path(), false)] {
            let helper = location.join("ffmpeg");
            library_path_script(&helper);
            for original in [None, Some(""), Some("/caller/lib")] {
                let mut child = std::process::Command::new(std::env::current_exe().unwrap());
                child
                    .arg("appimage_wrapped_helper_child")
                    .arg("--nocapture")
                    .env("APPDIR", &appdir)
                    .env("DMC_TEST_WRAPPED_HELPER", &helper)
                    .env("DMC_TEST_PRIORITY_WRAPPER", &first)
                    .env("DMC_TEST_SECOND_WRAPPER", &second)
                    .env("DMC_UNRELATED_FIXTURE", "retained")
                    .env("LD_LIBRARY_PATH", "/synthetic-inherited/lib")
                    .env(
                        "DMC_APPIMAGE_LD_LIBRARY_PATH_STATE",
                        if original.is_some() { "set" } else { "unset" },
                    )
                    .env(
                        "DMC_TEST_EXPECTED_LIBRARY_PATH",
                        format!(
                            "{}|retained",
                            if bundled {
                                "/synthetic-inherited/lib"
                            } else {
                                original.unwrap_or("missing")
                            }
                        ),
                    );
                if let Some(original) = original {
                    child.env("DMC_APPIMAGE_ORIGINAL_LD_LIBRARY_PATH", original);
                } else {
                    child.env_remove("DMC_APPIMAGE_ORIGINAL_LD_LIBRARY_PATH");
                }
                let output = child.output().unwrap();
                assert!(output.status.success(), "{output:?}");
                assert!(
                    String::from_utf8_lossy(&output.stdout).contains("1 passed"),
                    "{output:?}"
                );
            }
        }
    }

    #[test]
    fn media_bookmark_transfer_is_explicit_and_does_not_change_parent_environment() {
        let original = std::env::var_os("DMC_MEDIA_BOOKMARK_DIRECTORY");
        let directory = Path::new("/synthetic/container/folder grants");
        let command = command_with_bookmarks("fixture-ffmpeg", Some(directory));
        let values: Vec<_> = command.get_envs().collect();
        assert!(values.contains(&(
            std::ffi::OsStr::new("DMC_MEDIA_BOOKMARK_DIRECTORY"),
            Some(directory.as_os_str())
        )));
        let full = command_with_bookmarks("fixture-ffmpeg", None);
        assert!(full
            .get_envs()
            .any(|(name, value)| name == "DMC_MEDIA_BOOKMARK_DIRECTORY" && value.is_none()));
        assert_eq!(std::env::var_os("DMC_MEDIA_BOOKMARK_DIRECTORY"), original);
    }
}
