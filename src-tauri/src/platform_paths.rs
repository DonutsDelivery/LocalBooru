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
    #[cfg(all(target_os = "macos", feature = "store-lite"))]
    let directory = crate::macos_folder_access::helper_bookmark_directory();
    #[cfg(not(all(target_os = "macos", feature = "store-lite")))]
    let directory = None;
    command_with_bookmarks(program, directory)
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
