//! Native helper discovery independent of a desktop user's shell configuration.
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

static RESOURCES: OnceLock<PathBuf> = OnceLock::new();

#[cfg(any(desktop, test))]
pub fn desktop_data_dir(
    os: &str,
    base: Option<PathBuf>,
    app_data: Result<PathBuf, String>,
    flatpak: bool,
) -> Result<PathBuf, String> {
    // Flatpak retains the host HOME while redirecting XDG data to its container.
    // A home-relative legacy path may be inaccessible and must never win here.
    if os == "linux" && flatpak {
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
}
