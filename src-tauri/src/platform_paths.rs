//! Native helper discovery independent of a desktop user's shell configuration.
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

static RESOURCES: OnceLock<PathBuf> = OnceLock::new();

#[cfg(any(desktop, test))]
pub fn desktop_data_dir(
    os: &str,
    base: Option<PathBuf>,
    app_data: Result<PathBuf, String>,
) -> Result<PathBuf, String> {
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
            desktop_data_dir("macos", Some(home.clone()), Ok(app_data.clone())).unwrap(),
            app_data
        );
        let legacy = home.join(".localbooru");
        std::fs::create_dir_all(&legacy).unwrap();
        assert_eq!(
            desktop_data_dir("macos", Some(home.clone()), Ok(app_data)).unwrap(),
            legacy
        );
        assert!(desktop_data_dir("macos", None, Ok(home)).is_err());
    }
    #[test]
    fn other_desktop_libraries_keep_their_existing_layout() {
        for os in ["linux", "windows"] {
            assert_eq!(
                desktop_data_dir(
                    os,
                    Some(PathBuf::from("User æ space")),
                    Err("unavailable".into())
                )
                .unwrap(),
                PathBuf::from("User æ space/.localbooru")
            );
        }
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
