//! App-private Python fallback for desktop installations without a usable Python.
//! Assets and hashes are pinned; an interrupted install never becomes a runtime.
use flate2::read::GzDecoder;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::sync::Mutex;
use std::time::Duration;

static INSTALL: Mutex<()> = Mutex::new(());

#[derive(Deserialize)]
struct Manifest {
    release: String,
    assets: Vec<Asset>,
}
#[derive(Deserialize)]
struct Asset {
    target: String,
    url: String,
    sha256: String,
    size: u64,
}

fn target(os: &str, arch: &str) -> Option<&'static str> {
    match (os, arch) {
        ("linux", "x86_64") => Some("x86_64-unknown-linux-gnu"),
        ("linux", "aarch64") => Some("aarch64-unknown-linux-gnu"),
        ("macos", "aarch64") => Some("aarch64-apple-darwin"),
        ("macos", "x86_64") => Some("x86_64-apple-darwin"),
        ("windows", "x86_64") => Some("x86_64-pc-windows-msvc"),
        _ => None,
    }
}

fn binary(root: &Path) -> PathBuf {
    root.join(if cfg!(target_os = "windows") {
        "python/python.exe"
    } else {
        "python/bin/python3"
    })
}

fn probe(python: &Path) -> Result<(), String> {
    let result = std::process::Command::new(python)
        .args([
            "-I",
            "-c",
            "import sys, ssl, venv, ensurepip; assert sys.version_info[:2] == (3, 12)",
        ])
        .output()
        .map_err(|error| format!("Could not start the app's Python runtime: {error}"))?;
    if !result.status.success() {
        return Err("The app's Python runtime failed its Python 3.12/venv/SSL check.".into());
    }
    Ok(())
}

/// Called from the blocking addon installer, never from a GUI callback.
pub fn interpreter(data_dir: &Path, require_312: bool) -> Result<PathBuf, String> {
    let existing = if require_312 {
        super::sidecar::find_python_version(12, 12)
    } else {
        super::sidecar::find_python()
    };
    if let Some(python) = existing {
        if !require_312 || probe(&python).is_ok() {
            return Ok(python);
        }
    }
    let target = target(std::env::consts::OS, std::env::consts::ARCH)
        .ok_or("Local Python addons are unavailable on this platform. Connect to a desktop server instead.")?;
    let _lock = INSTALL.lock().map_err(|_| "Python installer lock failed")?;
    let manifest: Manifest = serde_json::from_str(include_str!("python_runtime.json"))
        .map_err(|error| format!("Invalid bundled Python manifest: {error}"))?;
    let asset = manifest
        .assets
        .iter()
        .find(|asset| asset.target == target)
        .ok_or("No bundled Python runtime matches this desktop")?;
    let parent = data_dir.join("runtimes/python");
    let root = parent.join(format!("3.12.15-{}-{target}", manifest.release));
    if root.exists() {
        let receipt = std::fs::read_to_string(root.join("archive.sha256"))
            .map_err(|_| "An incomplete Python runtime exists. Remove it from the app's runtimes folder and retry.")?;
        if receipt.trim() != asset.sha256 {
            return Err(
                "The Python runtime receipt does not match this app's pinned runtime.".into(),
            );
        }
        probe(&binary(&root))?;
        return Ok(binary(&root));
    }
    std::fs::create_dir_all(&parent)
        .map_err(|error| format!("Cannot create runtime storage: {error}"))?;
    let stage = tempfile::Builder::new()
        .prefix(".python-install-")
        .tempdir_in(&parent)
        .map_err(|error| format!("Cannot create Python install staging: {error}"))?;
    let archive_path = stage.path().join("runtime.tar.gz");
    log::info!("[PythonRuntime] Downloading pinned Python 3.12 for {target}");
    tauri::async_runtime::block_on(download(asset, &archive_path))?;
    let unpacked = stage.path().join("runtime");
    std::fs::create_dir(&unpacked).map_err(|error| error.to_string())?;
    extract(&archive_path, &unpacked)?;
    probe(&binary(&unpacked))?;
    std::fs::write(unpacked.join("archive.sha256"), &asset.sha256)
        .map_err(|error| error.to_string())?;
    // A concurrent app may have completed the same runtime. Never overwrite it.
    if let Err(error) = std::fs::rename(&unpacked, &root) {
        if std::fs::read_to_string(root.join("archive.sha256"))
            .ok()
            .as_deref()
            != Some(asset.sha256.as_str())
        {
            return Err(format!("Could not publish Python runtime: {error}"));
        }
    }
    probe(&binary(&root))?;
    Ok(binary(&root))
}

async fn download(asset: &Asset, output: &Path) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .https_only(true)
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(600))
        .build()
        .map_err(|error| error.to_string())?;
    let mut response = client
        .get(&asset.url)
        .send()
        .await
        .and_then(reqwest::Response::error_for_status)
        .map_err(|error| format!("Could not download the app's Python runtime: {error}"))?;
    let mut file = std::fs::File::create(output).map_err(|error| error.to_string())?;
    let mut hash = Sha256::new();
    let mut size = 0u64;
    while let Some(chunk) = response.chunk().await.map_err(|error| error.to_string())? {
        size += chunk.len() as u64;
        if size > asset.size {
            return Err("Python runtime download exceeds its pinned size".into());
        }
        hash.update(&chunk);
        file.write_all(&chunk).map_err(|error| error.to_string())?;
    }
    if size != asset.size || format!("{:x}", hash.finalize()) != asset.sha256 {
        return Err(
            "Python runtime download failed its size/SHA-256 check. Retry the installation.".into(),
        );
    }
    file.sync_all().map_err(|error| error.to_string())
}

fn safe_member(path: &Path) -> bool {
    let mut components = path.components();
    components.next() == Some(Component::Normal("python".as_ref()))
        && components.all(|component| matches!(component, Component::Normal(_) | Component::CurDir))
}

fn safe_link(member: &Path, link: &Path, hard: bool) -> bool {
    let base = if hard {
        Path::new("")
    } else {
        member.parent().unwrap_or(Path::new(""))
    };
    let mut parts = Vec::new();
    for component in base.components().chain(link.components()) {
        match component {
            Component::Normal(part) => parts.push(part),
            Component::CurDir => (),
            Component::ParentDir if parts.len() > 1 => {
                parts.pop();
            }
            _ => return false,
        }
    }
    parts.first().is_some_and(|part| *part == "python")
}

fn extract(archive: &Path, destination: &Path) -> Result<(), String> {
    let file = std::fs::File::open(archive).map_err(|error| error.to_string())?;
    let mut tar = tar::Archive::new(GzDecoder::new(file));
    let mut total = 0u64;
    for (index, entry) in tar
        .entries()
        .map_err(|error| error.to_string())?
        .enumerate()
    {
        let mut entry = entry.map_err(|error| error.to_string())?;
        total = total
            .checked_add(entry.size())
            .ok_or("Python archive size overflow")?;
        if index >= 20_000 || total > 512 * 1024 * 1024 {
            return Err("Python archive exceeds extraction limits".into());
        }
        let path = entry
            .path()
            .map_err(|error| error.to_string())?
            .into_owned();
        let kind = entry.header().entry_type();
        if !safe_member(&path)
            || !(kind.is_file() || kind.is_dir() || kind.is_symlink() || kind.is_hard_link())
        {
            return Err("Python archive contains an unsafe entry".into());
        }
        if kind.is_symlink() || kind.is_hard_link() {
            let link = entry
                .link_name()
                .map_err(|error| error.to_string())?
                .ok_or("Python archive link lacks a target")?;
            if !safe_link(&path, &link, kind.is_hard_link()) {
                return Err("Python archive link escapes its runtime".into());
            }
        }
        if !entry
            .unpack_in(destination)
            .map_err(|error| error.to_string())?
        {
            return Err("Python archive entry escaped its runtime".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;
    // AC: @donut-create-plugin ac-managed-setup
    #[test]
    fn pinned_targets_include_both_mac_architectures_and_windows() {
        let manifest: Manifest = serde_json::from_str(include_str!("python_runtime.json")).unwrap();
        for (os, arch) in [
            ("macos", "aarch64"),
            ("macos", "x86_64"),
            ("windows", "x86_64"),
            ("linux", "x86_64"),
            ("linux", "aarch64"),
        ] {
            let asset = manifest
                .assets
                .iter()
                .find(|asset| Some(asset.target.as_str()) == target(os, arch))
                .unwrap();
            assert!(asset.url.starts_with(
                "https://github.com/astral-sh/python-build-standalone/releases/download/"
            ));
            assert_eq!(asset.sha256.len(), 64);
            assert!(asset.size < 40 * 1024 * 1024);
        }
        assert!(target("android", "aarch64").is_none());
    }
    #[test]
    fn archive_paths_and_links_stay_inside_python() {
        assert!(safe_member(Path::new("python/bin/python3")));
        for path in [
            "/python/bin/python",
            "python/../settings.json",
            "other/bin/python",
        ] {
            assert!(!safe_member(Path::new(path)));
        }
        assert!(safe_link(
            Path::new("python/bin/python3"),
            Path::new("python3.12"),
            false
        ));
        assert!(safe_link(
            Path::new("python/lib/example"),
            Path::new("../bin/python3"),
            false
        ));
        assert!(!safe_link(
            Path::new("python/bin/python3"),
            Path::new("../../outside"),
            false
        ));
        assert!(!safe_link(
            Path::new("python/bin/python3"),
            Path::new("/tmp/outside"),
            false
        ));
        assert!(safe_link(
            Path::new("python/bin/python"),
            Path::new("python/bin/python3"),
            true
        ));
    }
    #[test]
    fn extraction_preserves_space_and_unicode_paths() {
        let temporary = tempfile::Builder::new()
            .prefix("DMC Python æ space ")
            .tempdir()
            .unwrap();
        let archive = temporary.path().join("runtime.tar.gz");
        let mut builder = tar::Builder::new(flate2::write::GzEncoder::new(
            std::fs::File::create(&archive).unwrap(),
            flate2::Compression::fast(),
        ));
        let mut header = tar::Header::new_gnu();
        header.set_size(4);
        header.set_mode(0o755);
        header.set_cksum();
        builder
            .append_data(&mut header, "python/bin/python3", &b"test"[..])
            .unwrap();
        builder.into_inner().unwrap().finish().unwrap();
        let destination = temporary.path().join("Runtime æ space");
        std::fs::create_dir(&destination).unwrap();
        extract(&archive, &destination).unwrap();
        let mut contents = String::new();
        std::fs::File::open(destination.join("python/bin/python3"))
            .unwrap()
            .read_to_string(&mut contents)
            .unwrap();
        assert_eq!(contents, "test");
    }
}
