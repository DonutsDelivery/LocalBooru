use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::{Mutex, OnceLock};

#[cfg(target_os = "windows")]
fn suppress_console_window(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    command.creation_flags(0x08000000); // CREATE_NO_WINDOW
}

#[cfg(not(target_os = "windows"))]
fn suppress_console_window(_command: &mut Command) {}

/// Check if ffmpeg is available on the system.
pub fn check_ffmpeg_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let mut command = Command::new("ffmpeg");
        command
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        suppress_console_window(&mut command);
        command.status().map(|s| s.success()).unwrap_or(false)
    })
}

/// Check if ffprobe is available on the system.
pub fn check_ffprobe_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let mut command = Command::new("ffprobe");
        command
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        suppress_console_window(&mut command);
        command.status().map(|s| s.success()).unwrap_or(false)
    })
}

/// Get video metadata (width, height, duration) in a single ffprobe call.
pub fn get_video_metadata(file_path: &str) -> Option<(i32, i32, f64)> {
    if !check_ffprobe_available() {
        return None;
    }

    let mut command = Command::new("ffprobe");
    command.args([
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration:stream=width,height",
        "-of",
        "json",
        file_path,
    ]);
    suppress_console_window(&mut command);
    let output = command.output().ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let json: serde_json::Value = serde_json::from_str(&stdout).ok()?;

    let stream = json.get("streams")?.as_array()?.first()?;
    let width = stream.get("width")?.as_i64()? as i32;
    let height = stream.get("height")?.as_i64()? as i32;

    let duration_str = json.get("format")?.get("duration")?.as_str()?;
    let duration = duration_str.parse::<f64>().ok()?;

    Some((width, height, duration))
}

/// Get video duration in seconds using ffprobe.
pub fn get_video_duration(file_path: &str) -> Option<f64> {
    get_video_metadata(file_path).map(|(_, _, d)| d)
}

/// Get video dimensions (width, height) using ffprobe.
pub fn get_video_dimensions(file_path: &str) -> Option<(i32, i32)> {
    get_video_metadata(file_path).map(|(w, h, _)| (w, h))
}

/// Get low-priority prefix for subprocess commands (ionice + nice on Linux).
pub fn get_low_priority_prefix() -> Vec<String> {
    #[cfg(target_os = "linux")]
    {
        // Check if ionice and nice are available
        let ionice_ok = Command::new("ionice")
            .arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false);

        if ionice_ok {
            return vec![
                "ionice".into(),
                "-c".into(),
                "3".into(),
                "nice".into(),
                "-n".into(),
                "19".into(),
            ];
        }
    }
    vec![]
}

/// Get hardware acceleration args for ffmpeg (NVDEC if available).
pub fn get_hwaccel_args() -> Vec<String> {
    static ARGS: OnceLock<Vec<String>> = OnceLock::new();
    ARGS.get_or_init(|| {
        if !check_ffmpeg_available() {
            return vec![];
        }
        let mut command = Command::new("ffmpeg");
        command.arg("-hwaccels");
        suppress_console_window(&mut command);
        let output = command.output().ok();

        if let Some(out) = output {
            let stdout = String::from_utf8_lossy(&out.stdout);
            if stdout.contains("cuda") {
                return vec!["-hwaccel".into(), "cuda".into()];
            }
        }
        vec![]
    })
    .clone()
}

/// Get the preview directory for a given file hash.
pub fn get_preview_dir(data_dir: &Path, file_hash: &str) -> PathBuf {
    let hash_prefix = &file_hash[..16.min(file_hash.len())];
    data_dir.join("previews").join(hash_prefix)
}

/// Get existing complete preview frame paths for a file hash.
pub fn get_preview_frames(data_dir: &Path, file_hash: &str) -> Vec<PathBuf> {
    let preview_dir = get_preview_dir(data_dir, file_hash);
    if !preview_dir.join(".complete").is_file() {
        return vec![];
    }

    let frames: Vec<PathBuf> = (0..8)
        .map(|i| preview_dir.join(format!("frame_{}.webp", i)))
        .collect();
    if frames.iter().all(|path| path.is_file()) {
        frames
    } else {
        vec![]
    }
}

/// Delete preview frames for a file hash.
pub fn delete_preview_frames(data_dir: &Path, file_hash: &str) -> bool {
    let preview_dir = get_preview_dir(data_dir, file_hash);
    if preview_dir.exists() {
        std::fs::remove_dir_all(&preview_dir).is_ok()
    } else {
        false
    }
}

/// Extract preview frames from a video using ffmpeg.
///
/// Uses batched ffmpeg (single command with multiple -ss/-i pairs) for 3-4x speedup.
/// Skips first/last 5% to avoid black frames.
pub fn extract_preview_frames(
    video_path: &str,
    output_dir: &Path,
    num_frames: usize,
    frame_width: u32,
) -> Vec<PathBuf> {
    if !check_ffmpeg_available() {
        return vec![];
    }

    std::fs::create_dir_all(output_dir).ok();

    let duration = match get_video_duration(video_path) {
        Some(d) if d > 0.1 => d,
        _ => return vec![],
    };

    // Skip first/last 5% of video
    let start = duration * 0.05;
    let end = duration * 0.95;
    let interval = (end - start) / num_frames as f64;

    let low_priority = get_low_priority_prefix();
    let hwaccel = get_hwaccel_args();

    let mut cmd_args: Vec<String> = low_priority;
    cmd_args.push("ffmpeg".into());
    cmd_args.push("-y".into());

    // Add skip_frame for keyframe-only decoding
    cmd_args.extend(["-skip_frame".into(), "nokey".into()]);
    cmd_args.extend(hwaccel);

    // Add input for each seek point
    let mut seek_times = Vec::new();
    for i in 0..num_frames {
        let t = start + interval * i as f64;
        seek_times.push(t);
        cmd_args.extend([
            "-ss".into(),
            format!("{:.3}", t),
            "-i".into(),
            video_path.into(),
        ]);
    }

    // Add output mapping for each input
    let mut output_paths = Vec::new();
    for i in 0..num_frames {
        let out_path = output_dir.join(format!("frame_{}.webp", i));
        cmd_args.extend([
            "-map".into(),
            format!("{}:v", i),
            "-frames:v".into(),
            "1".into(),
            "-vf".into(),
            format!("scale={}:-1", frame_width),
            "-c:v".into(),
            "libwebp".into(),
            "-quality".into(),
            "80".into(),
            out_path.to_string_lossy().into_owned(),
        ]);
        output_paths.push(out_path);
    }

    let (program, args) = if cmd_args.len() > 1 && cmd_args[0] == "ionice" {
        (cmd_args[0].clone(), cmd_args[1..].to_vec())
    } else {
        (cmd_args[0].clone(), cmd_args[1..].to_vec())
    };

    let mut command = Command::new(&program);
    command
        .args(&args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    suppress_console_window(&mut command);
    let result = command.status();

    match result {
        Ok(status) if status.success() => {
            output_paths.retain(|p| p.exists());
            output_paths
        }
        _ => {
            // Clean up any partial frames
            for p in &output_paths {
                let _ = std::fs::remove_file(p);
            }
            vec![]
        }
    }
}

static PREVIEW_GENERATIONS: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();

pub(crate) struct PreviewGenerationGuard {
    key: PathBuf,
}

impl PreviewGenerationGuard {
    fn claim(key: PathBuf) -> Option<Self> {
        let generations = PREVIEW_GENERATIONS.get_or_init(|| Mutex::new(HashSet::new()));
        let mut active = generations.lock().unwrap();
        if !active.insert(key.clone()) {
            return None;
        }
        Some(Self { key })
    }
}

impl Drop for PreviewGenerationGuard {
    fn drop(&mut self) {
        if let Some(generations) = PREVIEW_GENERATIONS.get() {
            generations.lock().unwrap().remove(&self.key);
        }
    }
}

pub(crate) fn claim_preview_generation(
    data_dir: &Path,
    file_hash: &str,
) -> Option<PreviewGenerationGuard> {
    PreviewGenerationGuard::claim(get_preview_dir(data_dir, file_hash))
}

pub(crate) fn generate_video_previews_claimed(
    video_path: &str,
    file_hash: &str,
    data_dir: &Path,
    num_frames: usize,
    _guard: &PreviewGenerationGuard,
) -> Vec<PathBuf> {
    let output_dir = get_preview_dir(data_dir, file_hash);
    let existing = get_preview_frames(data_dir, file_hash);
    if !existing.is_empty() {
        return existing;
    }

    let suffix = chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default();
    let staging_dir = output_dir.with_file_name(format!(
        ".{}-{}-{}",
        output_dir.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id(),
        suffix
    ));
    let _ = std::fs::remove_dir_all(&staging_dir);
    let generated = extract_preview_frames(video_path, &staging_dir, num_frames, 400);
    if generated.len() != num_frames {
        let _ = std::fs::remove_dir_all(&staging_dir);
        return vec![];
    }
    if std::fs::write(staging_dir.join(".complete"), b"complete\n").is_err() {
        let _ = std::fs::remove_dir_all(&staging_dir);
        return vec![];
    }
    let _ = std::fs::remove_dir_all(&output_dir);
    if std::fs::rename(&staging_dir, &output_dir).is_err() {
        let _ = std::fs::remove_dir_all(&staging_dir);
        return vec![];
    }
    get_preview_frames(data_dir, file_hash)
}

/// Generate preview frames for a video file.
///
/// Returns paths to generated frame images, or empty vec on failure.
pub fn generate_video_previews(
    video_path: &str,
    file_hash: &str,
    data_dir: &Path,
    num_frames: usize,
) -> Vec<PathBuf> {
    let Some(guard) = claim_preview_generation(data_dir, file_hash) else {
        return vec![];
    };
    generate_video_previews_claimed(video_path, file_hash, data_dir, num_frames, &guard)
}

/// Generate a video thumbnail using ffmpeg.
///
/// Seeks to the middle of the video and extracts a single keyframe.
pub fn generate_video_thumbnail(video_path: &str, output_path: &str, size: u32) -> bool {
    if !check_ffmpeg_available() {
        return false;
    }

    // Get duration to seek to middle
    let seek_time = get_video_duration(video_path)
        .map(|d| if d > 1.0 { d / 2.0 } else { 0.5 })
        .unwrap_or(0.5);

    let mut cmd_args = get_low_priority_prefix();
    cmd_args.push("ffmpeg".into());
    cmd_args.extend(["-y".into(), "-skip_frame".into(), "nokey".into()]);
    cmd_args.extend(get_hwaccel_args());
    cmd_args.extend([
        "-ss".into(),
        format!("{:.3}", seek_time),
        "-i".into(),
        video_path.into(),
        "-vframes".into(),
        "1".into(),
        "-vsync".into(),
        "passthrough".into(),
        "-vf".into(),
        format!("scale={}:-1", size),
        "-c:v".into(),
        "libwebp".into(),
        "-quality".into(),
        "85".into(),
        output_path.into(),
    ]);

    let (program, args) = (cmd_args[0].clone(), cmd_args[1..].to_vec());

    let mut command = Command::new(&program);
    command
        .args(&args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    suppress_console_window(&mut command);
    command
        .status()
        .map(|s| s.success() && Path::new(output_path).exists())
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preview_cache_requires_a_complete_published_frame_set() {
        let root = std::env::temp_dir().join(format!(
            "localbooru-preview-complete-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let preview_dir = get_preview_dir(&root, "preview-hash");
        std::fs::create_dir_all(&preview_dir).unwrap();
        std::fs::write(preview_dir.join("frame_0.webp"), b"partial").unwrap();
        assert!(get_preview_frames(&root, "preview-hash").is_empty());

        for index in 1..8 {
            std::fs::write(preview_dir.join(format!("frame_{}.webp", index)), b"frame").unwrap();
        }
        assert!(get_preview_frames(&root, "preview-hash").is_empty());
        std::fs::write(preview_dir.join(".complete"), b"complete\n").unwrap();
        assert_eq!(get_preview_frames(&root, "preview-hash").len(), 8);
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn preview_generation_claim_is_single_flight() {
        let key = PathBuf::from(format!(
            "preview-claim-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let first = PreviewGenerationGuard::claim(key.clone()).unwrap();
        assert!(PreviewGenerationGuard::claim(key.clone()).is_none());
        drop(first);
        assert!(PreviewGenerationGuard::claim(key).is_some());
    }
}
