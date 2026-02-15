use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

/// Check if ffmpeg is available on the system.
pub fn check_ffmpeg_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        Command::new("ffmpeg")
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    })
}

/// Check if ffprobe is available on the system.
pub fn check_ffprobe_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        Command::new("ffprobe")
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    })
}

/// Get video duration in seconds using ffprobe.
pub fn get_video_duration(file_path: &str) -> Option<f64> {
    if !check_ffprobe_available() {
        return None;
    }

    let output = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            file_path,
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout.trim().parse::<f64>().ok()
}

/// Get video dimensions (width, height) using ffprobe.
pub fn get_video_dimensions(file_path: &str) -> Option<(i32, i32)> {
    if !check_ffprobe_available() {
        return None;
    }

    let output = Command::new("ffprobe")
        .args([
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            file_path,
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = stdout.trim().split(',').collect();
    if parts.len() == 2 {
        let w = parts[0].parse::<i32>().ok()?;
        let h = parts[1].parse::<i32>().ok()?;
        Some((w, h))
    } else {
        None
    }
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
                "ionice".into(), "-c".into(), "3".into(),
                "nice".into(), "-n".into(), "19".into(),
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
        let output = Command::new("ffmpeg")
            .args(["-hwaccels"])
            .output()
            .ok();

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

/// Get existing preview frame paths for a file hash.
pub fn get_preview_frames(data_dir: &Path, file_hash: &str) -> Vec<PathBuf> {
    let preview_dir = get_preview_dir(data_dir, file_hash);
    if !preview_dir.exists() {
        return vec![];
    }

    let mut frames: Vec<PathBuf> = (0..8)
        .map(|i| preview_dir.join(format!("frame_{}.webp", i)))
        .filter(|p| p.exists())
        .collect();
    frames.sort();
    frames
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
            "-ss".into(), format!("{:.3}", t),
            "-i".into(), video_path.into(),
        ]);
    }

    // Add output mapping for each input
    let mut output_paths = Vec::new();
    for i in 0..num_frames {
        let out_path = output_dir.join(format!("frame_{}.webp", i));
        cmd_args.extend([
            "-map".into(), format!("{}:v", i),
            "-frames:v".into(), "1".into(),
            "-vf".into(), format!("scale={}:-1", frame_width),
            "-c:v".into(), "libwebp".into(),
            "-quality".into(), "80".into(),
            out_path.to_string_lossy().into_owned(),
        ]);
        output_paths.push(out_path);
    }

    let (program, args) = if cmd_args.len() > 1 && cmd_args[0] == "ionice" {
        (cmd_args[0].clone(), cmd_args[1..].to_vec())
    } else {
        (cmd_args[0].clone(), cmd_args[1..].to_vec())
    };

    let result = Command::new(&program)
        .args(&args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();

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

/// Generate preview frames for a video file.
///
/// Returns paths to generated frame images, or empty vec on failure.
pub fn generate_video_previews(
    video_path: &str,
    file_hash: &str,
    data_dir: &Path,
    num_frames: usize,
) -> Vec<PathBuf> {
    // Check if previews already exist
    let existing = get_preview_frames(data_dir, file_hash);
    if !existing.is_empty() {
        return existing;
    }

    let output_dir = get_preview_dir(data_dir, file_hash);
    extract_preview_frames(video_path, &output_dir, num_frames, 400)
}

/// Generate a video thumbnail using ffmpeg.
///
/// Seeks to the middle of the video and extracts a single keyframe.
pub fn generate_video_thumbnail(
    video_path: &str,
    output_path: &str,
    size: u32,
) -> bool {
    if !check_ffmpeg_available() {
        return false;
    }

    // Get duration to seek to middle
    let seek_time = get_video_duration(video_path)
        .map(|d| if d > 1.0 { d / 2.0 } else { 0.5 })
        .unwrap_or(0.5);

    let mut cmd_args = get_low_priority_prefix();
    cmd_args.push("ffmpeg".into());
    cmd_args.extend([
        "-y".into(),
        "-skip_frame".into(), "nokey".into(),
    ]);
    cmd_args.extend(get_hwaccel_args());
    cmd_args.extend([
        "-ss".into(), format!("{:.3}", seek_time),
        "-i".into(), video_path.into(),
        "-vframes".into(), "1".into(),
        "-vsync".into(), "passthrough".into(),
        "-vf".into(), format!("scale={}:-1", size),
        "-c:v".into(), "libwebp".into(),
        "-quality".into(), "85".into(),
        output_path.into(),
    ]);

    let (program, args) = (cmd_args[0].clone(), cmd_args[1..].to_vec());

    Command::new(&program)
        .args(&args)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|s| s.success() && Path::new(output_path).exists())
        .unwrap_or(false)
}
