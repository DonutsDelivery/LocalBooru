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
        let mut command =
            crate::platform_paths::media_command(crate::platform_paths::helper("ffmpeg"));
        command
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        suppress_console_window(&mut command);
        command.status().map(|s| s.success()).unwrap_or(false)
    })
}

fn probe_diagnostic(
    isolated: bool,
    helper: &std::ffi::OsStr,
    phase: &str,
    detail: &str,
) -> Option<String> {
    isolated.then(|| {
        let bounded: String = detail.chars().take(512).collect();
        format!("ffprobe diagnostic: helper={helper:?} phase={phase} detail={bounded}")
    })
}

fn isolated_probe_diagnostic(command: &Command, phase: &str, detail: &str) {
    if let Some(message) = probe_diagnostic(
        std::env::var_os("LOCALBOORU_ISOLATED_INSTANCE_ID").is_some(),
        command.get_program(),
        phase,
        detail,
    ) {
        eprintln!("{message}");
    }
}

fn probe_version_result(
    result: std::io::Result<std::process::ExitStatus>,
    mut diagnostic: impl FnMut(&str, &str),
) -> bool {
    match result {
        Ok(status) if status.success() => true,
        Ok(status) => {
            diagnostic("version status", &status.to_string());
            false
        }
        Err(error) => {
            diagnostic("version spawn", &format!("{error:?}"));
            false
        }
    }
}

#[cfg(not(target_os = "android"))]
fn parse_probe_metadata(stdout: &[u8]) -> Result<(i32, i32, f64), String> {
    // Preserve the previous parser's lossy UTF-8 treatment of unused fields.
    let stdout = String::from_utf8_lossy(stdout);
    let json: serde_json::Value =
        serde_json::from_str(&stdout).map_err(|e| format!("JSON parse: {}", e))?;
    let stream = json
        .get("streams")
        .and_then(|v| v.as_array())
        .and_then(|v| v.first())
        .ok_or("missing or invalid streams[0]")?;
    let width = stream
        .get("width")
        .and_then(|v| v.as_i64())
        .ok_or("missing or invalid width")? as i32;
    let height = stream
        .get("height")
        .and_then(|v| v.as_i64())
        .ok_or("missing or invalid height")? as i32;
    let duration_str = json
        .get("format")
        .and_then(|v| v.get("duration"))
        .and_then(|v| v.as_str())
        .ok_or("missing or invalid duration")?;
    let duration = duration_str
        .parse::<f64>()
        .map_err(|e| format!("duration parse: {}", e))?;
    Ok((width, height, duration))
}

/// Check if ffprobe is available on the system.
pub fn check_ffprobe_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let mut command =
            crate::platform_paths::media_command(crate::platform_paths::helper("ffprobe"));
        command
            .arg("-version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());
        suppress_console_window(&mut command);
        let result = command.status();
        probe_version_result(result, |phase, detail| {
            isolated_probe_diagnostic(&command, phase, detail)
        })
    })
}

/// Get video metadata (width, height, duration) in a single ffprobe call.
#[cfg(not(target_os = "android"))]
pub fn get_video_metadata(file_path: &str) -> Option<(i32, i32, f64)> {
    if !check_ffprobe_available() {
        return None;
    }

    let mut command =
        crate::platform_paths::media_command(crate::platform_paths::helper("ffprobe"));
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
    let result = command.output();
    probe_metadata_result(result, file_path, |phase, detail| {
        isolated_probe_diagnostic(&command, phase, detail)
    })
}

#[cfg(not(target_os = "android"))]
fn probe_metadata_result(
    result: std::io::Result<std::process::Output>,
    file_path: &str,
    mut diagnostic: impl FnMut(&str, &str),
) -> Option<(i32, i32, f64)> {
    let output = match result {
        Ok(output) => output,
        Err(error) => {
            diagnostic("metadata spawn", &format!("{error:?}"));
            return None;
        }
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).replace(file_path, "<media>");
        diagnostic(
            "metadata status",
            &format!("{} stderr={}", output.status, stderr),
        );
        return None;
    }
    match parse_probe_metadata(&output.stdout) {
        Ok(metadata) => Some(metadata),
        Err(error) => {
            diagnostic("metadata parse", &error);
            None
        }
    }
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
        let mut command =
            crate::platform_paths::media_command(crate::platform_paths::helper("ffmpeg"));
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
#[cfg(not(target_os = "android"))]
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
    let media_helper = crate::platform_paths::helper("ffmpeg");
    cmd_args.push(media_helper.to_string_lossy().into_owned());
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

    let mut command = crate::platform_paths::media_command_with_helper(&program, &media_helper);
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
const MAX_CONCURRENT_PREVIEW_GENERATIONS: usize = 2;

pub(crate) struct PreviewGenerationGuard {
    key: PathBuf,
}

impl PreviewGenerationGuard {
    fn claim(key: PathBuf) -> Option<Self> {
        let generations = PREVIEW_GENERATIONS.get_or_init(|| Mutex::new(HashSet::new()));
        let mut active = generations.lock().unwrap();
        if active.contains(&key) || active.len() >= MAX_CONCURRENT_PREVIEW_GENERATIONS {
            return None;
        }
        active.insert(key.clone());
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
    source_is_current: impl FnOnce() -> bool,
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
    if !publish_preview_staging(&staging_dir, &output_dir, source_is_current) {
        return vec![];
    }
    get_preview_frames(data_dir, file_hash)
}

fn publish_preview_staging(
    staging_dir: &Path,
    output_dir: &Path,
    source_is_current: impl FnOnce() -> bool,
) -> bool {
    if !source_is_current() {
        let _ = std::fs::remove_dir_all(&staging_dir);
        return false;
    }
    if std::fs::write(staging_dir.join(".complete"), b"complete\n").is_err() {
        let _ = std::fs::remove_dir_all(staging_dir);
        return false;
    }
    let _ = std::fs::remove_dir_all(output_dir);
    if std::fs::rename(staging_dir, output_dir).is_err() {
        let _ = std::fs::remove_dir_all(staging_dir);
        return false;
    }
    true
}

/// Generate a video thumbnail using ffmpeg.
///
/// Seeks to the middle of the video and extracts a single keyframe.
#[cfg(not(target_os = "android"))]
pub fn generate_video_thumbnail(video_path: &str, output_path: &str, size: u32) -> bool {
    if !check_ffmpeg_available() {
        log::warn!("Cannot generate video thumbnail: ffmpeg is unavailable");
        return false;
    }

    // Get duration to seek to middle
    let seek_time = get_video_duration(video_path)
        .map(|d| if d > 1.0 { d / 2.0 } else { 0.5 })
        .unwrap_or(0.5);

    // Decode in software. ffmpeg advertising CUDA does not mean a usable GPU
    // exists, and keyframe-only decoding can leave no frame at the seek time.
    // A single output frame needs neither -vsync (removed in newer ffmpeg) nor
    // hardware acceleration.
    for position in [seek_time, 0.0] {
        let mut cmd_args = get_low_priority_prefix();
        let media_helper = crate::platform_paths::helper("ffmpeg");
        cmd_args.push(media_helper.to_string_lossy().into_owned());
        cmd_args.extend([
            "-hide_banner".into(),
            "-loglevel".into(),
            "error".into(),
            "-y".into(),
            "-ss".into(),
            format!("{:.3}", position),
            "-i".into(),
            video_path.into(),
            "-frames:v".into(),
            "1".into(),
            "-an".into(),
            "-sn".into(),
            "-vf".into(),
            format!("scale={}:-1", size),
            "-c:v".into(),
            "libwebp".into(),
            "-quality".into(),
            "85".into(),
            output_path.into(),
        ]);

        let mut command =
            crate::platform_paths::media_command_with_helper(&cmd_args[0], &media_helper);
        command.args(&cmd_args[1..]);
        suppress_console_window(&mut command);
        match command.output() {
            Ok(output) if output.status.success() && Path::new(output_path).is_file() => {
                return true;
            }
            Ok(output) => log::warn!(
                "Video thumbnail extraction failed for {} at {:.3}s: {}",
                video_path,
                position,
                String::from_utf8_lossy(&output.stderr).trim()
            ),
            Err(error) => log::warn!(
                "Cannot start ffmpeg for video thumbnail {}: {}",
                video_path,
                error
            ),
        }
        let _ = std::fs::remove_file(output_path);
    }
    false
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
    fn preview_generation_claim_is_single_flight_and_bounded() {
        let first_key = PathBuf::from(format!(
            "preview-claim-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let second_key = first_key.with_extension("second");
        let waiting_key = first_key.with_extension("waiting");
        let first = PreviewGenerationGuard::claim(first_key.clone()).unwrap();
        assert!(PreviewGenerationGuard::claim(first_key).is_none());
        let second = PreviewGenerationGuard::claim(second_key).unwrap();
        assert!(PreviewGenerationGuard::claim(waiting_key.clone()).is_none());
        drop(first);
        assert!(PreviewGenerationGuard::claim(waiting_key).is_some());
        drop(second);
    }

    #[test]
    fn changed_source_discards_staging_before_preview_publication() {
        let root = std::env::temp_dir().join(format!(
            "localbooru-preview-validation-{}-{}",
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        let output_dir = root.join("published");
        let stale_staging = root.join("stale-staging");
        std::fs::create_dir_all(&output_dir).unwrap();
        std::fs::write(output_dir.join("existing"), b"existing").unwrap();
        std::fs::create_dir_all(&stale_staging).unwrap();
        std::fs::write(stale_staging.join("frame_0.webp"), b"stale").unwrap();

        assert!(!publish_preview_staging(
            &stale_staging,
            &output_dir,
            || false
        ));
        assert!(!stale_staging.exists());
        assert_eq!(
            std::fs::read(output_dir.join("existing")).unwrap(),
            b"existing"
        );

        let current_staging = root.join("current-staging");
        std::fs::create_dir_all(&current_staging).unwrap();
        std::fs::write(current_staging.join("frame_0.webp"), b"current").unwrap();
        assert!(publish_preview_staging(
            &current_staging,
            &output_dir,
            || true
        ));
        assert!(!current_staging.exists());
        assert_eq!(
            std::fs::read(output_dir.join("frame_0.webp")).unwrap(),
            b"current"
        );
        assert!(output_dir.join(".complete").is_file());
        let _ = std::fs::remove_dir_all(root);
    }
}

#[cfg(target_os = "android")]
pub fn get_video_metadata(file_path: &str) -> Option<(i32, i32, f64)> {
    crate::android_media::metadata(file_path)
}
#[cfg(target_os = "android")]
pub fn extract_preview_frames(
    video_path: &str,
    output_dir: &Path,
    num_frames: usize,
    frame_width: u32,
) -> Vec<PathBuf> {
    crate::android_media::previews(video_path, output_dir, num_frames, frame_width)
}
#[cfg(target_os = "android")]
pub fn generate_video_thumbnail(video_path: &str, output_path: &str, size: u32) -> bool {
    crate::android_media::thumbnail(video_path, output_path, size)
}

#[cfg(all(test, not(target_os = "android")))]
mod probe_diagnostic_tests {
    use super::{
        parse_probe_metadata, probe_diagnostic, probe_metadata_result, probe_version_result,
    };
    use std::ffi::OsStr;
    use std::process::{ExitStatus, Output};

    fn status(code: i32) -> ExitStatus {
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            ExitStatus::from_raw(code << 8)
        }
        #[cfg(windows)]
        {
            use std::os::windows::process::ExitStatusExt;
            ExitStatus::from_raw(code as u32)
        }
    }

    #[test]
    fn version_success_and_failures_keep_availability_semantics() {
        let mut events = Vec::new();
        assert!(probe_version_result(Ok(status(0)), |p, d| events.push(format!("{p}:{d}"))));
        assert!(events.is_empty());
        assert!(!probe_version_result(Ok(status(7)), |p, d| events.push(format!("{p}:{d}"))));
        assert!(events[0].starts_with("version status:"));
        events.clear();
        assert!(!probe_version_result(
            Err(std::io::Error::from_raw_os_error(2)),
            |p, d| events.push(format!("{p}:{d}"))
        ));
        assert!(events[0].starts_with("version spawn:"));
        assert!(events[0].contains("code: 2"));
    }

    #[test]
    fn metadata_spawn_failure_and_nonzero_exit_are_distinct() {
        let mut events = Vec::new();
        assert!(probe_metadata_result(
            Err(std::io::Error::from_raw_os_error(5)),
            "owned-media",
            |p, d| events.push(format!("{p}:{d}"))
        )
        .is_none());
        assert!(events[0].starts_with("metadata spawn:"));
        events.clear();
        let output = Output {
            status: status(1),
            stdout: Vec::new(),
            stderr: b"could not open owned-media".to_vec(),
        };
        assert!(
            probe_metadata_result(Ok(output), "owned-media", |p, d| events
                .push(format!("{p}:{d}")))
            .is_none()
        );
        assert!(events[0].starts_with("metadata status:"));
        assert!(events[0].contains("<media>"));
        assert!(!events[0].contains("owned-media"));
    }

    #[test]
    fn metadata_success_is_silent_and_parse_failure_is_observable() {
        let mut events = Vec::new();
        let output = |stdout: &[u8]| Output {
            status: status(0),
            stdout: stdout.to_vec(),
            stderr: Vec::new(),
        };
        assert_eq!(
            probe_metadata_result(
                Ok(output(
                    br#"{"streams":[{"width":320,"height":180}],"format":{"duration":"3.0"}}"#
                )),
                "owned-media",
                |p, d| events.push(format!("{p}:{d}"))
            ),
            Some((320, 180, 3.0))
        );
        assert!(events.is_empty());
        assert!(
            probe_metadata_result(Ok(output(b"{")), "owned-media", |p, d| events
                .push(format!("{p}:{d}")))
            .is_none()
        );
        assert!(events[0].starts_with("metadata parse:JSON parse"));
    }

    #[test]
    fn diagnostics_are_isolated_only_and_unicode_bounded() {
        let detail = "é".repeat(600);
        assert!(probe_diagnostic(false, OsStr::new("ffprobe"), "phase", &detail).is_none());
        let value = probe_diagnostic(true, OsStr::new("ffprobe"), "phase", &detail).unwrap();
        assert!(value.contains("helper=\"ffprobe\" phase=phase"));
        assert_eq!(value.matches('é').count(), 512);
    }

    #[test]
    fn preserves_lossy_utf8_in_unused_json_fields() {
        let mut stdout =
            br#"{"streams":[{"width":320,"height":180}],"format":{"duration":"3"},"unused":""#
                .to_vec();
        stdout.push(0xff);
        stdout.extend_from_slice(b"\"}");
        assert_eq!(parse_probe_metadata(&stdout).unwrap(), (320, 180, 3.0));
    }

    #[test]
    fn parses_metadata_and_reports_invalid_responses() {
        assert_eq!(
            parse_probe_metadata(
                br#"{"streams":[{"width":1920,"height":1080}],"format":{"duration":"2.5"}}"#
            )
            .unwrap(),
            (1920, 1080, 2.5)
        );
        assert!(parse_probe_metadata(b"{")
            .unwrap_err()
            .contains("JSON parse"));
        assert!(parse_probe_metadata(br#"{"streams":[]}"#)
            .unwrap_err()
            .contains("streams"));
        assert!(parse_probe_metadata(br#"{"streams":[{"width":1}]}"#)
            .unwrap_err()
            .contains("height"));
        assert!(parse_probe_metadata(
            br#"{"streams":[{"width":1,"height":2}],"format":{"duration":"bad"}}"#
        )
        .unwrap_err()
        .contains("duration parse"));
    }
}
