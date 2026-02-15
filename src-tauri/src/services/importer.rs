use std::path::{Path, PathBuf};

use rusqlite::params;
use serde_json::json;

use crate::server::error::AppError;
use crate::server::state::AppState;
use crate::services::events::event_type;
use crate::services::video_preview;

/// Video file extensions.
const VIDEO_EXTENSIONS: &[&str] = &["webm", "mp4", "mov", "avi", "mkv"];

/// Image file extensions.
const IMAGE_EXTENSIONS: &[&str] = &["png", "jpg", "jpeg", "gif", "webp", "bmp", "tiff", "tif"];

/// All supported media extensions.
pub fn all_extensions() -> Vec<&'static str> {
    let mut all = Vec::new();
    all.extend_from_slice(IMAGE_EXTENSIONS);
    all.extend_from_slice(VIDEO_EXTENSIONS);
    all
}

/// Check if a file is a video based on extension.
pub fn is_video_file(file_path: &str) -> bool {
    Path::new(file_path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| VIDEO_EXTENSIONS.contains(&e.to_lowercase().as_str()))
        .unwrap_or(false)
}

/// Check if a file is a supported media file.
pub fn is_media_file(path: &Path) -> bool {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());

    match ext {
        Some(e) if all_extensions().contains(&e.as_str()) => {
            // Filter out video thumbnails (e.g., video.mp4.png)
            !is_video_thumbnail(path)
        }
        _ => false,
    }
}

/// Check if a file is a video thumbnail (e.g., video.mp4.png).
pub fn is_video_thumbnail(path: &Path) -> bool {
    let name = match path.file_name().and_then(|n| n.to_str()) {
        Some(n) => n.to_lowercase(),
        None => return false,
    };

    let suffix = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();

    // Must end with an image extension
    if !IMAGE_EXTENSIONS.contains(&suffix.as_str()) {
        return false;
    }

    // Check if filename has 2+ dots (e.g., "video.mp4.png")
    name.chars().filter(|c| *c == '.').count() >= 2
}

// ─── Hashing ─────────────────────────────────────────────────────────────────

/// Calculate full xxhash of a file (reads entire file in 64KB chunks).
pub fn calculate_file_hash(file_path: &str) -> Result<String, std::io::Error> {
    use std::io::Read;

    let mut file = std::fs::File::open(file_path)?;
    let mut hasher = xxhash_rust::xxh64::Xxh64::new(0);
    let mut buffer = vec![0u8; 65536]; // 64KB chunks

    loop {
        let n = file.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        hasher.update(&buffer[..n]);
    }

    Ok(format!("{:016x}", hasher.digest()))
}

/// Calculate a quick hash from file size + first/last 64KB.
///
/// ~100x faster than full hash for large files.
/// Sufficient for duplicate detection in most cases.
pub fn calculate_quick_hash(file_path: &str) -> Result<String, std::io::Error> {
    use std::io::{Read, Seek, SeekFrom};

    let mut file = std::fs::File::open(file_path)?;
    let metadata = file.metadata()?;
    let file_size = metadata.len();
    let chunk_size: u64 = 65536;

    let mut hasher = xxhash_rust::xxh64::Xxh64::new(0);

    // Include file size in hash
    hasher.update(&file_size.to_le_bytes());

    // Read first chunk
    let first_read = chunk_size.min(file_size) as usize;
    let mut buffer = vec![0u8; first_read];
    file.read_exact(&mut buffer)?;
    hasher.update(&buffer);

    // Read last chunk (if file is large enough)
    if file_size > chunk_size * 2 {
        file.seek(SeekFrom::End(-(chunk_size as i64)))?;
        let mut last_buf = vec![0u8; chunk_size as usize];
        file.read_exact(&mut last_buf)?;
        hasher.update(&last_buf);
    }

    Ok(format!("{:016x}", hasher.digest()))
}

// ─── Image dimensions ────────────────────────────────────────────────────────

/// Get image dimensions (width, height). Uses image crate for images, ffprobe for videos.
pub fn get_image_dimensions(file_path: &str) -> Option<(u32, u32)> {
    if is_video_file(file_path) {
        video_preview::get_video_dimensions(file_path).map(|(w, h)| (w as u32, h as u32))
    } else {
        image::image_dimensions(file_path).ok()
    }
}

// ─── Thumbnail generation ────────────────────────────────────────────────────

/// Generate a thumbnail for an image file using the image crate.
pub fn generate_thumbnail(file_path: &str, output_path: &str, size: u32) -> bool {
    match image::open(file_path) {
        Ok(img) => {
            let thumb = img.thumbnail(size, size);
            let rgb = thumb.to_rgb8();

            // Save as WebP
            match rgb.save_with_format(output_path, image::ImageFormat::WebP) {
                Ok(()) => true,
                Err(e) => {
                    log::error!("Failed to save thumbnail: {}", e);
                    false
                }
            }
        }
        Err(e) => {
            log::error!("Failed to open image for thumbnail: {}", e);
            false
        }
    }
}

/// Find an existing thumbnail file for a video (media player naming conventions).
pub fn find_existing_thumbnail(video_path: &str) -> Option<PathBuf> {
    let video = Path::new(video_path);
    let video_dir = video.parent()?;
    let video_name = video.file_stem()?.to_str()?;
    let video_full = video.file_name()?.to_str()?;

    let image_exts = [".jpg", ".jpeg", ".png", ".webp"];
    let suffixes = ["", "-poster", "-thumb", "-fanart"];

    for suffix in &suffixes {
        for ext in &image_exts {
            // Pattern: video.mp4.jpg
            let candidate = video_dir.join(format!("{}{}{}", video_full, suffix, ext));
            if candidate.exists() {
                return Some(candidate);
            }
            // Pattern: video.jpg
            let candidate = video_dir.join(format!("{}{}{}", video_name, suffix, ext));
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    // Folder-level thumbnails
    for name in &["folder", "poster", "thumb", "cover"] {
        for ext in &image_exts {
            let candidate = video_dir.join(format!("{}{}", name, ext));
            if candidate.exists() {
                return Some(candidate);
            }
        }
    }

    None
}

/// Generate a thumbnail for a video file.
///
/// First checks for existing thumbnails (e.g., video.mp4.jpg) and uses those.
/// Falls back to ffmpeg if none found.
pub fn generate_video_thumbnail(video_path: &str, output_path: &str, size: u32) -> bool {
    // Check for existing thumbnail first
    if let Some(existing) = find_existing_thumbnail(video_path) {
        if let Ok(img) = image::open(&existing) {
            let thumb = img.thumbnail(size, size);
            if thumb
                .to_rgb8()
                .save_with_format(output_path, image::ImageFormat::WebP)
                .is_ok()
            {
                log::info!("Used existing thumbnail: {}", existing.display());
                return true;
            }
        }
    }

    // Fall back to ffmpeg
    video_preview::generate_video_thumbnail(video_path, output_path, size)
}

// ─── Import function ─────────────────────────────────────────────────────────

/// Result of an import operation.
#[derive(Debug)]
pub struct ImportResult {
    pub status: ImportStatus,
    pub image_id: Option<i64>,
    pub directory_id: Option<i64>,
    pub filename: Option<String>,
    pub message: Option<String>,
}

#[derive(Debug, PartialEq)]
pub enum ImportStatus {
    Imported,
    Duplicate,
    Error,
}

/// Import an image/video file by reference into a directory database.
///
/// This is the core import function. It:
/// 1. Checks for duplicates (by path, then by hash)
/// 2. Creates image + image_file records in the directory DB
/// 3. Generates thumbnails (spawned in background)
/// 4. Broadcasts IMAGE_ADDED event
///
/// Uses quick hash (first+last 64KB + size) for fast duplicate detection.
pub fn import_image(
    state: &AppState,
    file_path: &str,
    directory_id: i64,
) -> Result<ImportResult, AppError> {
    let path = Path::new(file_path);

    if !path.exists() {
        return Ok(ImportResult {
            status: ImportStatus::Error,
            image_id: None,
            directory_id: Some(directory_id),
            filename: None,
            message: Some("File not found".into()),
        });
    }

    if !path.is_file() {
        return Ok(ImportResult {
            status: ImportStatus::Error,
            image_id: None,
            directory_id: Some(directory_id),
            filename: None,
            message: Some("Not a file".into()),
        });
    }

    let dir_pool = state.directory_db().get_pool(directory_id)?;
    let dir_conn = dir_pool.get()?;

    // Check if path already imported
    let existing: Option<i64> = dir_conn
        .query_row(
            "SELECT image_id FROM image_files WHERE original_path = ?1",
            params![file_path],
            |row| row.get(0),
        )
        .ok();

    if existing.is_some() {
        return Ok(ImportResult {
            status: ImportStatus::Duplicate,
            image_id: existing,
            directory_id: Some(directory_id),
            filename: None,
            message: Some("Path already imported".into()),
        });
    }

    // Calculate quick hash
    let quick_hash = calculate_quick_hash(file_path)
        .map_err(|e| AppError::Internal(format!("Hash error: {}", e)))?;

    // Check for duplicate by hash
    let existing_by_hash: Option<i64> = dir_conn
        .query_row(
            "SELECT id FROM images WHERE file_hash = ?1",
            params![&quick_hash],
            |row| row.get(0),
        )
        .ok();

    if let Some(existing_id) = existing_by_hash {
        // Same hash, different path — add new file reference
        dir_conn.execute(
            "INSERT INTO image_files (image_id, original_path, file_exists, file_status) VALUES (?1, ?2, 1, 'available')",
            params![existing_id, file_path],
        )?;
        return Ok(ImportResult {
            status: ImportStatus::Duplicate,
            image_id: Some(existing_id),
            directory_id: Some(directory_id),
            filename: None,
            message: Some("Duplicate file (added new path reference)".into()),
        });
    }

    // Get file metadata
    let metadata = std::fs::metadata(file_path)
        .map_err(|e| AppError::Internal(format!("Stat error: {}", e)))?;
    let file_size = metadata.len() as i64;

    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    let filename = format!("{}.{}", &quick_hash[..16.min(quick_hash.len())], ext);
    let original_filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("")
        .to_string();
    let import_source = path
        .parent()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default();

    let is_video = is_video_file(file_path);

    // Get dimensions and duration
    let (width, height) = get_image_dimensions(file_path)
        .map(|(w, h)| (Some(w as i32), Some(h as i32)))
        .unwrap_or((None, None));
    let duration: Option<f64> = if is_video {
        video_preview::get_video_duration(file_path)
    } else {
        None
    };

    let now = chrono::Utc::now().to_rfc3339();

    // Insert image record
    dir_conn.execute(
        "INSERT INTO images (filename, original_filename, file_hash, width, height, file_size, duration, import_source, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            &filename,
            &original_filename,
            &quick_hash,
            width,
            height,
            file_size,
            duration,
            &import_source,
            &now,
        ],
    )?;

    let image_id = dir_conn.last_insert_rowid();

    // Insert file reference
    dir_conn.execute(
        "INSERT INTO image_files (image_id, original_path, file_exists, file_status) VALUES (?1, ?2, 1, 'available')",
        params![image_id, file_path],
    )?;

    // Generate thumbnail
    let thumbnails_dir = state.thumbnails_dir();
    std::fs::create_dir_all(&thumbnails_dir).ok();
    let thumb_name = format!("{}.webp", &quick_hash[..16.min(quick_hash.len())]);
    let thumb_path = thumbnails_dir.join(&thumb_name);

    if !thumb_path.exists() {
        if is_video {
            generate_video_thumbnail(file_path, &thumb_path.to_string_lossy(), 400);
        } else {
            generate_thumbnail(file_path, &thumb_path.to_string_lossy(), 400);
        }
    }

    // Broadcast event
    if let Some(events) = state.events() {
        events.library.broadcast(
            event_type::IMAGE_ADDED,
            json!({
                "image_id": image_id,
                "directory_id": directory_id,
                "filename": &filename,
                "thumbnail": format!("/api/images/{}/thumbnail?directory_id={}", image_id, directory_id)
            }),
        );
    }

    Ok(ImportResult {
        status: ImportStatus::Imported,
        image_id: Some(image_id),
        directory_id: Some(directory_id),
        filename: Some(filename),
        message: None,
    })
}
