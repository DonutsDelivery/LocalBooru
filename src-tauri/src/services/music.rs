//! Local audio indexing for watched folders. Metadata extraction is deliberately
//! best effort: an untagged or malformed file still appears in Songs.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::UNIX_EPOCH;

use rusqlite::{params, OptionalExtension};
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::db::library::LibraryContext;
use crate::server::error::AppError;

const AUDIO_EXTENSIONS: &[&str] = &[
    "mp3", "flac", "m4a", "aac", "ogg", "opus", "wav", "aiff", "aif", "wma",
];

pub fn is_audio_file(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| AUDIO_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()))
}

fn tag<'a>(tags: &'a Value, names: &[&str]) -> Option<&'a str> {
    let obj = tags.as_object()?;
    obj.iter()
        .find(|(key, value)| {
            names.iter().any(|name| key.eq_ignore_ascii_case(name))
                && value.as_str().is_some_and(|s| !s.trim().is_empty())
        })
        .and_then(|(_, value)| value.as_str())
        .map(str::trim)
}

fn number(value: Option<&str>, fallback: i64) -> i64 {
    value
        .and_then(|s| s.split(|c: char| !c.is_ascii_digit()).next())
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(fallback)
}

fn artwork_for(path: &Path, lib: &LibraryContext, has_embedded_art: bool) -> Option<String> {
    let parent = path.parent()?;
    for name in [
        "cover.jpg",
        "folder.jpg",
        "front.jpg",
        "cover.png",
        "folder.png",
        "front.png",
    ] {
        let candidate = parent.join(name);
        if candidate.is_file() {
            return Some(candidate.to_string_lossy().to_string());
        }
    }
    if !has_embedded_art {
        return None;
    }
    let digest = Sha256::digest(path.to_string_lossy().as_bytes());
    let art_dir = lib.data_dir.join("music-artwork");
    std::fs::create_dir_all(&art_dir).ok()?;
    let art_path = art_dir.join(format!("{:x}.jpg", digest));
    if !art_path.is_file() {
        let status = Command::new("ffmpeg")
            .args(["-v", "error", "-y", "-i"])
            .arg(path)
            .args(["-map", "0:v:0", "-frames:v", "1"])
            .arg(&art_path)
            .status()
            .ok()?;
        if !status.success() {
            let _ = std::fs::remove_file(&art_path);
            return None;
        }
    }
    Some(art_path.to_string_lossy().to_string())
}

pub fn index_audio_file(
    lib: &LibraryContext,
    directory_id: i64,
    path: &Path,
) -> Result<(), AppError> {
    if !is_audio_file(path) || !path.is_file() {
        return Ok(());
    }
    let metadata = std::fs::metadata(path)?;
    let size = metadata.len() as i64;
    let modified = metadata
        .modified()
        .ok()
        .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_secs() as i64);
    let path_string = path.to_string_lossy().to_string();
    let mut conn = lib.main_pool.get()?;
    let current: Option<(i64, Option<i64>)> = conn
        .query_row(
            "SELECT file_size, modified_at FROM music_tracks WHERE path = ?1 AND is_available = 1",
            params![path_string],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    if current == Some((size, modified)) {
        return Ok(());
    }

    let probe = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
        ])
        .arg(path)
        .output();
    let data: Value = probe
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| serde_json::from_slice(&output.stdout).ok())
        .unwrap_or(Value::Null);
    let format = &data["format"];
    let tags = &format["tags"];
    let stream_tags = data["streams"]
        .as_array()
        .and_then(|streams| {
            streams
                .iter()
                .find(|stream| stream["codec_type"] == "audio")
        })
        .map(|stream| &stream["tags"])
        .unwrap_or(&Value::Null);
    let field = |names: &[&str]| tag(tags, names).or_else(|| tag(stream_tags, names));
    let fallback_title = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Untitled");
    let title = field(&["title"]).unwrap_or(fallback_title).to_string();
    let artist = field(&["artist", "ARTIST"])
        .unwrap_or("Unknown Artist")
        .to_string();
    let album_artist = field(&["album_artist", "albumartist"])
        .unwrap_or(&artist)
        .to_string();
    let album = field(&["album"]).unwrap_or("Unknown Album").to_string();
    let genre = field(&["genre"]).map(str::to_string);
    let year = number(field(&["date", "year"]), 0);
    let year = (year > 0).then_some(year);
    let disc = number(field(&["disc", "discnumber"]), 1);
    let track = number(field(&["track", "tracknumber"]), 0);
    let duration = format["duration"]
        .as_str()
        .and_then(|s| s.parse::<f64>().ok())
        .or_else(|| {
            data["streams"].as_array().and_then(|streams| {
                streams.iter().find_map(|stream| {
                    stream["duration"]
                        .as_str()
                        .and_then(|s| s.parse::<f64>().ok())
                })
            })
        });
    let has_embedded_art = data["streams"].as_array().is_some_and(|streams| {
        streams.iter().any(|stream| {
            stream["codec_type"] == "video"
                && (stream["disposition"]["attached_pic"] == 1 || stream["codec_name"].is_string())
        })
    });
    let artwork_path = artwork_for(path, lib, has_embedded_art);
    // Untagged files are separate fallback albums, so Songs never disappear
    // into an arbitrary folder grouping.
    let album_key = if album == "Unknown Album" {
        format!("untagged:{}", path_string)
    } else {
        format!(
            "{}\u{1f}{}\u{1f}{}",
            album_artist.to_lowercase(),
            album.to_lowercase(),
            year.unwrap_or(0)
        )
    };

    let tx = conn.transaction()?;
    tx.execute(
        "INSERT INTO music_albums (album_key, title, artist, genre, year, artwork_path)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)
         ON CONFLICT(album_key) DO UPDATE SET
           genre = COALESCE(music_albums.genre, excluded.genre),
           year = COALESCE(music_albums.year, excluded.year),
           artwork_path = COALESCE(music_albums.artwork_path, excluded.artwork_path)",
        params![album_key, album, album_artist, genre, year, artwork_path],
    )?;
    let album_id: i64 = tx.query_row(
        "SELECT id FROM music_albums WHERE album_key = ?1",
        params![album_key],
        |row| row.get(0),
    )?;
    tx.execute(
        "INSERT INTO music_tracks
         (path, directory_id, album_id, title, artist, album_artist, genre, year,
          disc_number, track_number, duration, file_size, modified_at, artwork_path, is_available)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, 1)
         ON CONFLICT(path) DO UPDATE SET
          directory_id=excluded.directory_id, album_id=excluded.album_id,
          title=excluded.title, artist=excluded.artist, album_artist=excluded.album_artist,
          genre=excluded.genre, year=excluded.year, disc_number=excluded.disc_number,
          track_number=excluded.track_number, duration=excluded.duration,
          file_size=excluded.file_size, modified_at=excluded.modified_at,
          artwork_path=excluded.artwork_path, is_available=1",
        params![
            path_string,
            directory_id,
            album_id,
            title,
            artist,
            album_artist,
            genre,
            year,
            disc,
            track,
            duration,
            size,
            modified,
            artwork_path
        ],
    )?;
    tx.commit()?;
    Ok(())
}

pub fn mark_audio_missing(lib: &LibraryContext, path: &Path) -> Result<(), AppError> {
    if !is_audio_file(path) {
        return Ok(());
    }
    let conn = lib.main_pool.get()?;
    conn.execute(
        "UPDATE music_tracks SET is_available = 0 WHERE path = ?1",
        params![path.to_string_lossy()],
    )?;
    Ok(())
}

pub fn reconcile_audio_files(lib: &LibraryContext, directory_id: i64) -> Result<(), AppError> {
    let conn = lib.main_pool.get()?;
    let mut stmt =
        conn.prepare("SELECT path FROM music_tracks WHERE directory_id = ?1 AND is_available = 1")?;
    let paths: Vec<PathBuf> = stmt
        .query_map(params![directory_id], |row| row.get::<_, String>(0))?
        .filter_map(Result::ok)
        .map(PathBuf::from)
        .collect();
    drop(stmt);
    for path in paths {
        if !path.is_file() {
            mark_audio_missing(lib, &path)?;
        }
    }
    Ok(())
}

/// Hide tracks when a watched folder is removed. Keep the rows so favorites
/// and music collection memberships return if the same files are watched again.
pub fn mark_directory_unavailable(lib: &LibraryContext, directory_id: i64) -> Result<(), AppError> {
    let conn = lib.main_pool.get()?;
    conn.execute(
        "UPDATE music_tracks SET is_available = 0 WHERE directory_id = ?1",
        params![directory_id],
    )?;
    Ok(())
}
