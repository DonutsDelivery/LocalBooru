//! Local audio indexing for watched folders. Metadata extraction is deliberately
//! best effort: an untagged or malformed file still appears in Songs.

use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

use lofty::config::ParseOptions;
use lofty::file::{AudioFile, TaggedFileExt};
use lofty::picture::{Picture, PictureType};
use lofty::probe::Probe;
use lofty::tag::{Accessor, ItemKey};
use rusqlite::{params, OptionalExtension};
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

fn number(value: Option<&str>, fallback: i64) -> i64 {
    value
        .and_then(|s| s.split(|c: char| !c.is_ascii_digit()).next())
        .and_then(|s| s.parse::<i64>().ok())
        .unwrap_or(fallback)
}

fn folder_artwork(path: &Path) -> Option<String> {
    let parent = path.parent()?;
    const PREFERRED_NAMES: &[&str] = &[
        "cover.jpg",
        "cover.jpeg",
        "folder.jpg",
        "folder.jpeg",
        "front.jpg",
        "front.jpeg",
        "cover.png",
        "folder.png",
        "front.png",
        "cover.webp",
        "folder.webp",
        "front.webp",
        "albumart.jpg",
        "albumart.jpeg",
        "albumart.png",
        "albumart.webp",
    ];
    for name in PREFERRED_NAMES {
        let candidate = parent.join(name);
        if candidate.is_file() {
            return Some(candidate.to_string_lossy().to_string());
        }
    }
    // Filesystems on Linux are case-sensitive, while album art often arrives
    // with names such as Folder.jpg or Front.jpg.
    let mut images = std::fs::read_dir(parent)
        .ok()?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|candidate| {
            candidate.is_file()
                && candidate
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .is_some_and(|ext| {
                        matches!(
                            ext.to_ascii_lowercase().as_str(),
                            "jpg" | "jpeg" | "png" | "webp"
                        )
                    })
        })
        .collect::<Vec<_>>();
    images.sort();
    let named = |candidate: &Path| candidate.file_name()?.to_str().map(str::to_lowercase);
    for name in PREFERRED_NAMES {
        if let Some(candidate) = images
            .iter()
            .find(|candidate| named(candidate).as_deref() == Some(*name))
        {
            return Some(candidate.to_string_lossy().to_string());
        }
    }
    if let Some(candidate) = images.iter().find(|candidate| {
        named(candidate).is_some_and(|name| {
            (name.contains("cover") || name.contains("front") || name.contains("albumart"))
                && !name.contains("back")
                && !name.contains("inside")
                && !name.contains("booklet")
        })
    }) {
        return Some(candidate.to_string_lossy().to_string());
    }
    let plausible = |candidate: &Path| {
        named(candidate).is_some_and(|name| {
            !name.contains("back") && !name.contains("inside") && !name.contains("booklet")
        })
    };
    // A single image alongside the tracks is useful when the cover uses the
    // album title or catalog number as its filename.
    if images.len() == 1 && plausible(&images[0]) {
        return Some(images[0].to_string_lossy().to_string());
    }
    // When several images remain, use a uniquely square one. Otherwise a
    // random booklet or scan could be mistaken for the front cover.
    let mut square_images = images
        .iter()
        .filter(|candidate| plausible(candidate))
        .filter(|candidate| {
            image::image_dimensions(candidate).is_ok_and(|(width, height)| {
                height > 0 && ((width as f64 / height as f64) - 1.0).abs() <= 0.05
            })
        });
    let square = square_images.next()?;
    square_images
        .next()
        .is_none()
        .then(|| square.to_string_lossy().to_string())
}

fn artwork_for(path: &Path, lib: &LibraryContext, picture: Option<&Picture>) -> Option<String> {
    if let Some(cover) = folder_artwork(path) {
        return Some(cover);
    }
    let picture = picture?;
    // Convert once to a browser-supported image format. This also avoids
    // trusting a tag's MIME label, which is sometimes absent or incorrect.
    let mut hasher = Sha256::new();
    hasher.update(path.to_string_lossy().as_bytes());
    hasher.update(picture.data());
    let digest = hasher.finalize();
    let art_dir = lib.data_dir.join("music-artwork");
    std::fs::create_dir_all(&art_dir).ok()?;
    let art_path = art_dir.join(format!("{:x}.png", digest));
    if !art_path.is_file() {
        let image = image::load_from_memory(picture.data()).ok()?;
        let temp_path = art_dir.join(format!("{}.tmp", uuid::Uuid::new_v4()));
        if image
            .save_with_format(&temp_path, image::ImageFormat::Png)
            .is_err()
        {
            let _ = std::fs::remove_file(&temp_path);
            return None;
        }
        if std::fs::rename(&temp_path, &art_path).is_err() {
            let _ = std::fs::remove_file(&temp_path);
            if !art_path.is_file() {
                return None;
            }
        }
    }
    Some(art_path.to_string_lossy().to_string())
}

fn refresh_album_artwork(tx: &rusqlite::Transaction<'_>, album_id: i64) -> Result<(), AppError> {
    let mut stmt = tx.prepare(
        "SELECT id, artwork_path FROM music_tracks
         WHERE album_id=?1 AND is_available=1 ORDER BY disc_number,track_number,id",
    )?;
    let artwork = stmt
        .query_map(params![album_id], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, Option<String>>(1)?))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    drop(stmt);
    let mut cover: Option<String> = None;
    for (track_id, path) in artwork {
        if let Some(path) = path {
            if Path::new(&path).is_file() {
                if cover.is_none() {
                    cover = Some(path);
                }
            } else {
                tx.execute(
                    "UPDATE music_tracks SET artwork_path=NULL WHERE id=?1",
                    params![track_id],
                )?;
            }
        }
    }
    tx.execute(
        "UPDATE music_albums SET artwork_path=?1 WHERE id=?2",
        params![cover, album_id],
    )?;
    Ok(())
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
        .map(|duration| duration.as_millis() as i64);
    let path_string = path.to_string_lossy().to_string();
    let mut conn = lib.main_pool.get()?;
    let current: Option<(i64, Option<i64>, Option<String>, i64)> = conn
        .query_row(
            "SELECT file_size, modified_at, artwork_path, album_id FROM music_tracks WHERE path = ?1 AND is_available = 1",
            params![path_string],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .optional()?;
    if let Some((old_size, old_modified, old_artwork, _)) = &current {
        let current_cover = folder_artwork(path);
        let cover_changed = current_cover.as_deref() != old_artwork.as_deref()
            && (current_cover.is_some()
                || old_artwork
                    .as_deref()
                    .is_some_and(|art| !Path::new(art).is_file()));
        if *old_size == size && *old_modified == modified && !cover_changed {
            return Ok(());
        }
    }

    let tagged = lofty::read_from_path(path).ok();
    // Some containers have usable tags but malformed or missing audio properties.
    // Keep their metadata even when the full properties pass cannot complete.
    let tags_only = if tagged.as_ref().is_none_or(|file| file.tags().is_empty()) {
        Probe::open(path).ok().and_then(|probe| {
            probe
                .options(ParseOptions::new().read_properties(false))
                .read()
                .ok()
        })
    } else {
        None
    };
    let tag = tagged
        .as_ref()
        .filter(|file| !file.tags().is_empty())
        .or(tags_only.as_ref())
        .and_then(|file| {
            file.tags()
                .iter()
                .find(|tag| {
                    tag.title().is_some() || tag.artist().is_some() || tag.album().is_some()
                })
                .or_else(|| file.primary_tag())
                .or_else(|| file.first_tag())
        });
    let fallback_title = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("Untitled");
    let title = tag
        .and_then(|tag| tag.title())
        .as_deref()
        .unwrap_or(fallback_title)
        .to_string();
    let artist = tag
        .and_then(|tag| tag.artist())
        .as_deref()
        .unwrap_or("Unknown Artist")
        .to_string();
    let album_artist = tag
        .and_then(|tag| tag.get_string(&ItemKey::AlbumArtist))
        .unwrap_or(&artist)
        .to_string();
    let album = tag
        .and_then(|tag| tag.album())
        .as_deref()
        .unwrap_or("Unknown Album")
        .to_string();
    let genre = tag
        .and_then(|tag| tag.genre())
        .map(|value| value.to_string());
    let year = tag
        .and_then(|tag| tag.year())
        .map(i64::from)
        .unwrap_or_else(|| {
            number(
                tag.and_then(|tag| tag.get_string(&ItemKey::RecordingDate)),
                0,
            )
        });
    let year = (year > 0).then_some(year);
    let disc = tag
        .and_then(|tag| tag.disk())
        .map(i64::from)
        .unwrap_or_else(|| number(tag.and_then(|tag| tag.get_string(&ItemKey::DiscNumber)), 1));
    let track = tag
        .and_then(|tag| tag.track())
        .map(i64::from)
        .unwrap_or_else(|| number(tag.and_then(|tag| tag.get_string(&ItemKey::TrackNumber)), 0));
    let duration = tagged
        .as_ref()
        .map(|file| file.properties().duration().as_secs_f64());
    let picture = tag
        .and_then(|tag| {
            tag.get_picture_type(PictureType::CoverFront)
                .or_else(|| tag.pictures().first())
        })
        .or_else(|| {
            tagged
                .as_ref()
                .filter(|file| !file.tags().is_empty())
                .or(tags_only.as_ref())
                .and_then(|file| {
                    file.tags().iter().find_map(|tag| {
                        tag.get_picture_type(PictureType::CoverFront)
                            .or_else(|| tag.pictures().first())
                    })
                })
        });
    let artwork_path = artwork_for(path, lib, picture);
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
    refresh_album_artwork(&tx, album_id)?;
    if let Some((_, _, _, previous_album_id)) = current {
        if previous_album_id != album_id {
            refresh_album_artwork(&tx, previous_album_id)?;
        }
    }
    tx.commit()?;
    Ok(())
}

pub fn mark_audio_missing(lib: &LibraryContext, path: &Path) -> Result<(), AppError> {
    if !is_audio_file(path) {
        return Ok(());
    }
    let mut conn = lib.main_pool.get()?;
    let tx = conn.transaction()?;
    let album_id: Option<i64> = tx
        .query_row(
            "SELECT album_id FROM music_tracks WHERE path=?1",
            params![path.to_string_lossy()],
            |row| row.get(0),
        )
        .optional()?;
    tx.execute(
        "UPDATE music_tracks SET is_available = 0 WHERE path = ?1",
        params![path.to_string_lossy()],
    )?;
    if let Some(album_id) = album_id {
        refresh_album_artwork(&tx, album_id)?;
    }
    tx.commit()?;
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
    let mut conn = lib.main_pool.get()?;
    let tx = conn.transaction()?;
    let mut stmt = tx.prepare(
        "SELECT DISTINCT album_id FROM music_tracks WHERE directory_id=?1 AND is_available=1",
    )?;
    let album_ids = stmt
        .query_map(params![directory_id], |row| row.get::<_, i64>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    drop(stmt);
    tx.execute(
        "UPDATE music_tracks SET is_available = 0 WHERE directory_id = ?1",
        params![directory_id],
    )?;
    for album_id in album_ids {
        refresh_album_artwork(&tx, album_id)?;
    }
    tx.commit()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use lofty::config::WriteOptions;
    use lofty::picture::MimeType;
    use lofty::tag::{Tag, TagExt, TagType};

    fn wav_file(path: &Path) {
        // Two silent PCM samples in a canonical mono 8 kHz WAV container.
        let bytes = [
            b'R', b'I', b'F', b'F', 40, 0, 0, 0, b'W', b'A', b'V', b'E', b'f', b'm', b't', b' ',
            16, 0, 0, 0, 1, 0, 1, 0, 0x40, 0x1f, 0, 0, 0x80, 0x3e, 0, 0, 2, 0, 16, 0, b'd', b'a',
            b't', b'a', 4, 0, 0, 0, 0, 0, 0, 0,
        ];
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn indexes_tags_and_shared_cover_without_external_tools() {
        let root = tempfile::tempdir().unwrap();
        let lib = LibraryContext::create(root.path(), "Music test").unwrap();
        let music_dir = root.path().join("album");
        std::fs::create_dir(&music_dir).unwrap();
        let cover = music_dir.join("cover.png");
        image::RgbImage::new(2, 2).save(&cover).unwrap();

        let first = music_dir.join("first.wav");
        wav_file(&first);
        let mut tag = Tag::new(TagType::RiffInfo);
        tag.set_title("First song".into());
        tag.set_artist("Singer".into());
        tag.set_album("Release".into());
        tag.set_genre("Jazz".into());
        tag.set_year(2024);
        tag.set_track(4);
        tag.save_to_path(&first, WriteOptions::default()).unwrap();
        let parsed = Probe::open(&first)
            .unwrap()
            .options(ParseOptions::new().read_properties(false))
            .read()
            .unwrap();
        assert_eq!(
            parsed.tags().iter().find_map(|tag| tag.title()).as_deref(),
            Some("First song")
        );
        index_audio_file(&lib, 7, &first).unwrap();

        let second = music_dir.join("second.wav");
        wav_file(&second);
        let mut second_tag = Tag::new(TagType::RiffInfo);
        second_tag.set_title("Second song".into());
        second_tag.set_artist("Singer".into());
        second_tag.set_album("Release".into());
        second_tag.set_year(2024);
        second_tag
            .save_to_path(&second, WriteOptions::default())
            .unwrap();
        index_audio_file(&lib, 7, &second).unwrap();

        let conn = lib.main_pool.get().unwrap();
        let first_row: (String, String, String, Option<String>, Option<i64>, i64, i64, Option<String>) = conn
            .query_row(
                "SELECT t.title,t.artist,a.title,t.genre,t.year,t.disc_number,t.track_number,t.artwork_path
                 FROM music_tracks t JOIN music_albums a ON a.id=t.album_id WHERE t.path=?1",
                params![first.to_string_lossy()],
                |row| Ok((row.get(0)?,row.get(1)?,row.get(2)?,row.get(3)?,row.get(4)?,row.get(5)?,row.get(6)?,row.get(7)?)),
            ).unwrap();
        assert_eq!(first_row.0, "First song");
        assert_eq!(first_row.1, "Singer");
        assert_eq!(first_row.2, "Release");
        assert_eq!(first_row.3.as_deref(), Some("Jazz"));
        assert_eq!(first_row.4, Some(2024));
        assert_eq!((first_row.5, first_row.6), (1, 4));
        assert_eq!(first_row.7.as_deref(), cover.to_str());
        let second_artwork: Option<String> = conn
            .query_row(
                "SELECT artwork_path FROM music_tracks WHERE path=?1",
                params![second.to_string_lossy()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(second_artwork.as_deref(), cover.to_str());
        drop(conn);

        std::fs::remove_file(&cover).unwrap();
        index_audio_file(&lib, 7, &first).unwrap();
        let conn = lib.main_pool.get().unwrap();
        let remaining_art: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM music_tracks WHERE artwork_path IS NOT NULL",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(remaining_art, 0);
        let album_art: Option<String> = conn
            .query_row(
                "SELECT artwork_path FROM music_albums WHERE title='Release'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(album_art, None);
        drop(conn);

        let fallback = music_dir.join("untagged.wav");
        wav_file(&fallback);
        index_audio_file(&lib, 7, &fallback).unwrap();
        let conn = lib.main_pool.get().unwrap();
        let (title, album): (String, String) = conn
            .query_row(
                "SELECT t.title,a.title FROM music_tracks t JOIN music_albums a ON a.id=t.album_id
             WHERE t.path=?1",
                params![fallback.to_string_lossy()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(title, "untagged");
        assert_eq!(album, "Unknown Album");
    }

    #[test]
    fn embedded_artwork_cache_changes_with_picture_bytes() {
        let root = tempfile::tempdir().unwrap();
        let lib = LibraryContext::create(root.path(), "Art test").unwrap();
        let source = root.path().join("song.mp3");
        let picture = |color: image::Rgb<u8>| {
            let image = image::RgbImage::from_pixel(2, 2, color);
            let mut bytes = std::io::Cursor::new(Vec::new());
            image::DynamicImage::ImageRgb8(image)
                .write_to(&mut bytes, image::ImageFormat::Png)
                .unwrap();
            Picture::new_unchecked(
                PictureType::CoverFront,
                Some(MimeType::Png),
                None,
                bytes.into_inner(),
            )
        };
        let first = artwork_for(&source, &lib, Some(&picture(image::Rgb([255, 0, 0])))).unwrap();
        let second = artwork_for(&source, &lib, Some(&picture(image::Rgb([0, 0, 255])))).unwrap();
        assert_ne!(first, second);
        assert!(Path::new(&first).is_file());
        assert!(Path::new(&second).is_file());
        assert_ne!(
            std::fs::read(first).unwrap(),
            std::fs::read(second).unwrap()
        );
    }

    #[test]
    fn missing_cover_track_clears_album_artwork() {
        let root = tempfile::tempdir().unwrap();
        let lib = LibraryContext::create(root.path(), "Missing art test").unwrap();
        let art = root.path().join("art.png");
        image::RgbImage::new(2, 2).save(&art).unwrap();
        let conn = lib.main_pool.get().unwrap();
        conn.execute(
            "INSERT INTO music_albums (id,album_key,title,artist,artwork_path)
             VALUES (1,'album','Release','Artist',?1)",
            params![art.to_string_lossy()],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO music_tracks (id,path,directory_id,album_id,title,artist,artwork_path)
             VALUES (1,'/track/first.mp3',1,1,'First','Artist',?1)",
            params![art.to_string_lossy()],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO music_tracks (id,path,directory_id,album_id,title,artist)
             VALUES (2,'/track/second.mp3',1,1,'Second','Artist')",
            [],
        )
        .unwrap();
        drop(conn);
        mark_audio_missing(&lib, Path::new("/track/first.mp3")).unwrap();
        let conn = lib.main_pool.get().unwrap();
        let album_art: Option<String> = conn
            .query_row(
                "SELECT artwork_path FROM music_albums WHERE id=1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(album_art, None);
    }
}
