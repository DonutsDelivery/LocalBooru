//! A local read is authorized only by a recent real main-window file drop.
use std::collections::HashMap;
use std::fs::{self, File, Metadata};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime};

const MAX_AGE: Duration = Duration::from_secs(90);

struct Grant {
    canonical: PathBuf,
    granted: Instant,
    length: u64,
    modified: Option<SystemTime>,
    #[cfg(unix)]
    identity: (u64, u64),
}

fn grants() -> &'static Mutex<HashMap<PathBuf, Grant>> {
    static GRANTS: OnceLock<Mutex<HashMap<PathBuf, Grant>>> = OnceLock::new();
    GRANTS.get_or_init(|| Mutex::new(HashMap::new()))
}

fn file_type(path: &Path) -> Option<(&'static str, u64)> {
    let extension = path.extension()?.to_str()?.to_ascii_lowercase();
    let mime = match extension.as_str() {
        "json" => "application/json",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "bmp" => "image/bmp",
        "tif" | "tiff" => "image/tiff",
        _ => return None,
    };
    Some((
        mime,
        if mime == "application/json" { 4 } else { 32 } * 1024 * 1024,
    ))
}

#[cfg(unix)]
fn identity(metadata: &Metadata) -> (u64, u64) {
    use std::os::unix::fs::MetadataExt;
    (metadata.dev(), metadata.ino())
}

pub fn grant_drop(paths: &[PathBuf]) {
    let Ok(mut allowed) = grants().lock() else {
        return;
    };
    allowed.clear();
    for path in paths.iter().take(16) {
        let Some((_, limit)) = file_type(path) else {
            continue;
        };
        let Ok(canonical) = path.canonicalize() else {
            continue;
        };
        let Ok(metadata) = fs::metadata(&canonical) else {
            continue;
        };
        if !metadata.is_file() || metadata.len() == 0 || metadata.len() > limit {
            continue;
        }
        allowed.insert(
            path.clone(),
            Grant {
                canonical,
                granted: Instant::now(),
                length: metadata.len(),
                modified: metadata.modified().ok(),
                #[cfg(unix)]
                identity: identity(&metadata),
            },
        );
    }
}

pub fn read_drop(path: &Path) -> Result<(String, String, Vec<u8>), String> {
    // Consume before reading, including failed reads: callers cannot replay a
    // drop grant or use it as a lasting filesystem capability.
    let grant = grants()
        .lock()
        .map_err(|_| "The file drop is unavailable.")?
        .remove(path)
        .ok_or("Drop this file into the studio before reading it.")?;
    if grant.granted.elapsed() > MAX_AGE {
        return Err("The file drop expired. Drop the file again.".into());
    }
    if path.canonicalize().ok().as_ref() != Some(&grant.canonical) {
        return Err("The dropped file changed. Drop it again.".into());
    }
    let (mime, limit) = file_type(path).ok_or("Drop a supported image or workflow JSON.")?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("The dropped filename is invalid.")?
        .to_owned();
    let mut file =
        File::open(&grant.canonical).map_err(|_| "The dropped file could not be opened.")?;
    let metadata = file
        .metadata()
        .map_err(|_| "The dropped file could not be inspected.")?;
    if !metadata.is_file()
        || metadata.len() != grant.length
        || metadata.modified().ok() != grant.modified
        || metadata.len() > limit
    {
        return Err("The dropped file changed. Drop it again.".into());
    }
    #[cfg(unix)]
    if identity(&metadata) != grant.identity {
        return Err("The dropped file changed. Drop it again.".into());
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    (&mut file)
        .take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "The dropped file could not be read.")?;
    if bytes.len() as u64 != grant.length || bytes.len() as u64 > limit {
        return Err("The dropped file changed or exceeds its size limit.".into());
    }
    let after = file
        .metadata()
        .map_err(|_| "The dropped file could not be inspected after reading.")?;
    if after.len() != grant.length || after.modified().ok() != grant.modified {
        return Err("The dropped file changed while being read. Drop it again.".into());
    }
    Ok((name, mime.to_owned(), bytes))
}
