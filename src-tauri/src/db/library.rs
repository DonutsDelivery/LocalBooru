use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use crate::db::pool::{create_main_pool, DbPool};
use crate::db::directory_db::DirectoryDbManager;
use crate::db::schema::init_main_db;

/// Encapsulates everything needed for one library instance:
/// its own main database pool, directory database manager, and data directory.
pub struct LibraryContext {
    pub uuid: String,
    pub name: String,
    pub data_dir: PathBuf,
    pub main_pool: DbPool,
    pub directory_db: DirectoryDbManager,
}

impl LibraryContext {
    /// Open an existing library at the given data directory.
    /// The path must exist. The database and schema are initialized if needed.
    pub fn open(data_dir: &Path, name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        if !data_dir.exists() {
            return Err(format!("Library path does not exist: {}", data_dir.display()).into());
        }

        // Ensure subdirectories exist
        std::fs::create_dir_all(data_dir.join("thumbnails"))?;
        std::fs::create_dir_all(data_dir.join("directories"))?;

        // Create main database pool and initialize schema
        let main_pool = create_main_pool(data_dir)?;
        {
            let conn = main_pool.get()?;
            init_main_db(&conn)?;
        }

        // Create directory database manager
        let directory_db = DirectoryDbManager::new(data_dir);

        // Load or generate UUID
        let uuid = load_or_generate_library_uuid(data_dir)?;

        Ok(Self {
            uuid,
            name: name.to_string(),
            data_dir: data_dir.to_path_buf(),
            main_pool,
            directory_db,
        })
    }

    /// Create a new empty library at the given path.
    /// Creates the directory structure, database, and settings.
    pub fn create(data_dir: &Path, name: &str) -> Result<Self, Box<dyn std::error::Error>> {
        // Create directory structure
        std::fs::create_dir_all(data_dir)?;
        std::fs::create_dir_all(data_dir.join("thumbnails"))?;
        std::fs::create_dir_all(data_dir.join("directories"))?;

        // Create main database pool and initialize schema
        let main_pool = create_main_pool(data_dir)?;
        {
            let conn = main_pool.get()?;
            init_main_db(&conn)?;
        }

        // Create directory database manager
        let directory_db = DirectoryDbManager::new(data_dir);

        // Generate UUID for the new library
        let uuid = load_or_generate_library_uuid(data_dir)?;

        Ok(Self {
            uuid,
            name: name.to_string(),
            data_dir: data_dir.to_path_buf(),
            main_pool,
            directory_db,
        })
    }

    /// Get the thumbnails directory path for this library.
    pub fn thumbnails_dir(&self) -> PathBuf {
        self.data_dir.join("thumbnails")
    }
}

/// Manages all mounted libraries: one primary (always mounted) plus
/// any number of auxiliary libraries.
pub struct LibraryManager {
    primary: Arc<LibraryContext>,
    auxiliaries: RwLock<HashMap<String, Arc<LibraryContext>>>,
}

impl LibraryManager {
    /// Create a new LibraryManager with the given primary library.
    pub fn new(primary: LibraryContext) -> Self {
        Self {
            primary: Arc::new(primary),
            auxiliaries: RwLock::new(HashMap::new()),
        }
    }

    /// Get the primary library.
    pub fn primary(&self) -> &Arc<LibraryContext> {
        &self.primary
    }

    /// Get a library by UUID. Returns the primary for "primary" or its UUID,
    /// otherwise looks up auxiliaries.
    pub fn get(&self, uuid: &str) -> Option<Arc<LibraryContext>> {
        if uuid == "primary" || uuid == self.primary.uuid {
            return Some(self.primary.clone());
        }
        let auxiliaries = self.auxiliaries.read().unwrap();
        auxiliaries.get(uuid).cloned()
    }

    /// Get all mounted libraries (primary + auxiliaries).
    pub fn all_mounted(&self) -> Vec<Arc<LibraryContext>> {
        let mut result = vec![self.primary.clone()];
        let auxiliaries = self.auxiliaries.read().unwrap();
        result.extend(auxiliaries.values().cloned());
        result
    }

    /// Mount a library context. Returns the library's UUID.
    pub fn mount(&self, ctx: LibraryContext) -> String {
        let uuid = ctx.uuid.clone();
        let mut auxiliaries = self.auxiliaries.write().unwrap();
        auxiliaries.insert(uuid.clone(), Arc::new(ctx));
        uuid
    }

    /// Unmount a library by UUID. Returns true if it was mounted.
    pub fn unmount(&self, uuid: &str) -> bool {
        if uuid == "primary" || uuid == self.primary.uuid {
            return false; // Cannot unmount primary
        }
        let mut auxiliaries = self.auxiliaries.write().unwrap();
        auxiliaries.remove(uuid).is_some()
    }

    /// Check if a library is mounted.
    pub fn is_mounted(&self, uuid: &str) -> bool {
        if uuid == "primary" || uuid == self.primary.uuid {
            return true;
        }
        let auxiliaries = self.auxiliaries.read().unwrap();
        auxiliaries.contains_key(uuid)
    }
}

/// Load the library UUID from `settings.json` in `data_dir`, or generate a
/// new one if absent. The UUID is persisted so it survives across sessions.
fn load_or_generate_library_uuid(data_dir: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let settings_path = data_dir.join("settings.json");

    // Try to load existing UUID from settings.json
    if settings_path.exists() {
        let contents = std::fs::read_to_string(&settings_path)?;
        if let Ok(mut obj) = serde_json::from_str::<serde_json::Value>(&contents) {
            if let Some(uuid) = obj.get("library_uuid").and_then(|v| v.as_str()) {
                if !uuid.is_empty() {
                    return Ok(uuid.to_owned());
                }
            }

            // settings.json exists but has no library_uuid — generate and merge
            let uuid = generate_uuid_v4();
            obj.as_object_mut()
                .ok_or("settings.json is not a JSON object")?
                .insert("library_uuid".into(), serde_json::Value::String(uuid.clone()));
            std::fs::write(&settings_path, serde_json::to_string_pretty(&obj)?)?;
            return Ok(uuid);
        }
    }

    // No settings.json at all — create one with just the UUID
    let uuid = generate_uuid_v4();
    let obj = serde_json::json!({ "library_uuid": uuid });
    std::fs::write(&settings_path, serde_json::to_string_pretty(&obj)?)?;
    Ok(uuid)
}

/// Generate a random UUID v4 string.
fn generate_uuid_v4() -> String {
    use rand::Rng;
    let mut bytes: [u8; 16] = rand::thread_rng().gen();
    // Set version to 4
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    // Set variant to RFC 4122
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5],
        bytes[6], bytes[7],
        bytes[8], bytes[9],
        bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
    )
}
