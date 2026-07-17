use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock as StdRwLock};

use tokio::sync::RwLock;

use crate::addons::manager::AddonManager;
use crate::db::directory_db::DirectoryDbManager;
use crate::db::library::{LibraryContext, LibraryManager};
use crate::db::pool::DbPool;
use crate::routes::cast::CastState;
use crate::routes::migration::{create_migration_state, SharedMigrationState};
use crate::routes::models::{create_model_registry, ModelRegistry};
use crate::routes::network::{HandshakeManager, SharedHandshakeManager};
use crate::routes::share::{create_share_sessions, ShareSessions};
use crate::routes::svp_web::{create_download_registry, WebDownloadRegistry};
use crate::services::directory_watcher::DirectoryWatcher;
use crate::services::events::{create_events, SharedEvents};
use crate::services::rate_limit::RateLimiter;
use crate::services::task_queue::BackgroundTaskQueue;
use crate::services::transcode::TranscodeManager;
use crate::svp_manager_snapshot::ManagerGraphSnapshotStore;

/// Shared application state available to all axum handlers.
#[derive(Clone)]
pub struct AppState {
    inner: Arc<AppStateInner>,
}

struct AppStateInner {
    /// Library manager (primary + auxiliary libraries)
    library_manager: LibraryManager,
    /// Server port
    port: u16,
    /// Per-install JWT signing secret (loaded from or generated into settings.json)
    jwt_secret: String,
    /// Event broadcasters (SSE)
    events: SharedEvents,
    /// Background task queue
    task_queue: Arc<BackgroundTaskQueue>,
    /// Addon manager (sidecar lifecycle + registry)
    addon_manager: AddonManager,
    /// Transcode manager (FFmpeg HLS streaming)
    transcode_manager: TranscodeManager,
    /// Rate limiter (in-memory, per-IP sliding window)
    rate_limiter: Arc<RateLimiter>,
    /// Active media share sessions (token -> session)
    share_sessions: ShareSessions,
    /// Cast/Chromecast session state
    cast_state: Arc<RwLock<CastState>>,
    /// Active web video downloads (yt-dlp), keyed by download_id
    web_download_registry: WebDownloadRegistry,
    /// ML model download state registry, keyed by model name
    model_registry: ModelRegistry,
    /// Data migration state (per-directory <-> main DB)
    migration_state: SharedMigrationState,
    /// Network handshake nonce manager (SSL pinning / QR verification)
    handshake_manager: SharedHandshakeManager,
    /// Shared HTTP client (connection pool reused across requests)
    http_client: reqwest::Client,
    /// Immutable SVP Manager graph snapshots trusted by desktop session routes.
    manager_graph_snapshots: ManagerGraphSnapshotStore,
    /// Directory watcher (set after AppState construction to break circular dep)
    directory_watcher: std::sync::OnceLock<Arc<DirectoryWatcher>>,
    /// Tauri asset-protocol scope (set during app setup). Used to grant `asset://`
    /// read access to watch directories dynamically. The scope is append-only and
    /// `forbid` is permanent, so we only ever `allow_directory` here — a removed
    /// watch dir stays asset-readable until the next launch (config scope is `[]`).
    asset_scope: std::sync::OnceLock<tauri::scope::fs::Scope>,
    /// Explicitly selected non-library media, keyed by an unguessable URL token.
    direct_files: StdRwLock<HashMap<String, PathBuf>>,
    /// Family mode lock state (true = locked, hides non-family-safe content)
    family_mode_locked: AtomicBool,
    /// Whether the HTTP server is listening and accepting connections
    server_ready: AtomicBool,
    /// Remote server proxy target. When set, /remote/* requests are forwarded to this server,
    /// retrying on `fallback_url` if the primary fails with a network-level error.
    remote_proxy: RwLock<Option<RemoteProxyConfig>>,
}

/// Remote proxy target with primary + optional fallback URL. The fallback is used
/// only on connect/network errors from the primary, not HTTP error responses —
/// a 5xx from the primary means the server is reachable, just unhappy.
#[derive(Clone, Debug)]
pub struct RemoteProxyConfig {
    pub primary_url: String,
    pub fallback_url: Option<String>,
    pub token: Option<String>,
}

/// Load the JWT secret from `settings.json` in `data_dir`, or generate a new
/// one if absent. The secret is persisted so tokens survive server restarts.
fn load_or_generate_jwt_secret(data_dir: &Path) -> Result<String, Box<dyn std::error::Error>> {
    let settings_path = data_dir.join("settings.json");

    // Try to load existing secret from settings.json
    if settings_path.exists() {
        let contents = std::fs::read_to_string(&settings_path)?;
        if let Ok(mut obj) = serde_json::from_str::<serde_json::Value>(&contents) {
            if let Some(secret) = obj.get("jwt_secret").and_then(|v| v.as_str()) {
                if !secret.is_empty() {
                    return Ok(secret.to_owned());
                }
            }

            // settings.json exists but has no jwt_secret — generate and merge
            let secret = generate_jwt_secret();
            obj.as_object_mut()
                .ok_or("settings.json is not a JSON object")?
                .insert(
                    "jwt_secret".into(),
                    serde_json::Value::String(secret.clone()),
                );
            std::fs::write(&settings_path, serde_json::to_string_pretty(&obj)?)?;
            return Ok(secret);
        }
    }

    // No settings.json at all — create one with just the secret
    let secret = generate_jwt_secret();
    let obj = serde_json::json!({ "jwt_secret": secret });
    std::fs::write(&settings_path, serde_json::to_string_pretty(&obj)?)?;
    Ok(secret)
}

/// Generate a random 32-byte hex-encoded JWT secret.
fn generate_jwt_secret() -> String {
    use rand::Rng;
    let bytes: [u8; 32] = rand::thread_rng().gen();
    bytes.iter().map(|b| format!("{:02x}", b)).collect()
}

/// Determine whether family mode should start locked based on settings.json.
/// Returns true if family_mode.enabled && family_mode.auto_lock_on_start.
fn load_family_mode_initial_lock(data_dir: &Path) -> bool {
    let settings_path = data_dir.join("settings.json");
    if !settings_path.exists() {
        return false;
    }
    let contents = match std::fs::read_to_string(&settings_path) {
        Ok(c) => c,
        Err(_) => return false,
    };
    let obj: serde_json::Value = match serde_json::from_str(&contents) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let fm = match obj.get("family_mode") {
        Some(v) => v,
        None => return false,
    };
    let enabled = fm.get("enabled").and_then(|v| v.as_bool()).unwrap_or(false);
    let auto_lock = fm
        .get("auto_lock_on_start")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);
    enabled && auto_lock
}

impl AppState {
    /// Create new AppState, initializing database pools and schema.
    pub fn new(data_dir: &Path, port: u16) -> Result<Self, Box<dyn std::error::Error>> {
        let snapshots = ManagerGraphSnapshotStore::new(data_dir.join("svp-manager-snapshots"));
        Self::new_with_manager_graph_snapshots(data_dir, port, snapshots)
    }

    pub fn new_with_manager_graph_snapshots(
        data_dir: &Path,
        port: u16,
        manager_graph_snapshots: ManagerGraphSnapshotStore,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        // Ensure data directory exists
        std::fs::create_dir_all(data_dir)?;

        // Create primary library context (opens/creates DB, loads UUID)
        let primary = LibraryContext::open(data_dir, "Local Library")?;
        let library_manager = LibraryManager::new(primary);

        // Load or generate per-install JWT secret
        let jwt_secret = load_or_generate_jwt_secret(data_dir)?;

        // Create event broadcasters
        let events = create_events();

        // Create task queue
        let task_queue = Arc::new(BackgroundTaskQueue::new());

        // Create addon manager
        let addon_manager = AddonManager::new(data_dir);

        // Create transcode manager
        let transcode_manager = TranscodeManager::new();

        // Create rate limiter
        let rate_limiter = Arc::new(RateLimiter::new());

        // Create share sessions map
        let share_sessions = create_share_sessions();

        // Create cast state
        let cast_state = Arc::new(RwLock::new(CastState::new()));

        // Create web download registry (SVP web video / yt-dlp)
        let web_download_registry = create_download_registry();

        // Create ML model registry
        let model_registry = create_model_registry();

        // Create migration state
        let migration_state = create_migration_state();

        // Create handshake nonce manager
        let handshake_manager = Arc::new(HandshakeManager::new());

        // Create shared HTTP client (connection pool reused across requests).
        // connect_timeout caps how long we wait on TCP SYN before giving up — without
        // it the OS retries for ~75s on unreachable hosts, which would block the remote
        // proxy's primary→fallback retry path on Tailscale-only devices. 5s is generous
        // enough for cellular-over-VPN handshakes while still failing fast on dead hosts.
        let http_client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(5))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());

        // Determine initial family mode lock state from settings
        let family_mode_locked = load_family_mode_initial_lock(data_dir);

        Ok(Self {
            inner: Arc::new(AppStateInner {
                library_manager,
                port,
                jwt_secret,
                events,
                task_queue,
                addon_manager,
                transcode_manager,
                rate_limiter,
                share_sessions,
                cast_state,
                web_download_registry,
                model_registry,
                migration_state,
                handshake_manager,
                http_client,
                manager_graph_snapshots,
                directory_watcher: std::sync::OnceLock::new(),
                asset_scope: std::sync::OnceLock::new(),
                direct_files: StdRwLock::new(HashMap::new()),
                family_mode_locked: AtomicBool::new(family_mode_locked),
                server_ready: AtomicBool::new(false),
                remote_proxy: RwLock::new(None),
            }),
        })
    }

    // ── Primary library backward-compatible accessors ────────────────────────

    /// Get the main library database pool (primary library).
    pub fn main_db(&self) -> &DbPool {
        &self.inner.library_manager.primary().main_pool
    }

    /// Get the directory database manager (primary library).
    pub fn directory_db(&self) -> &DirectoryDbManager {
        &self.inner.library_manager.primary().directory_db
    }

    /// Get the data directory path (primary library).
    pub fn data_dir(&self) -> &Path {
        &self.inner.library_manager.primary().data_dir
    }

    /// Get the thumbnails directory path (primary library).
    pub fn thumbnails_dir(&self) -> PathBuf {
        self.inner.library_manager.primary().thumbnails_dir()
    }

    // ── Multi-library accessors ─────────────────────────────────────────────

    /// Get the library manager.
    pub fn library_manager(&self) -> &LibraryManager {
        &self.inner.library_manager
    }

    /// Mount auxiliary libraries configured with `auto_mount = 1`.
    ///
    /// This performs filesystem and SQLite I/O and must run on a blocking worker,
    /// never on the HTTP-serving async runtime or Tauri's setup thread.
    pub fn auto_mount_libraries(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let conn = self.inner.library_manager.primary().main_pool.get()?;
        let mut stmt = conn.prepare(
            "SELECT uuid, name, path FROM mounted_libraries WHERE auto_mount = 1 ORDER BY mount_order"
        )?;
        let libraries: Vec<(String, String, String)> = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .collect();
        drop(stmt);
        drop(conn);

        for (uuid, name, path) in libraries {
            let lib_path = PathBuf::from(&path);
            // An unmounted external drive may leave an empty mount-point folder
            // behind. The folder existing is not enough: only mount a library
            // whose database marker is actually present.
            if !lib_path.join("library.db").is_file() {
                log::warn!(
                    "[Libraries] Skipping auto-mount for '{}' at {}: library.db is unavailable",
                    name,
                    path
                );
                continue;
            }
            match LibraryContext::open(&lib_path, &name) {
                Ok(ctx) => {
                    if ctx.uuid != uuid {
                        log::warn!(
                            "[Libraries] UUID mismatch for '{}': expected {}, got {}",
                            name,
                            uuid,
                            ctx.uuid
                        );
                    }
                    self.inner.library_manager.mount(ctx);
                    if let Ok(conn) = self.inner.library_manager.primary().main_pool.get() {
                        let _ = conn.execute(
                            "UPDATE mounted_libraries SET last_mounted_at = datetime('now') WHERE uuid = ?1",
                            rusqlite::params![uuid],
                        );
                    }
                    log::info!("[Libraries] Auto-mounted library '{}' from {}", name, path);
                }
                Err(e) => {
                    log::warn!(
                        "[Libraries] Failed to auto-mount library '{}' at {}: {}",
                        name,
                        path,
                        e
                    );
                }
            }
        }

        Ok(())
    }

    /// Resolve a library by UUID. Returns the primary library when `library_id`
    /// is `None` or `"primary"`. Returns 404 error if the library is not found
    /// or not mounted.
    pub fn resolve_library(
        &self,
        library_id: Option<&str>,
    ) -> Result<Arc<LibraryContext>, crate::server::error::AppError> {
        match library_id {
            None | Some("primary") => Ok(self.inner.library_manager.primary().clone()),
            Some(uuid) => self.inner.library_manager.get(uuid).ok_or_else(|| {
                crate::server::error::AppError::NotFound(format!(
                    "Library '{}' not found or not mounted",
                    uuid
                ))
            }),
        }
    }

    // ── Other accessors (unchanged) ─────────────────────────────────────────

    /// Get the server port.
    pub fn port(&self) -> u16 {
        self.inner.port
    }

    /// Get the per-install JWT signing secret.
    pub fn jwt_secret(&self) -> &str {
        &self.inner.jwt_secret
    }

    /// Get the event broadcasters.
    pub fn events(&self) -> Option<&SharedEvents> {
        Some(&self.inner.events)
    }

    /// Get the background task queue.
    pub fn task_queue(&self) -> Option<&BackgroundTaskQueue> {
        Some(&*self.inner.task_queue)
    }

    /// Get the task queue Arc (for starting the worker).
    pub fn task_queue_arc(&self) -> Arc<BackgroundTaskQueue> {
        self.inner.task_queue.clone()
    }

    /// Get the addon manager.
    pub fn addon_manager(&self) -> &AddonManager {
        &self.inner.addon_manager
    }

    /// Get the transcode manager.
    pub fn transcode_manager(&self) -> &TranscodeManager {
        &self.inner.transcode_manager
    }

    /// Get the rate limiter.
    pub fn rate_limiter(&self) -> &RateLimiter {
        &self.inner.rate_limiter
    }

    /// Get the active share sessions map.
    pub fn share_sessions(&self) -> &ShareSessions {
        &self.inner.share_sessions
    }

    /// Get the cast/Chromecast session state.
    pub fn cast_state(&self) -> &Arc<RwLock<CastState>> {
        &self.inner.cast_state
    }

    /// Get the web download registry (SVP web video / yt-dlp downloads).
    pub fn web_download_registry(&self) -> &WebDownloadRegistry {
        &self.inner.web_download_registry
    }

    /// Get the ML model download registry.
    pub fn model_registry(&self) -> &ModelRegistry {
        &self.inner.model_registry
    }

    /// Get the migration state (per-directory <-> main DB migration).
    pub fn migration_state(&self) -> &SharedMigrationState {
        &self.inner.migration_state
    }

    /// Get the network handshake nonce manager.
    pub fn handshake_manager(&self) -> &SharedHandshakeManager {
        &self.inner.handshake_manager
    }

    /// Get the shared HTTP client (reuses connection pool across requests).
    pub fn http_client(&self) -> &reqwest::Client {
        &self.inner.http_client
    }

    /// Get the currently trusted SVP Manager graph snapshot store.
    pub fn manager_graph_snapshots(&self) -> &ManagerGraphSnapshotStore {
        &self.inner.manager_graph_snapshots
    }

    /// Set the directory watcher (called once after AppState construction).
    pub fn set_directory_watcher(&self, watcher: Arc<DirectoryWatcher>) {
        let _ = self.inner.directory_watcher.set(watcher);
    }

    /// Get the directory watcher, if set.
    pub fn directory_watcher(&self) -> Option<&Arc<DirectoryWatcher>> {
        self.inner.directory_watcher.get()
    }

    /// Set the Tauri asset-protocol scope (called once during app setup).
    pub fn set_asset_scope(&self, scope: tauri::scope::fs::Scope) {
        let _ = self.inner.asset_scope.set(scope);
    }

    /// Grant `asset://` read access to a watch directory (recursive). No-op when
    /// the scope hasn't been set yet (e.g. headless/test contexts). Idempotent.
    pub fn allow_asset_dir(&self, path: &str) {
        if let Some(scope) = self.inner.asset_scope.get() {
            if let Err(e) = scope.allow_directory(path, true) {
                log::warn!("[AssetScope] Failed to allow '{}': {}", path, e);
            } else {
                log::debug!("[AssetScope] Allowed asset access to '{}'", path);
            }
        }
    }

    /// Register one user-selected file for Range-capable HTTP playback.
    pub fn register_direct_file(&self, path: &Path) -> String {
        let token = uuid::Uuid::new_v4().to_string();
        if let Ok(mut files) = self.inner.direct_files.write() {
            files.clear();
            files.insert(token.clone(), path.to_path_buf());
        }
        token
    }

    /// Resolve an unguessable direct-file token to its selected path.
    pub fn direct_file_path(&self, token: &str) -> Option<PathBuf> {
        self.inner.direct_files.read().ok()?.get(token).cloned()
    }

    /// Revoke a direct-file capability when its lightbox closes or is replaced.
    pub fn revoke_direct_file(&self, token: &str) {
        if let Ok(mut files) = self.inner.direct_files.write() {
            files.remove(token);
        }
    }

    /// Check if family mode is currently locked.
    pub fn is_family_mode_locked(&self) -> bool {
        self.inner.family_mode_locked.load(Ordering::Relaxed)
    }

    /// Set the family mode lock state.
    pub fn set_family_mode_locked(&self, locked: bool) {
        self.inner
            .family_mode_locked
            .store(locked, Ordering::Relaxed);
    }

    /// Check if the HTTP server is ready (listening on port).
    pub fn is_server_ready(&self) -> bool {
        self.inner.server_ready.load(Ordering::Relaxed)
    }

    /// Mark the HTTP server as ready.
    pub fn set_server_ready(&self, ready: bool) {
        self.inner.server_ready.store(ready, Ordering::Relaxed);
    }

    /// Set the remote proxy target for mobile remote-server mode.
    /// `url` is the primary URL; `fallback_url` is used on network failures.
    /// Passing `None` for `url` clears the proxy.
    pub async fn set_remote_proxy(
        &self,
        url: Option<String>,
        fallback_url: Option<String>,
        token: Option<String>,
    ) {
        let mut proxy = self.inner.remote_proxy.write().await;
        *proxy = url.map(|primary_url| RemoteProxyConfig {
            primary_url,
            fallback_url,
            token,
        });
    }

    /// Get the remote proxy target, if set.
    pub async fn get_remote_proxy(&self) -> Option<RemoteProxyConfig> {
        self.inner.remote_proxy.read().await.clone()
    }

    /// Check if local network access is enabled in settings.json.
    /// Used to determine whether to bind to 0.0.0.0 or 127.0.0.1.
    pub fn is_lan_enabled(&self) -> bool {
        let settings_path = self
            .inner
            .library_manager
            .primary()
            .data_dir
            .join("settings.json");
        let contents = match std::fs::read_to_string(&settings_path) {
            Ok(c) => c,
            Err(_) => return false,
        };
        let obj: serde_json::Value = match serde_json::from_str(&contents) {
            Ok(v) => v,
            Err(_) => return false,
        };
        obj.get("network")
            .and_then(|n| n.get("local_network_enabled"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_test_dir(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("localbooru-{}-{}", name, uuid::Uuid::new_v4()))
    }

    #[test]
    fn auxiliary_libraries_are_deferred_until_background_mount() {
        let primary_dir = temp_test_dir("primary");
        let auxiliary_dir = temp_test_dir("auxiliary");
        std::fs::create_dir_all(&primary_dir).unwrap();
        let auxiliary = LibraryContext::create(&auxiliary_dir, "Auxiliary").unwrap();
        let auxiliary_uuid = auxiliary.uuid.clone();
        drop(auxiliary);

        // First run: create and register a brand-new external library.
        let state = AppState::new(&primary_dir, 0).unwrap();
        {
            let conn = state.main_db().get().unwrap();
            conn.execute(
                "INSERT INTO mounted_libraries (uuid, name, path, auto_mount, mount_order)
                 VALUES (?1, 'Auxiliary', ?2, 1, 1)",
                rusqlite::params![auxiliary_uuid, auxiliary_dir.to_string_lossy()],
            )
            .unwrap();
        }
        drop(state);

        // Restart: constructing core state must not touch the external library.
        // The background mount phase opens it only after the server can bind.
        let restarted = AppState::new(&primary_dir, 0).unwrap();
        assert!(!restarted.library_manager().is_mounted(&auxiliary_uuid));
        restarted.auto_mount_libraries().unwrap();
        assert!(restarted.library_manager().is_mounted(&auxiliary_uuid));

        drop(restarted);
        let _ = std::fs::remove_dir_all(primary_dir);
        let _ = std::fs::remove_dir_all(auxiliary_dir);
    }

    #[test]
    fn auto_mount_does_not_create_database_in_empty_mount_point() {
        let primary_dir = temp_test_dir("primary-empty-mount");
        let mount_point = temp_test_dir("empty-mount");
        std::fs::create_dir_all(&primary_dir).unwrap();
        std::fs::create_dir_all(&mount_point).unwrap();

        let state = AppState::new(&primary_dir, 0).unwrap();
        let missing_uuid = uuid::Uuid::new_v4().to_string();
        {
            let conn = state.main_db().get().unwrap();
            conn.execute(
                "INSERT INTO mounted_libraries (uuid, name, path, auto_mount, mount_order)
                 VALUES (?1, 'Offline', ?2, 1, 1)",
                rusqlite::params![missing_uuid, mount_point.to_string_lossy()],
            )
            .unwrap();
        }

        state.auto_mount_libraries().unwrap();
        assert!(!state.library_manager().is_mounted(&missing_uuid));
        assert!(!mount_point.join("library.db").exists());

        drop(state);
        let _ = std::fs::remove_dir_all(primary_dir);
        let _ = std::fs::remove_dir_all(mount_point);
    }
}
