// SPDX-License-Identifier: GPL-3.0-only
//! Opted-in, source-scoped retained media over authenticated Iroh.
//! A hub bearer credential is never sent to another peer. Every serve requires
//! a signed recipient grant and a fresh check at the pinned hub.
use super::transport::{self, HubBootstrap, HubRequest, TransportMode};
use crate::server::state::AppState;
use base64::{engine::general_purpose::URL_SAFE_NO_PAD, Engine};
use iroh::{Endpoint, EndpointAddr, EndpointId, RelayMode, SecretKey, Signature};
use rand::RngCore;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs::OpenOptions,
    io::Write,
    path::{Path, PathBuf},
    sync::{Arc, Mutex as StdMutex, OnceLock, Weak},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::{
    io::{AsyncReadExt, AsyncSeekExt, AsyncWriteExt},
    sync::{Mutex, Semaphore},
    time::timeout,
};

pub const ALPN: &[u8] = b"donutmediacenter/peer-read/1";
const READ_PREFIX: &[u8] = b"DMC-HUB-PEER-READ-1\0";
const REPORT_PREFIX: &[u8] = b"DMC-HUB-PROVIDER-1\0";
const LIMIT: u64 = 2 * 1024 * 1024 * 1024;
const IDLE: Duration = Duration::from_secs(30);
const MAX_FRAME: usize = 32 * 1024;
static RUNTIME: OnceLock<StdMutex<Weak<Runtime>>> = OnceLock::new();
static AUTHORITIES: OnceLock<StdMutex<()>> = OnceLock::new();
type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Serialize, Deserialize)]
struct Authority {
    source_id: String,
    origin: String,
    instance_id: String,
    realm: String,
    signing_key: String,
    bootstrap: HubBootstrap,
    mode: TransportMode,
}
#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Grant {
    pub payload: String,
    pub signature: String,
}
#[derive(Deserialize)]
struct Claims {
    v: u8,
    iss: String,
    instance_id: String,
    realm: String,
    aud: String,
    sub: String,
    post_id: String,
    variant: String,
    sha256: String,
    byte_size: u64,
    revision: String,
    iat: u64,
    exp: u64,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestHead {
    version: u8,
    grant: Grant,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ResponseHead {
    version: u8,
    status: u16,
    byte_size: u64,
    sha256: String,
    mime_type: Option<String>,
}
#[derive(Clone, Serialize)]
pub struct ProviderIdentity {
    pub endpoint_id: String,
    pub relay_url: String,
}
struct Provider {
    endpoint: Endpoint,
    key: SecretKey,
}
struct Runtime {
    state: AppState,
    providers: Mutex<HashMap<String, Arc<Provider>>>,
    cancelled: tokio_util::sync::CancellationToken,
    permits: Arc<Semaphore>,
}
/// App-owned lifetime; the global lookup and task loops keep only Weak references.
pub struct PeerRuntime {
    inner: Arc<Runtime>,
}
impl Drop for PeerRuntime {
    fn drop(&mut self) {
        self.inner.cancelled.cancel();
        if let Ok(mut lookup) = RUNTIME.get_or_init(|| StdMutex::new(Weak::new())).lock() {
            *lookup = Weak::new();
        }
    }
}

pub fn start(state: AppState) -> PeerRuntime {
    let inner = Arc::new(Runtime {
        state,
        providers: Mutex::new(HashMap::new()),
        cancelled: tokio_util::sync::CancellationToken::new(),
        permits: Arc::new(Semaphore::new(8)),
    });
    if let Ok(mut lookup) = RUNTIME.get_or_init(|| StdMutex::new(Weak::new())).lock() {
        *lookup = Arc::downgrade(&inner);
    }
    let weak = Arc::downgrade(&inner);
    let cancelled = inner.cancelled.clone();
    tauri::async_runtime::spawn(async move {
        let mut timer = tokio::time::interval(Duration::from_secs(45));
        timer.tick().await;
        loop {
            tokio::select! { _ = cancelled.cancelled() => break, _ = timer.tick() => {} }
            let Some(runtime) = weak.upgrade() else {
                break;
            };
            let state = runtime.state.clone();
            drop(runtime);
            let sources = (|| -> Result<Vec<String>> {
                let conn = super::offline::open(state.data_dir()).map_err(|e| e.to_string())?;
                let mut stmt = conn.prepare("SELECT o.source_id FROM offline_sources o JOIN remote_sources s USING(source_id) WHERE s.enabled=1").map_err(|e| e.to_string())?;
                let rows = stmt
                    .query_map([], |r| r.get(0))
                    .map_err(|e| e.to_string())?;
                rows.collect::<std::result::Result<Vec<_>, _>>()
                    .map_err(|e| e.to_string())
            })();
            if let Ok(sources) = sources {
                for sid in sources {
                    let result = tokio::select! { _ = cancelled.cancelled() => break, result = publish_retained(&state, &sid) => result };
                    if let Ok(conn) = super::offline::open(state.data_dir()) {
                        let _ = conn.execute(
                            "UPDATE offline_sources SET peer_error=?2 WHERE source_id=?1",
                            rusqlite::params![sid, result.err()],
                        );
                    }
                }
            }
        }
    });
    PeerRuntime { inner }
}
fn runtime() -> Result<Arc<Runtime>> {
    RUNTIME
        .get_or_init(|| StdMutex::new(Weak::new()))
        .lock()
        .map_err(|_| "Peer runtime unavailable")?
        .upgrade()
        .ok_or_else(|| "Peer runtime is not started".into())
}
fn now() -> Result<u64> {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|value| value.as_secs())
        .map_err(|_| "Invalid system clock".into())
}
fn hash_valid(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
fn private_write(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = tempfile::NamedTempFile::new_in(path.parent().ok_or("Missing parent")?)
        .map_err(|_| "Unable to create protected peer state")?;
    file.write_all(bytes)
        .map_err(|_| "Unable to save protected peer state")?;
    file.as_file()
        .sync_all()
        .map_err(|_| "Unable to flush peer state")?;
    file.persist(path)
        .map_err(|_| "Unable to commit protected peer state")?;
    Ok(())
}
fn authorities(data_dir: &Path) -> Result<Vec<Authority>> {
    let path = data_dir.join("online-peer-authorities.json");
    if !path.exists() {
        return Ok(vec![]);
    }
    let data = std::fs::read(path).map_err(|_| "Unable to read peer authority state")?;
    if data.len() > 1024 * 1024 {
        return Err("Peer authority state is oversized".into());
    }
    serde_json::from_slice(&data).map_err(|_| "Invalid peer authority state".into())
}
/// Only call after the descriptor was validated against the enrolled source and
/// pinned hub. Trust is persisted independently of a grant received from a peer.
#[allow(clippy::too_many_arguments)]
pub fn remember_authority(
    data_dir: &Path,
    source_id: &str,
    origin: &str,
    instance_id: &str,
    realm: &str,
    signing_key: &str,
    bootstrap: &HubBootstrap,
    mode: TransportMode,
) -> Result<()> {
    transport::validate_bootstrap(bootstrap)?;
    if !hash_valid(signing_key)
        || !matches!(realm, "main" | "adult")
        || mode == TransportMode::Anonymous
    {
        return Err("Unsupported peer authority".into());
    }
    signing_key
        .parse::<EndpointId>()
        .map_err(|_| "Invalid peer signing key")?;
    let _guard = AUTHORITIES
        .get_or_init(|| StdMutex::new(()))
        .lock()
        .map_err(|_| "Peer authority state busy")?;
    let mut known = authorities(data_dir)?;
    known.retain(|item| item.source_id != source_id);
    if known.len() >= 128 {
        return Err("Too many peer authorities".into());
    }
    known.push(Authority {
        source_id: source_id.into(),
        origin: origin.trim_end_matches('/').into(),
        instance_id: instance_id.into(),
        realm: realm.into(),
        signing_key: signing_key.into(),
        bootstrap: bootstrap.clone(),
        mode,
    });
    private_write(
        &data_dir.join("online-peer-authorities.json"),
        &serde_json::to_vec(&known).map_err(|_| "Unable to encode peer authorities")?,
    )
}
fn persistent_key(data_dir: &Path, label: &str) -> Result<SecretKey> {
    // Distinct stable keys prevent Standard and Relayed endpoints for the same
    // profile from competing for one relay identity. Relay URLs also scope it.
    let label_digest = format!("{:x}", Sha256::digest(label.as_bytes()));
    let path = data_dir.join(format!("online-peer-identity-{label_digest}.key"));
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    match options.open(&path) {
        Ok(mut file) => {
            let mut bytes = [0u8; 32];
            rand::rngs::OsRng.fill_bytes(&mut bytes);
            file.write_all(&bytes)
                .map_err(|_| "Unable to persist peer identity")?;
            file.sync_all()
                .map_err(|_| "Unable to flush peer identity")?;
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(_) => return Err("Unable to create persistent peer identity".into()),
    }
    let info = std::fs::symlink_metadata(&path).map_err(|_| "Unable to inspect peer identity")?;
    if !info.is_file() || info.len() != 32 {
        return Err("Invalid peer identity file".into());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if info.permissions().mode() & 0o777 != 0o600 {
            return Err("Peer identity must be private mode 0600".into());
        }
    }
    let bytes: [u8; 32] = std::fs::read(path)
        .map_err(|_| "Unable to read peer identity")?
        .try_into()
        .map_err(|_| "Invalid peer identity length")?;
    Ok(SecretKey::from_bytes(&bytes))
}

pub async fn ensure_provider(
    data_dir: &Path,
    bootstrap: &HubBootstrap,
    mode: TransportMode,
) -> Result<ProviderIdentity> {
    let provider = provider(data_dir, bootstrap, mode).await?;
    Ok(ProviderIdentity {
        endpoint_id: provider.endpoint.id().to_string(),
        relay_url: bootstrap.relay_url.clone(),
    })
}
async fn provider(
    data_dir: &Path,
    bootstrap: &HubBootstrap,
    mode: TransportMode,
) -> Result<Arc<Provider>> {
    transport::validate_bootstrap(bootstrap)?;
    if mode == TransportMode::Anonymous {
        return Err("Anonymous peer hosting is unsupported".into());
    }
    let runtime = runtime()?;
    if runtime.state.data_dir() != data_dir {
        return Err("Peer runtime belongs to another profile".into());
    }
    let label = format!("{}:{mode:?}", bootstrap.relay_url);
    let mut providers = runtime.providers.lock().await;
    if let Some(provider) = providers.get(&label) {
        return Ok(provider.clone());
    }
    if providers.len() >= 8 {
        return Err("Too many peer relay endpoints".into());
    }
    let key = persistent_key(data_dir, &label)?;
    let relay: iroh::RelayUrl = bootstrap
        .relay_url
        .parse()
        .map_err(|_| "Invalid peer relay")?;
    let mut builder = Endpoint::empty_builder()
        .secret_key(key.clone())
        .clear_address_lookup()
        .alpns(vec![ALPN.to_vec()])
        .relay_mode(RelayMode::Custom(relay.into()));
    if mode == TransportMode::Relayed {
        builder = builder.clear_ip_transports();
    }
    let endpoint = timeout(IDLE, builder.bind())
        .await
        .map_err(|_| "Peer bind timed out")?
        .map_err(|_| "Unable to bind peer endpoint")?;
    let provider = Arc::new(Provider {
        endpoint: endpoint.clone(),
        key,
    });
    providers.insert(label, provider.clone());
    let weak = Arc::downgrade(&runtime);
    let cancelled = runtime.cancelled.clone();
    let listener_relay = bootstrap.relay_url.clone();
    tokio::spawn(async move {
        loop {
            let incoming = tokio::select! { _ = cancelled.cancelled() => break, incoming = endpoint.accept() => match incoming { Some(incoming) => incoming, None => break } };
            let Some(runtime) = weak.upgrade() else { break };
            let permit = match runtime.permits.clone().try_acquire_owned() {
                Ok(permit) => permit,
                Err(_) => {
                    incoming.refuse();
                    continue;
                }
            };
            let state = runtime.state.clone();
            let listener_relay = listener_relay.clone();
            let endpoint_id = endpoint.id().to_string();
            let cancel = runtime.cancelled.clone();
            drop(runtime);
            tokio::spawn(async move {
                let _permit = permit;
                tokio::select! {
                    _ = cancel.cancelled() => {},
                    _ = timeout(Duration::from_secs(15 * 60), async {
                        if let Ok(Ok(connection)) = timeout(IDLE, incoming).await {
                            let _ = serve(&state, &connection, &endpoint_id, &listener_relay, mode).await;
                            connection.close(0u32.into(), b"peer read complete");
                        }
                    }) => {}
                }
            });
        }
        endpoint.close().await;
    });
    Ok(provider)
}

fn verify(grant: &Grant, authority: &Authority, audience: &str, recipient: &str) -> Result<Claims> {
    if grant.payload.len() > MAX_FRAME || grant.signature.len() > 128 {
        return Err("Oversized peer grant".into());
    }
    let payload = URL_SAFE_NO_PAD
        .decode(&grant.payload)
        .map_err(|_| "Invalid grant payload")?;
    let signature: [u8; 64] = URL_SAFE_NO_PAD
        .decode(&grant.signature)
        .map_err(|_| "Invalid grant signature")?
        .try_into()
        .map_err(|_| "Invalid grant signature size")?;
    let mut signed = READ_PREFIX.to_vec();
    signed.extend_from_slice(&payload);
    let key: EndpointId = authority
        .signing_key
        .parse()
        .map_err(|_| "Invalid grant authority")?;
    key.verify(&signed, &Signature::from_bytes(&signature))
        .map_err(|_| "Untrusted peer grant")?;
    let claims: Claims = serde_json::from_slice(&payload).map_err(|_| "Invalid grant claims")?;
    let now = now()?;
    if claims.v != 1
        || claims.iss != authority.origin
        || claims.instance_id != authority.instance_id
        || claims.realm != authority.realm
        || claims.aud != audience
        || claims.sub != recipient
        || claims.iat > now + 5
        || claims.iat.saturating_add(60) < now
        || claims.exp <= now
        || claims.exp > claims.iat.saturating_add(60)
        || !hash_valid(&claims.sha256)
        || claims.byte_size == 0
        || claims.byte_size > LIMIT
        || !matches!(claims.variant.as_str(), "original" | "thumbnail" | "sample")
        || claims.post_id.is_empty()
        || claims.post_id.len() > 18
        || !claims.post_id.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err("Peer grant scope or lifetime mismatch".into());
    }
    Ok(claims)
}
async fn frame<T: Serialize>(send: &mut iroh::endpoint::SendStream, head: &T) -> Result<()> {
    let bytes = serde_json::to_vec(head).map_err(|_| "Unable to encode peer frame")?;
    if bytes.len() > MAX_FRAME {
        return Err("Peer frame oversized".into());
    }
    timeout(IDLE, async {
        AsyncWriteExt::write_u32(send, bytes.len() as u32).await?;
        AsyncWriteExt::write_all(send, &bytes).await
    })
    .await
    .map_err(|_| "Peer write timed out")?
    .map_err(|_| "Unable to write peer frame")
}
async fn read_frame<T: for<'a> Deserialize<'a>>(
    recv: &mut iroh::endpoint::RecvStream,
) -> Result<T> {
    timeout(IDLE, async {
        let length = AsyncReadExt::read_u32(recv)
            .await
            .map_err(|_| "Missing peer frame")? as usize;
        if length == 0 || length > MAX_FRAME {
            return Err("Invalid peer frame size".into());
        }
        let mut bytes = vec![0; length];
        AsyncReadExt::read_exact(recv, &mut bytes)
            .await
            .map_err(|_| "Incomplete peer frame")?;
        serde_json::from_slice(&bytes).map_err(|_| "Invalid peer frame".into())
    })
    .await
    .map_err(|_| "Peer read timed out")?
}
async fn live_check(authority: &Authority, grant: &Grant) -> Result<()> {
    let response = transport::request(
        &authority.bootstrap,
        authority.mode,
        HubRequest {
            method: "POST".into(),
            path: "/api/client/v1/peer/grants/check".into(),
            headers: vec![
                ("accept".into(), "application/json".into()),
                ("content-type".into(), "application/json".into()),
            ],
            body: serde_json::to_vec(grant).map_err(|_| "Invalid peer grant")?,
        },
    )
    .await?;
    if response.status != 200 {
        return Err("Hub no longer permits community delivery".into());
    }
    let value: serde_json::Value = serde_json::from_slice(&response.into_bytes(MAX_FRAME).await?)
        .map_err(|_| "Invalid hub grant check")?;
    if value["valid"] != true {
        return Err("Hub denied community delivery".into());
    }
    Ok(())
}
async fn serve(
    state: &AppState,
    connection: &iroh::endpoint::Connection,
    audience: &str,
    listener_relay: &str,
    listener_mode: TransportMode,
) -> Result<()> {
    serve_with_check(
        state,
        connection,
        audience,
        listener_relay,
        listener_mode,
        |authority, grant| Box::pin(live_check(authority, grant)),
    )
    .await
}

// Private seam keeps the real file/policy/protocol path exercised in ephemeral
// tests. Production always supplies the pinned hub's no-bearer live check.
async fn serve_with_check<F>(
    state: &AppState,
    connection: &iroh::endpoint::Connection,
    audience: &str,
    listener_relay: &str,
    listener_mode: TransportMode,
    check: F,
) -> Result<()>
where
    F: for<'a> Fn(&'a Authority, &'a Grant) -> futures_util::future::BoxFuture<'a, Result<()>>,
{
    let (mut send, mut recv) = timeout(IDLE, connection.accept_bi())
        .await
        .map_err(|_| "Peer stream timed out")?
        .map_err(|_| "Unable to accept peer stream")?;
    let request: RequestHead = read_frame(&mut recv).await?;
    if request.version != 1 {
        return Err("Unsupported peer protocol".into());
    }
    let mut extra = [0];
    if timeout(IDLE, AsyncReadExt::read(&mut recv, &mut extra))
        .await
        .map_err(|_| "Peer request timed out")?
        .map_err(|_| "Peer request failed")?
        != 0
    {
        return Err("Peer request must not contain a body".into());
    }
    let recipient = connection.remote_id().to_string();
    // Find an explicitly enrolled authority that actually validates the signature.
    let mut selected = None;
    for authority in authorities(state.data_dir())? {
        if authority.mode != listener_mode || authority.bootstrap.relay_url != listener_relay {
            continue;
        }
        if let Ok(claims) = verify(&request.grant, &authority, audience, &recipient) {
            selected = Some((authority, claims));
            break;
        }
    }
    let (authority, claims) = selected.ok_or("No enrolled authority permits this peer read")?;
    let selected_config = crate::routes::online::load_hub(state.data_dir(), &authority.source_id)
        .map_err(|e| e.to_string())?
        .ok_or("Peer source is no longer enrolled")?;
    if selected_config.transport_mode != listener_mode
        || selected_config.bootstrap.relay_url != listener_relay
        || selected_config.bootstrap.peer_id != authority.bootstrap.peer_id
        || selected_config.instance_id != authority.instance_id
    {
        return Err("Source transport changed; old peer endpoint is no longer permitted".into());
    }
    let initial_family_lock = state.is_family_mode_locked();
    let record = super::offline::retained_variant(
        state.data_dir(),
        &authority.source_id,
        &claims.post_id,
        &claims.variant,
        initial_family_lock,
    )?
    .ok_or("This rendition is not opted in for community hosting")?;
    if !record.authoritative_digest
        || record.sha256 != claims.sha256
        || record.bytes != claims.byte_size
        || record.revision != claims.revision
        || record.canonical_origin != authority.origin
        || record.instance_id != authority.instance_id
    {
        return Err("Retained rendition differs from grant".into());
    }
    let mut options = std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    let file = options
        .open(&record.path)
        .map_err(|_| "Retained rendition unavailable")?;
    if !file
        .metadata()
        .map_err(|_| "Unable to inspect retained media")?
        .is_file()
    {
        return Err("Retained media is not a file".into());
    }
    let mut file = tokio::fs::File::from_std(file);
    let mut digest = Sha256::new();
    let mut size = 0u64;
    let mut chunk = vec![0u8; 64 * 1024];
    loop {
        let read = timeout(IDLE, file.read(&mut chunk))
            .await
            .map_err(|_| "Retained media stalled")?
            .map_err(|_| "Retained media read failed")?;
        if read == 0 {
            break;
        }
        size += read as u64;
        if size > claims.byte_size {
            return Err("Retained rendition size changed".into());
        }
        digest.update(&chunk[..read]);
    }
    if size != claims.byte_size || format!("{:x}", digest.finalize()) != claims.sha256 {
        return Err("Retained rendition checksum changed".into());
    }
    // Recheck after disk verification, immediately before delivering any bytes.
    check(&authority, &request.grant).await?;
    verify(&request.grant, &authority, audience, &recipient)?;
    let current = super::offline::retained_variant(
        state.data_dir(),
        &authority.source_id,
        &claims.post_id,
        &claims.variant,
        state.is_family_mode_locked(),
    )?
    .ok_or("Hosting was withdrawn")?;
    if state.is_family_mode_locked() != initial_family_lock
        || current.sha256 != claims.sha256
        || current.bytes != claims.byte_size
        || current.revision != claims.revision
    {
        return Err("Hosting permission changed before delivery".into());
    }
    file.seek(std::io::SeekFrom::Start(0))
        .await
        .map_err(|_| "Unable to rewind rendition")?;
    frame(
        &mut send,
        &ResponseHead {
            version: 1,
            status: 200,
            byte_size: claims.byte_size,
            sha256: claims.sha256.clone(),
            mime_type: record.mime_type,
        },
    )
    .await?;
    loop {
        let current = super::offline::retained_variant(
            state.data_dir(),
            &authority.source_id,
            &claims.post_id,
            &claims.variant,
            state.is_family_mode_locked(),
        )?
        .ok_or("Hosting withdrawn during delivery")?;
        if current.sha256 != claims.sha256
            || current.bytes != claims.byte_size
            || current.revision != claims.revision
        {
            return Err("Retained permission changed during delivery".into());
        }
        if state.is_family_mode_locked() != initial_family_lock {
            return Err("Family policy changed during community delivery".into());
        }
        let read = timeout(IDLE, file.read(&mut chunk))
            .await
            .map_err(|_| "Retained media stalled")?
            .map_err(|_| "Retained media read failed")?;
        if read == 0 {
            break;
        }
        timeout(IDLE, send.write_all(&chunk[..read]))
            .await
            .map_err(|_| "Peer write stalled")?
            .map_err(|_| "Peer write failed")?;
    }
    send.finish()
        .map_err(|_| "Unable to finish peer response")?;
    let _ = timeout(IDLE, send.stopped()).await;
    Ok(())
}

/// Sign a one-use hub challenge with the actual selected endpoint's identity.
/// The envelope carries no bearer credentials and is bounded to 100 renditions.
pub async fn sign_report(
    data_dir: &Path,
    bootstrap: &HubBootstrap,
    mode: TransportMode,
    challenge: &serde_json::Value,
    items: Vec<serde_json::Value>,
) -> Result<Grant> {
    let provider = provider(data_dir, bootstrap, mode).await?;
    let timestamp = now()?;
    let origin = challenge["origin"]
        .as_str()
        .ok_or("Missing challenge origin")?;
    let realm = challenge["realm"]
        .as_str()
        .ok_or("Missing challenge realm")?;
    if items.len() > 100
        || challenge["relay_url"] != bootstrap.relay_url
        || challenge["expires_at"].as_u64().unwrap_or(0) <= timestamp
        || !authorities(data_dir)?.iter().any(|a| {
            a.origin == origin
                && a.realm == realm
                && a.bootstrap.peer_id == bootstrap.peer_id
                && a.bootstrap.relay_url == bootstrap.relay_url
                && a.mode == mode
        })
    {
        return Err("Provider challenge differs from enrolled hub".into());
    }
    for item in &items {
        if !hash_valid(item["sha256"].as_str().unwrap_or(""))
            || !matches!(
                item["variant"].as_str(),
                Some("original" | "thumbnail" | "sample")
            )
            || !matches!(item["byte_size"].as_u64(), Some(1..=LIMIT))
        {
            return Err("Invalid provider manifest".into());
        }
    }
    let payload = serde_json::to_vec(&serde_json::json!({"challenge_id":challenge["id"], "nonce":challenge["nonce"], "origin":origin, "realm":realm, "iat":timestamp, "exp":timestamp+120, "items":items})).map_err(|_| "Unable to encode provider proof")?;
    if payload.len() > MAX_FRAME {
        return Err("Provider proof exceeds protocol limit".into());
    }
    let mut signed = REPORT_PREFIX.to_vec();
    signed.extend_from_slice(&payload);
    Ok(Grant {
        payload: URL_SAFE_NO_PAD.encode(&payload),
        signature: URL_SAFE_NO_PAD.encode(provider.key.sign(&signed).to_bytes()),
    })
}

/// Verified peer bytes land atomically only after exact size and digest checks.
/// Destination is a new staging path owned by the offline reservation, never an
/// arbitrary overwrite. The same pinned hub and selected mode scope both sides.
#[allow(clippy::too_many_arguments)]
pub async fn download(
    data_dir: &Path,
    source_id: &str,
    bootstrap: &HubBootstrap,
    mode: TransportMode,
    provider_id: &str,
    relay_url: &str,
    grant: &Grant,
    destination: &Path,
    expected_sha: &str,
    expected_size: u64,
    expected_revision: &str,
) -> Result<()> {
    if relay_url != bootstrap.relay_url || mode == TransportMode::Anonymous {
        return Err("Peer relay or transport differs from selected source".into());
    }
    let provider = provider(data_dir, bootstrap, mode).await?;
    let audience: EndpointId = provider_id
        .parse()
        .map_err(|_| "Invalid provider identity")?;
    let authority = authorities(data_dir)?
        .into_iter()
        .find(|a| {
            a.source_id == source_id
                && a.bootstrap.peer_id == bootstrap.peer_id
                && a.bootstrap.relay_url == bootstrap.relay_url
                && a.mode == mode
        })
        .ok_or("Peer authority is not enrolled")?;
    let claims = verify(
        grant,
        &authority,
        provider_id,
        &provider.endpoint.id().to_string(),
    )?;
    if claims.sha256 != expected_sha
        || claims.byte_size != expected_size
        || claims.revision != expected_revision
    {
        return Err("Peer grant differs from requested canonical rendition".into());
    }
    let relay: iroh::RelayUrl = relay_url.parse().map_err(|_| "Invalid peer relay")?;
    let runtime = runtime()?;
    let permit = timeout(IDLE, runtime.permits.clone().acquire_owned())
        .await
        .map_err(|_| "Peer queue timed out")?
        .map_err(|_| "Peer runtime stopped")?;
    let cancel = runtime.cancelled.clone();
    drop(runtime);
    let connection = tokio::select! {
        _ = cancel.cancelled() => return Err("Peer runtime stopped".into()),
        connection = timeout(IDLE, provider.endpoint.connect(EndpointAddr::new(audience).with_relay_url(relay), ALPN)) => connection.map_err(|_| "Peer connection timed out")?.map_err(|_| "Peer provider unavailable")?
    };
    let result = tokio::select! {
        _ = cancel.cancelled() => Err("Peer runtime stopped".into()),
        result = timeout(Duration::from_secs(15 * 60), receive(&connection, grant, destination, expected_sha, expected_size)) => result.map_err(|_| "Peer transfer timed out")?
    };
    connection.close(0u32.into(), b"peer read complete");
    drop(permit);
    result
}

async fn receive(
    connection: &iroh::endpoint::Connection,
    grant: &Grant,
    destination: &Path,
    expected_sha: &str,
    expected_size: u64,
) -> Result<()> {
    if !hash_valid(expected_sha) || expected_size == 0 || expected_size > LIMIT {
        return Err("Invalid canonical peer rendition".into());
    }
    let (mut send, mut recv) = timeout(IDLE, connection.open_bi())
        .await
        .map_err(|_| "Peer stream timed out")?
        .map_err(|_| "Peer stream unavailable")?;
    frame(
        &mut send,
        &RequestHead {
            version: 1,
            grant: grant.clone(),
        },
    )
    .await?;
    send.finish().map_err(|_| "Unable to finish peer request")?;
    let head: ResponseHead = read_frame(&mut recv).await?;
    if head.version != 1
        || head.status != 200
        || head.byte_size != expected_size
        || head.sha256 != expected_sha
    {
        return Err("Peer response differs from canonical rendition".into());
    }
    let staged =
        tempfile::NamedTempFile::new_in(destination.parent().ok_or("Missing peer staging parent")?)
            .map_err(|_| "Unable to stage peer rendition")?;
    let mut output = tokio::fs::File::from_std(
        staged
            .reopen()
            .map_err(|_| "Unable to open peer staging file")?,
    );
    let mut digest = Sha256::new();
    let mut length = 0u64;
    let mut chunk = vec![0u8; 64 * 1024];
    loop {
        let read = timeout(IDLE, AsyncReadExt::read(&mut recv, &mut chunk))
            .await
            .map_err(|_| "Peer response stalled")?
            .map_err(|_| "Peer response failed")?;
        if read == 0 {
            break;
        }
        length = length
            .checked_add(read as u64)
            .ok_or("Peer response size overflow")?;
        if length > expected_size {
            return Err("Peer response exceeded canonical size".into());
        }
        digest.update(&chunk[..read]);
        timeout(IDLE, output.write_all(&chunk[..read]))
            .await
            .map_err(|_| "Peer staging stalled")?
            .map_err(|_| "Peer staging failed")?;
    }
    if length != expected_size || format!("{:x}", digest.finalize()) != expected_sha {
        return Err("Peer rendition failed checksum verification".into());
    }
    output
        .sync_all()
        .await
        .map_err(|_| "Unable to flush peer rendition")?;
    drop(output);
    staged
        .persist_noclobber(destination)
        .map_err(|_| "Unable to commit verified peer rendition")?;
    Ok(())
}

/// Heartbeat all ready retained renditions in bounded proofs, independently of
/// downloads. Each proof refreshes its chunk for 120 seconds; empty withdraws.
pub async fn publish_retained(state: &AppState, source_id: &str) -> Result<()> {
    use crate::routes::online::{load_hub, load_source};
    let source = load_source(state.data_dir(), source_id).map_err(|e| e.to_string())?;
    let config = load_hub(state.data_dir(), source_id)
        .map_err(|e| e.to_string())?
        .ok_or("Community hosting requires an enrolled hub")?;
    enroll_authority(state.data_dir(), source_id, &source, &config)?;
    let provider =
        ensure_provider(state.data_dir(), &config.bootstrap, config.transport_mode).await?;
    let mut cursor = (String::new(), String::new());
    let mut done = false;
    let mut advertised = false;
    while !done {
        // Bound memory and pending challenges. Four proofs can cover a large
        // collection within the report TTL without exceeding the hub's cap of
        // eight one-use pending challenges per account.
        let mut batches = Vec::new();
        for _ in 0..4 {
            let data_dir = state.data_dir().to_path_buf();
            let source_id = source_id.to_string();
            let after = cursor.clone();
            let family_locked = state.is_family_mode_locked();
            let (records, next) = tokio::task::spawn_blocking(move || {
                super::offline::hosted_manifest_page(
                    &data_dir,
                    &source_id,
                    family_locked,
                    &after.0,
                    &after.1,
                )
            })
            .await
            .map_err(|_| "Peer manifest worker failed")??;
            let items: Vec<serde_json::Value> = records.into_iter().map(|r| serde_json::json!({"remote_post_id":r.remote_post_id,"variant":r.kind,"sha256":r.sha256,"byte_size":r.bytes})).collect();
            if !items.is_empty() {
                advertised = true;
                batches.push(items);
            }
            match next {
                Some(next) => cursor = next,
                None => {
                    done = true;
                    break;
                }
            }
        }
        use futures_util::StreamExt;
        let mut proofs = futures_util::stream::iter(
            batches
                .into_iter()
                .map(|items| publish_chunk(state, &source, &config, &provider, items)),
        )
        .buffer_unordered(4);
        while let Some(result) = proofs.next().await {
            result?;
        }
    }
    if !advertised {
        publish_chunk(state, &source, &config, &provider, vec![]).await?;
    }
    Ok(())
}

async fn publish_chunk(
    state: &AppState,
    source: &super::models::RemoteSource,
    config: &crate::routes::online::HubConfig,
    provider: &ProviderIdentity,
    items: Vec<serde_json::Value>,
) -> Result<()> {
    use crate::routes::online::{hub_request, hub_response_json};
    let response = hub_request(
        state.data_dir(),
        source,
        config,
        "POST",
        "/api/client/v1/peer/challenge",
        vec![("content-type".into(), "application/json".into())],
        serde_json::to_vec(&serde_json::json!({"endpoint_id":provider.endpoint_id}))
            .map_err(|_| "Unable to encode peer challenge")?,
    )
    .await
    .map_err(|e| e.to_string())?;
    let challenge = hub_response_json(response)
        .await
        .map_err(|e| e.to_string())?;
    let envelope = sign_report(
        state.data_dir(),
        &config.bootstrap,
        config.transport_mode,
        &challenge,
        items,
    )
    .await?;
    let response = hub_request(
        state.data_dir(),
        source,
        config,
        "POST",
        "/api/client/v1/peer/report",
        vec![("content-type".into(), "application/json".into())],
        serde_json::to_vec(&envelope).map_err(|_| "Unable to encode peer report")?,
    )
    .await
    .map_err(|e| e.to_string())?;
    hub_response_json(response)
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

/// Derive trust only from the already validated, enrolled pinned-hub descriptor.
pub(crate) fn enroll_authority(
    data_dir: &Path,
    source_id: &str,
    source: &super::models::RemoteSource,
    config: &crate::routes::online::HubConfig,
) -> Result<()> {
    let descriptor = &config.descriptor;
    let instance = super::hub::validate_descriptor(&source.normalized_base_url, descriptor)
        .map_err(|e| e.to_string())?;
    if instance != config.instance_id {
        return Err("Stored peer authority instance differs from source".into());
    }
    let signing_key = descriptor
        .pointer("/peer_read/signing_key")
        .and_then(serde_json::Value::as_str)
        .ok_or("Hub does not offer secured community hosting")?;
    if descriptor.pointer("/peer_read/requires_live_grant_check")
        != Some(&serde_json::Value::Bool(true))
    {
        return Err("Hub does not offer fresh peer grant checks".into());
    }
    remember_authority(
        data_dir,
        source_id,
        &source.normalized_base_url,
        &config.instance_id,
        descriptor
            .pointer("/instance/realm")
            .and_then(serde_json::Value::as_str)
            .ok_or("Hub realm missing")?,
        signing_key,
        &config.bootstrap,
        config.transport_mode,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn authority(key: &SecretKey) -> Authority {
        Authority {
            source_id: "source".into(),
            origin: "https://hub.example".into(),
            instance_id: "instance".into(),
            realm: "main".into(),
            signing_key: key.public().to_string(),
            bootstrap: HubBootstrap {
                peer_id: key.public().to_string(),
                relay_url: "https://relay.example".into(),
            },
            mode: TransportMode::Standard,
        }
    }
    fn signed(key: &SecretKey, claims: &serde_json::Value) -> Grant {
        let payload = serde_json::to_vec(claims).unwrap();
        let mut bytes = READ_PREFIX.to_vec();
        bytes.extend_from_slice(&payload);
        Grant {
            payload: URL_SAFE_NO_PAD.encode(payload),
            signature: URL_SAFE_NO_PAD.encode(key.sign(&bytes).to_bytes()),
        }
    }
    fn claims(audience: &str, recipient: &str, digest: &str, size: u64) -> serde_json::Value {
        let timestamp = now().unwrap();
        json!({"v":1,"iss":"https://hub.example","instance_id":"instance","realm":"main","aud":audience,"sub":recipient,"post_id":"12","variant":"original","sha256":digest,"byte_size":size,"revision":"revision-1","iat":timestamp,"exp":timestamp+60})
    }
    // AC: @creator-community-replication ac-5
    #[test]
    fn grants_require_pinned_origin_realm_instance_recipient_digest_and_lifetime() {
        let key = SecretKey::from_bytes(&[31; 32]);
        let trusted = authority(&key);
        let base = claims("provider", "recipient", &"ab".repeat(32), 7);
        assert!(verify(&signed(&key, &base), &trusted, "provider", "recipient").is_ok());
        for (field, value) in [
            ("iss", json!("https://other.example")),
            ("instance_id", json!("other")),
            ("realm", json!("adult")),
            ("sub", json!("other")),
            ("aud", json!("other")),
            ("sha256", json!("legacy-md5")),
            ("byte_size", json!(LIMIT + 1)),
            ("exp", json!(now().unwrap() - 1)),
            ("iat", json!(u64::MAX)),
        ] {
            let mut altered = base.clone();
            altered[field] = value;
            assert!(
                verify(&signed(&key, &altered), &trusted, "provider", "recipient").is_err(),
                "{field}"
            );
        }
        let other = SecretKey::from_bytes(&[32; 32]);
        assert!(verify(&signed(&other, &base), &trusted, "provider", "recipient").is_err());
    }
    #[test]
    fn identity_is_private_persistent_and_distinct_by_relay_and_mode() {
        let directory = tempfile::tempdir().unwrap();
        let standard = persistent_key(directory.path(), "https://relay.example:Standard").unwrap();
        assert_eq!(
            standard.public(),
            persistent_key(directory.path(), "https://relay.example:Standard")
                .unwrap()
                .public()
        );
        assert_ne!(
            standard.public(),
            persistent_key(directory.path(), "https://relay.example:Relayed")
                .unwrap()
                .public()
        );
        assert_ne!(
            standard.public(),
            persistent_key(directory.path(), "https://other.example:Standard")
                .unwrap()
                .public()
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            for file in std::fs::read_dir(directory.path()).unwrap() {
                assert_eq!(
                    file.unwrap().metadata().unwrap().permissions().mode() & 0o777,
                    0o600
                );
            }
        }
    }
    async fn ephemeral() -> Endpoint {
        Endpoint::empty_builder()
            .clear_address_lookup()
            .relay_mode(RelayMode::Disabled)
            .alpns(vec![ALPN.to_vec()])
            .bind_addr("127.0.0.1:0")
            .unwrap()
            .bind()
            .await
            .unwrap()
    }
    // Real authenticated ephemeral Iroh streams exercise the production receiver,
    // including exact body length/hash and authenticated recipient grant binding.
    async fn transfer(body: &[u8], extra: bool, shortened: bool, digest_wrong: bool) {
        let server = ephemeral().await;
        let client = ephemeral().await;
        let hub = SecretKey::from_bytes(&[30; 32]);
        let trusted = authority(&hub);
        let digest = format!("{:x}", Sha256::digest(body));
        let grant = signed(
            &hub,
            &claims(
                &server.id().to_string(),
                &client.id().to_string(),
                &digest,
                body.len() as u64,
            ),
        );
        let owned_body = body.to_vec();
        let server_clone = server.clone();
        let digest_clone = digest.clone();
        let task = tokio::spawn(async move {
            let incoming = timeout(Duration::from_secs(5), server_clone.accept())
                .await
                .unwrap()
                .unwrap();
            let connection = incoming.await.unwrap();
            let (mut send, mut recv) = connection.accept_bi().await.unwrap();
            let request: RequestHead = read_frame(&mut recv).await.unwrap();
            let verified = verify(
                &request.grant,
                &trusted,
                &server_clone.id().to_string(),
                &connection.remote_id().to_string(),
            )
            .unwrap();
            assert_eq!(verified.byte_size, owned_body.len() as u64);
            let mut trailing = [0];
            assert_eq!(
                AsyncReadExt::read(&mut recv, &mut trailing).await.unwrap(),
                0
            );
            frame(
                &mut send,
                &ResponseHead {
                    version: 1,
                    status: 200,
                    byte_size: owned_body.len() as u64,
                    sha256: digest_clone,
                    mime_type: None,
                },
            )
            .await
            .unwrap();
            let mut response = owned_body;
            if extra {
                response.push(0);
            }
            if shortened {
                response.pop();
            }
            if digest_wrong {
                response[0] ^= 1;
            }
            send.write_all(&response).await.unwrap();
            send.finish().unwrap();
            let _ = timeout(Duration::from_secs(5), send.stopped()).await;
        });
        let connection = timeout(Duration::from_secs(5), client.connect(server.addr(), ALPN))
            .await
            .unwrap()
            .unwrap();
        let directory = tempfile::tempdir().unwrap();
        let destination = directory.path().join("verified.bin");
        let result = receive(
            &connection,
            &grant,
            &destination,
            &digest,
            body.len() as u64,
        )
        .await;
        if extra || shortened || digest_wrong {
            assert!(result.is_err());
            assert!(!destination.exists());
            assert_eq!(
                std::fs::read_dir(directory.path()).unwrap().count(),
                0,
                "failed staging files must be removed"
            );
        } else {
            result.unwrap();
            assert_eq!(std::fs::read(&destination).unwrap(), body);
        }
        connection.close(0u32.into(), b"test complete");
        task.await.unwrap();
        client.close().await;
        server.close().await;
    }
    // AC: @creator-community-replication ac-5
    #[tokio::test]
    async fn ephemeral_iroh_peer_receive_verifies_signed_scope_and_bytes() {
        transfer(&vec![42; 200_000], false, false, false).await;
    }
    #[tokio::test]
    async fn ephemeral_iroh_peer_rejects_excess_truncation_and_corruption() {
        transfer(b"canonical bytes", true, false, false).await;
        transfer(b"canonical bytes", false, true, false).await;
        transfer(b"canonical bytes", false, false, true).await;
    }

    async fn actual_serve_case(case: &str) {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let directory = tempfile::tempdir().unwrap();
        let state = AppState::new(directory.path(), 0).unwrap();
        crate::routes::online::initialize_offline_database(directory.path()).unwrap();
        let hub = SecretKey::from_bytes(&[29; 32]);
        let authority = authority(&hub);
        remember_authority(
            directory.path(),
            "source",
            &authority.origin,
            &authority.instance_id,
            &authority.realm,
            &authority.signing_key,
            &authority.bootstrap,
            authority.mode,
        )
        .unwrap();
        let canonical = vec![79u8; 180_000];
        let digest = format!("{:x}", Sha256::digest(&canonical));
        let item = super::super::hub::catalog_item("source", "https://hub.example", &json!({
            "remote_post_id":"12","canonical_url":"https://hub.example/image/12","remote_revision":"revision-1",
            "caption":"fixture","tags":[],"rating":"pg","creator":{"id":"7","name":"Creator"},
            "variants":[{"kind":"original","asset_url":"https://hub.example/api/client/v1/media/12/original","mime_type":"image/png","sha256":digest,"byte_size":canonical.len()}]
        })).unwrap();
        let conn = super::super::offline::open(directory.path()).unwrap();
        conn.execute("INSERT INTO remote_sources(source_id,provider_family,display_name,normalized_base_url,capabilities_json,created_at) VALUES('source','native-hub','Hub','https://hub.example','{}','now')", []).unwrap();
        conn.execute("INSERT INTO remote_hub_transport VALUES('source','instance',?1,'https://relay.example','standard','{}')", rusqlite::params![hub.public().to_string()]).unwrap();
        conn.execute("INSERT INTO offline_sources(source_id,enabled,sync_all_creators,peer_hosting_enabled,state) VALUES('source',1,1,1,'ready')", []).unwrap();
        conn.execute("INSERT INTO offline_items(source_id,post_id,creator_id,media_type,revision,item_json,metadata_json,last_verified_at) VALUES('source','12','7','image','revision-1',?1,'{}',0)", rusqlite::params![serde_json::to_string(&item).unwrap()]).unwrap();
        let media_root = directory.path().join("online-offline");
        std::fs::create_dir_all(&media_root).unwrap();
        let media_path = media_root.join("retained.png");
        std::fs::write(&media_path, &canonical).unwrap();
        conn.execute("INSERT INTO offline_variants(source_id,post_id,revision,kind,sha256,authoritative_digest,bytes,relative_path,state,mime_type) VALUES('source','12','revision-1','original',?1,1,?2,'retained.png','ready','image/png')", rusqlite::params![digest, canonical.len() as i64]).unwrap();
        match case {
            "corrupt" => std::fs::write(&media_path, vec![80; canonical.len()]).unwrap(),
            "disabled" => {
                conn.execute("UPDATE offline_sources SET peer_hosting_enabled=0", [])
                    .unwrap();
            }
            "withdrawn" => {
                conn.execute("UPDATE offline_items SET withdrawn=1", [])
                    .unwrap();
            }
            "mode_changed" => {
                conn.execute(
                    "UPDATE remote_hub_transport SET transport_mode='relayed'",
                    [],
                )
                .unwrap();
            }
            _ => {}
        }
        drop(conn);
        let server = ephemeral().await;
        let client = ephemeral().await;
        let mut scoped = claims(
            &server.id().to_string(),
            &client.id().to_string(),
            &digest,
            canonical.len() as u64,
        );
        if case == "wrong_recipient" {
            scoped["sub"] = json!(SecretKey::from_bytes(&[28; 32]).public().to_string());
        }
        if case == "wrong_realm" {
            scoped["realm"] = json!("adult");
        }
        let grant = signed(&hub, &scoped);
        let checked = Arc::new(AtomicUsize::new(0));
        let checks = checked.clone();
        let server_endpoint = server.clone();
        let denied = case == "hub_denied";
        let task = tokio::spawn(async move {
            let incoming = timeout(Duration::from_secs(5), server_endpoint.accept())
                .await
                .unwrap()
                .unwrap();
            let connection = incoming.await.unwrap();
            let result = serve_with_check(
                &state,
                &connection,
                &server_endpoint.id().to_string(),
                "https://relay.example",
                TransportMode::Standard,
                move |_authority, _grant| {
                    let checks = checks.clone();
                    Box::pin(async move {
                        checks.fetch_add(1, Ordering::SeqCst);
                        if denied {
                            Err("Pinned hub withdrew grant".into())
                        } else {
                            Ok(())
                        }
                    })
                },
            )
            .await;
            connection.close(0u32.into(), b"serve test complete");
            result
        });
        let connection = timeout(Duration::from_secs(5), client.connect(server.addr(), ALPN))
            .await
            .unwrap()
            .unwrap();
        let destination = directory.path().join("received.png");
        let received = receive(
            &connection,
            &grant,
            &destination,
            &digest,
            canonical.len() as u64,
        )
        .await;
        connection.close(0u32.into(), b"receive test complete");
        let served = task.await.unwrap();
        if case == "valid" {
            received.unwrap();
            served.unwrap();
            assert_eq!(std::fs::read(&destination).unwrap(), canonical);
            assert_eq!(checked.load(Ordering::SeqCst), 1);
        } else {
            assert!(received.is_err(), "{case}");
            assert!(served.is_err(), "{case}");
            assert!(!destination.exists(), "{case}");
            assert_eq!(checked.load(Ordering::SeqCst), usize::from(denied), "scope, retained permission and disk digest must validate before the fresh hub check: {case}");
        }
        client.close().await;
        server.close().await;
    }

    // AC: @creator-community-replication ac-5
    #[tokio::test]
    async fn ephemeral_iroh_actual_serve_checks_retention_hash_and_fresh_authorization() {
        actual_serve_case("valid").await;
        actual_serve_case("corrupt").await;
        actual_serve_case("disabled").await;
        actual_serve_case("withdrawn").await;
        actual_serve_case("wrong_recipient").await;
        actual_serve_case("wrong_realm").await;
        actual_serve_case("mode_changed").await;
        actual_serve_case("hub_denied").await;
    }
}
