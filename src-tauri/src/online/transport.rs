// SPDX-License-Identifier: GPL-3.0-only
//! Pinned-peer HTTP-style hub requests over encrypted Iroh QUIC.
//! The protocol is specified in booru-node's docs/architecture/dmc-hub-relay.md.
use futures_core::Stream;
use futures_util::StreamExt;
use iroh::{Endpoint, EndpointAddr, EndpointId, RelayMode, RelayUrl};
use serde::{Deserialize, Serialize};
use std::{
    pin::Pin,
    sync::{Arc, OnceLock},
    time::Duration,
};
use tokio::{
    io::{AsyncReadExt, AsyncWriteExt},
    sync::Semaphore,
    time::{timeout, Instant},
};

pub const ALPN: &[u8] = b"donutmediacenter/hub-http/1";
const MAX_HEAD: usize = 32 * 1024;
const MAX_REQUEST: usize = 64 * 1024 * 1024;
const MAX_RESPONSE: u64 = 2 * 1024 * 1024 * 1024;
const IDLE: Duration = Duration::from_secs(30);
const RESPONSE_HEADERS: &[&str] = &[
    "content-type",
    "content-length",
    "content-range",
    "accept-ranges",
    "etag",
    "last-modified",
    "cache-control",
    "vary",
    "content-disposition",
    "retry-after",
    "www-authenticate",
];
static REQUESTS: OnceLock<Arc<Semaphore>> = OnceLock::new();

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HubBootstrap {
    pub peer_id: String,
    pub relay_url: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum TransportMode {
    Standard,
    Relayed,
    Anonymous,
}

pub struct HubRequest {
    pub method: String,
    /// Origin-relative API path, optionally followed by a query string.
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

pub struct HubResponse {
    pub status: u16,
    pub headers: Vec<(String, String)>,
    pub body: Pin<Box<dyn Stream<Item = Result<Vec<u8>, String>> + Send>>,
}
impl HubResponse {
    pub async fn into_bytes(mut self, max_bytes: usize) -> Result<Vec<u8>, String> {
        let mut bytes = Vec::new();
        while let Some(chunk) = self.body.next().await {
            let chunk = chunk?;
            if chunk.len() > max_bytes.saturating_sub(bytes.len()) {
                return Err("Hub response exceeds the requested limit".into());
            }
            bytes.extend_from_slice(&chunk);
        }
        Ok(bytes)
    }
}

#[derive(Serialize)]
struct RequestHead {
    version: u8,
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body_length: u64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResponseHead {
    version: u8,
    status: u16,
    headers: Vec<(String, String)>,
}

/// Validate locally without creating an endpoint, resolving names, or dialing.
pub fn validate_bootstrap(bootstrap: &HubBootstrap) -> Result<(), String> {
    parse_bootstrap(bootstrap).map(|_| ())
}
fn parse_bootstrap(bootstrap: &HubBootstrap) -> Result<(EndpointId, RelayUrl), String> {
    let peer = bootstrap
        .peer_id
        .parse()
        .map_err(|_| "Invalid Iroh peer ID")?;
    let url = reqwest::Url::parse(&bootstrap.relay_url).map_err(|_| "Invalid relay URL")?;
    if url.scheme() != "https"
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.path() != "/"
        || url.query().is_some()
        || url.fragment().is_some()
    {
        return Err("Relay must be a credential-free HTTPS origin".into());
    }
    let relay = bootstrap
        .relay_url
        .parse()
        .map_err(|_| "Invalid Iroh relay URL")?;
    Ok((peer, relay))
}

struct Session {
    endpoint: Endpoint,
    connection: iroh::endpoint::Connection,
}
impl Drop for Session {
    fn drop(&mut self) {
        self.connection.close(0u32.into(), b"request complete");
        // Also clean up the relay endpoint when a body is dropped before EOF.
        if !self.endpoint.is_closed() {
            if let Ok(runtime) = tokio::runtime::Handle::try_current() {
                let endpoint = self.endpoint.clone();
                runtime.spawn(async move {
                    let _ = timeout(IDLE, endpoint.close()).await;
                });
            }
        }
    }
}

pub async fn request(
    bootstrap: &HubBootstrap,
    mode: TransportMode,
    request: HubRequest,
) -> Result<HubResponse, String> {
    // This must precede every network operation, including endpoint binding.
    if mode == TransportMode::Anonymous {
        return Err("Anonymous hub transport is unsupported".into());
    }
    let (peer, relay) = parse_bootstrap(bootstrap)?;
    if request.body.len() > MAX_REQUEST {
        return Err("Hub request exceeds 64 MiB".into());
    }
    if !request.path.starts_with("/api/client/v1/")
        || request.path.contains('#')
        || request.path.len() > 8192
    {
        return Err("Hub request must use an origin-relative client API path".into());
    }
    let is_head = request.method == "HEAD";
    let head = serde_json::to_vec(&RequestHead {
        version: 1,
        method: request.method,
        path: request.path,
        headers: request.headers,
        body_length: request.body.len() as u64,
    })
    .map_err(|_| "Unable to encode hub request")?;
    if head.len() > MAX_HEAD {
        return Err("Hub request headers exceed 32 KiB".into());
    }
    // Thumbnail grids routinely issue more requests than the active stream limit.
    // Queue them fairly while bounding how long a stalled stream can block work.
    let permit = timeout(
        IDLE,
        REQUESTS
            .get_or_init(|| Arc::new(Semaphore::new(8)))
            .clone()
            .acquire_owned(),
    )
    .await
    .map_err(|_| "Hub request queue timed out")?
    .map_err(|_| "Hub request queue closed")?;
    // Empty builder has no lookup services. Standard retains IP paths and can
    // upgrade its initial relay connection; Relayed removes IP transport entirely.
    let mut builder = Endpoint::empty_builder()
        .clear_address_lookup()
        .relay_mode(RelayMode::Custom(relay.clone().into()));
    if mode == TransportMode::Relayed {
        builder = builder.clear_ip_transports();
    }
    let endpoint = timeout(IDLE, builder.bind())
        .await
        .map_err(|_| "Iroh bind timed out")?
        .map_err(|_| "Unable to initialize Iroh endpoint")?;
    let result = timeout(Duration::from_secs(120), async {
        let connection = endpoint.connect(EndpointAddr::new(peer).with_relay_url(relay), ALPN)
            .await.map_err(|_| "Unable to connect to pinned hub peer")?;
        let session = Session { endpoint: endpoint.clone(), connection };
        let (mut send, mut recv) = session.connection.open_bi().await.map_err(|_| "Unable to open hub stream")?;
        AsyncWriteExt::write_u32(&mut send, head.len() as u32).await.map_err(|_| "Unable to write hub frame")?;
        send.write_all(&head).await.map_err(|_| "Unable to write hub headers")?;
        send.write_all(&request.body).await.map_err(|_| "Unable to write hub body")?;
        send.finish().map_err(|_| "Unable to finish hub request")?;
        let len = AsyncReadExt::read_u32(&mut recv).await.map_err(|_| "Unable to read hub response frame")? as usize;
        if len == 0 || len > MAX_HEAD { return Err("Invalid hub response frame length".to_string()); }
        let mut bytes = vec![0; len];
        AsyncReadExt::read_exact(&mut recv, &mut bytes).await.map_err(|_| "Incomplete hub response headers")?;
        let response: ResponseHead = serde_json::from_slice(&bytes).map_err(|_| "Invalid hub response headers")?;
        if response.version != 1 || !(200..=599).contains(&response.status) { return Err("Unsupported hub response".into()); }
        if response.headers.len() > 32 { return Err("Too many hub response headers".into()); }
        let mut seen = reqwest::header::HeaderMap::new();
        let mut expected = None;
        for (name, value) in &response.headers {
            let name = reqwest::header::HeaderName::from_bytes(name.as_bytes()).map_err(|_| "Invalid response header name")?;
            if !RESPONSE_HEADERS.contains(&name.as_str()) || seen.contains_key(&name) {
                return Err("Unsafe or duplicated hub response header".into());
            }
            let parsed = reqwest::header::HeaderValue::from_str(value).map_err(|_| "Invalid response header value")?;
            if name == reqwest::header::CONTENT_LENGTH {
                if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
                    return Err("Invalid response content length".into());
                }
                expected = Some(value.parse::<u64>().map_err(|_| "Invalid response content length")?);
            }
            seen.insert(name, parsed);
        }
        let expected = if is_head || matches!(response.status, 204 | 304) { Some(0) } else { expected };
        if expected.is_some_and(|length| length > MAX_RESPONSE) { return Err("Hub response exceeds 2 GiB".into()); }
        let deadline = Instant::now() + Duration::from_secs(15 * 60);
        let body = async_stream::try_stream! {
            let _permit = permit;
            let session = session;
            let mut count = 0u64;
            loop {
                let mut chunk = vec![0u8; 64 * 1024];
                let wait = IDLE.min(deadline.saturating_duration_since(Instant::now()));
                let read = timeout(wait, AsyncReadExt::read(&mut recv, &mut chunk)).await
                    .map_err(|_| "Hub body timed out".to_string())?
                    .map_err(|_| "Hub body stream failed".to_string())?;
                if read == 0 {
                    if expected.is_some_and(|len| len != count) { Err("Incomplete hub response body".to_string())?; }
                    break;
                }
                count += read as u64;
                if count > MAX_RESPONSE || expected.is_some_and(|len| count > len) { Err("Hub response exceeds its limit".to_string())?; }
                chunk.truncate(read);
                yield chunk;
            }
            session.endpoint.close().await;
        };
        Ok(HubResponse { status: response.status, headers: response.headers, body: Box::pin(body) })
    }).await;
    match result {
        Ok(Ok(response)) => Ok(response),
        other => {
            endpoint.close().await;
            match other {
                Ok(Err(error)) => Err(error),
                _ => Err("Hub request timed out".into()),
            }
        }
    }
}
