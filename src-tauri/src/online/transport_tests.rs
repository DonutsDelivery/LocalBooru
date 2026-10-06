// SPDX-License-Identifier: GPL-3.0-only
//! Ephemeral loopback QUIC fixtures; no hub, relay, profile or real credentials.
//! Only endpoint binding/target addresses are substituted; production connects,
//! framing, streaming and pooling run against a single-use gateway fixture.
use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::Semaphore;

struct Fixture {
    endpoint: Endpoint,
    accepted: Arc<AtomicUsize>,
    binds: Arc<AtomicUsize>,
    heads: Arc<Mutex<Vec<serde_json::Value>>>,
    release: Arc<Semaphore>,
    task: tokio::task::JoinHandle<()>,
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
impl Fixture {
    async fn new() -> Self {
        let endpoint = ephemeral().await;
        let accepted = Arc::new(AtomicUsize::new(0));
        let heads = Arc::new(Mutex::new(Vec::new()));
        let release = Arc::new(Semaphore::new(0));
        let task = {
            let server = endpoint.clone();
            let accepted = accepted.clone();
            let heads = heads.clone();
            let release = release.clone();
            tokio::spawn(async move {
                while let Some(incoming) = server.accept().await {
                    let Ok(connection) = incoming.await else {
                        continue;
                    };
                    accepted.fetch_add(1, Ordering::SeqCst);
                    let heads = heads.clone();
                    let release = release.clone();
                    tokio::spawn(async move {
                        if let Ok((mut send, mut recv)) = connection.accept_bi().await {
                            let heads = heads.clone();
                            let release = release.clone();
                            let connection = connection.clone();
                            tokio::spawn(async move {
                                let Ok(length) = AsyncReadExt::read_u32(&mut recv).await else {
                                    return;
                                };
                                if length as usize > MAX_HEAD {
                                    return;
                                }
                                let mut bytes = vec![0; length as usize];
                                if AsyncReadExt::read_exact(&mut recv, &mut bytes)
                                    .await
                                    .is_err()
                                {
                                    return;
                                }
                                let head: serde_json::Value =
                                    serde_json::from_slice(&bytes).unwrap();
                                let body_length = head["body_length"].as_u64().unwrap();
                                let mut body = vec![0; body_length as usize];
                                if AsyncReadExt::read_exact(&mut recv, &mut body)
                                    .await
                                    .is_err()
                                {
                                    return;
                                }
                                heads.lock().unwrap().push(head.clone());
                                let path = head["path"].as_str().unwrap();
                                if path.ends_with("/disconnect-post") {
                                    connection.close(0u32.into(), b"synthetic post disconnect");
                                    return;
                                }
                                let denied = head["headers"][0][1] == "synthetic-revoked";
                                let body: &[u8] =
                                    if denied { b"denied" } else { b"synthetic-body" };
                                let length = if path.ends_with("/truncated") {
                                    body.len() + 1
                                } else {
                                    body.len()
                                };
                                if path.ends_with("/bad-frame") {
                                    let _ = AsyncWriteExt::write_u32(&mut send, 0).await;
                                    let _ = send.finish();
                                    let _ = send.stopped().await;
                                    connection.close(0u32.into(), b"complete");
                                    return;
                                }
                                let headers = if path.ends_with("/bad-header") {
                                    serde_json::json!([["set-cookie", "synthetic-unsafe"]])
                                } else {
                                    serde_json::json!([["content-length", length.to_string()]])
                                };
                                let reply = serde_json::to_vec(&serde_json::json!({
                                    "version":1, "status":if denied {401} else {200},
                                    "headers":headers
                                }))
                                .unwrap();
                                if AsyncWriteExt::write_u32(&mut send, reply.len() as u32)
                                    .await
                                    .is_err()
                                    || send.write_all(&reply).await.is_err()
                                {
                                    return;
                                }
                                if path.ends_with("/held") {
                                    let Ok(_permit) = release.acquire().await else {
                                        return;
                                    };
                                }
                                let _ = send.write_all(body).await;
                                let _ = send.finish();
                                let _ = timeout(IDLE, send.stopped()).await;
                                connection.close(0u32.into(), b"complete");
                            });
                        }
                    });
                }
            })
        };
        Self {
            endpoint,
            accepted,
            heads,
            release,
            task,
            binds: Arc::new(AtomicUsize::new(0)),
        }
    }
    fn bootstrap(&self) -> HubBootstrap {
        HubBootstrap {
            peer_id: self.endpoint.id().to_string(),
            relay_url: "https://relay.fixture.invalid".into(),
        }
    }
    fn request(&self, path: &str, token: &str) -> HubRequest {
        HubRequest {
            method: "GET".into(),
            path: path.into(),
            headers: vec![("Authorization".into(), token.into())],
            body: vec![],
        }
    }
    fn pool(&self) -> Arc<SessionPool> {
        self.pool_with(&[])
    }
    fn pool_with(&self, others: &[&Fixture]) -> Arc<SessionPool> {
        let mut targets = std::collections::HashMap::new();
        targets.insert(self.endpoint.id(), self.endpoint.addr());
        for other in others {
            targets.insert(other.endpoint.id(), other.endpoint.addr());
        }
        let binds = self.binds.clone();
        let mut pool = SessionPool::new();
        pool.binder = Some(Arc::new(move |peer, _relay, _mode, permit| {
            let target = targets.get(&peer).cloned();
            let binds = binds.clone();
            Box::pin(async move {
                binds.fetch_add(1, Ordering::SeqCst);
                let target = target.ok_or("Unknown synthetic pinned peer")?;
                let endpoint = ephemeral().await;
                let owner = EndpointOwner {
                    endpoint,
                    permit: Some(permit),
                };
                assert_eq!(target.id, peer);
                Ok(Arc::new(Session { owner, target }))
            })
        }));
        Arc::new(pool)
    }
    async fn close(self) {
        self.release.add_permits(16);
        self.endpoint.close().await;
        self.task.abort();
        let _ = self.task.await;
    }
}
async fn bytes(
    fixture: &Fixture,
    pool: &Arc<SessionPool>,
    bootstrap: &HubBootstrap,
    mode: TransportMode,
) {
    let response = timeout(
        Duration::from_secs(5),
        request_with_pool(
            bootstrap,
            mode,
            fixture.request("/api/client/v1/catalog", "synthetic-token"),
            pool,
        ),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(response.into_bytes(64).await.unwrap(), b"synthetic-body");
}

// AC: @creator-community-replication ac-3
#[tokio::test]
async fn single_use_gateway_requests_reuse_endpoint_with_fresh_auth() {
    let fixture = Fixture::new().await;
    let pool = fixture.pool();
    let bootstrap = fixture.bootstrap();
    futures_util::future::join_all(
        (0..4).map(|_| bytes(&fixture, &pool, &bootstrap, TransportMode::Standard)),
    )
    .await;
    for (token, expected_status) in [
        ("synthetic-first", 200),
        ("synthetic-revoked", 401),
        ("synthetic-second", 200),
    ] {
        let response = request_with_pool(
            &bootstrap,
            TransportMode::Standard,
            fixture.request("/api/client/v1/catalog", token),
            &pool,
        )
        .await
        .unwrap();
        assert_eq!(response.status, expected_status);
        response.into_bytes(64).await.unwrap();
    }
    assert_eq!(fixture.binds.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.accepted.load(Ordering::SeqCst), 7);
    {
        let heads = fixture.heads.lock().unwrap();
        assert_eq!(heads[4]["headers"][0][1], "synthetic-first");
        assert_eq!(heads[5]["headers"][0][1], "synthetic-revoked");
        assert_eq!(heads[6]["headers"][0][1], "synthetic-second");
    }
    drop(pool);
    fixture.close().await;
}

// AC: @creator-community-replication ac-5
#[tokio::test]
async fn exact_pin_relay_and_mode_never_share_sessions() {
    let fixture = Fixture::new().await;
    let other = Fixture::new().await;
    let pool = fixture.pool_with(&[&other]);
    let bootstrap = fixture.bootstrap();
    bytes(&fixture, &pool, &bootstrap, TransportMode::Standard).await;
    bytes(&fixture, &pool, &bootstrap, TransportMode::Relayed).await;
    let another_relay = HubBootstrap {
        relay_url: "https://other-relay.fixture.invalid".into(),
        ..bootstrap.clone()
    };
    bytes(&fixture, &pool, &another_relay, TransportMode::Standard).await;
    // Exact text is intentional, even when URL parsing would normalize it.
    let same_origin = HubBootstrap {
        relay_url: format!("{}/", bootstrap.relay_url),
        ..bootstrap.clone()
    };
    bytes(&fixture, &pool, &same_origin, TransportMode::Standard).await;
    bytes(&other, &pool, &other.bootstrap(), TransportMode::Standard).await;
    assert_eq!(fixture.binds.load(Ordering::SeqCst), 5);
    assert_eq!(fixture.accepted.load(Ordering::SeqCst), 4);
    assert_eq!(other.accepted.load(Ordering::SeqCst), 1);
    for (bad, mode, path) in [
        (
            bootstrap.clone(),
            TransportMode::Anonymous,
            "/api/client/v1/catalog",
        ),
        (
            HubBootstrap {
                peer_id: "invalid".into(),
                ..bootstrap.clone()
            },
            TransportMode::Standard,
            "/api/client/v1/catalog",
        ),
        (
            HubBootstrap {
                relay_url: "http://relay.fixture.invalid".into(),
                ..bootstrap.clone()
            },
            TransportMode::Standard,
            "/api/client/v1/catalog",
        ),
        (
            bootstrap.clone(),
            TransportMode::Standard,
            "https://other.invalid/api/client/v1/catalog",
        ),
    ] {
        assert!(
            request_with_pool(&bad, mode, fixture.request(path, "synthetic-token"), &pool)
                .await
                .is_err()
        );
    }
    assert_eq!(fixture.binds.load(Ordering::SeqCst), 5);
    drop(pool);
    fixture.close().await;
    other.close().await;
}

// AC: @creator-community-replication ac-3
#[tokio::test]
async fn dropped_and_failed_bodies_leave_other_streams_alive_and_release_capacity() {
    let fixture = Fixture::new().await;
    let pool = fixture.pool();
    let bootstrap = fixture.bootstrap();
    let surviving = request_with_pool(
        &bootstrap,
        TransportMode::Standard,
        fixture.request("/api/client/v1/held", "synthetic-token"),
        &pool,
    )
    .await
    .unwrap();
    let dropped = request_with_pool(
        &bootstrap,
        TransportMode::Standard,
        fixture.request("/api/client/v1/held", "synthetic-token"),
        &pool,
    )
    .await
    .unwrap();
    drop(dropped);
    let truncated = request_with_pool(
        &bootstrap,
        TransportMode::Standard,
        fixture.request("/api/client/v1/truncated", "synthetic-token"),
        &pool,
    )
    .await
    .unwrap();
    assert!(truncated
        .into_bytes(64)
        .await
        .unwrap_err()
        .contains("Incomplete"));
    let oversized = request_with_pool(
        &bootstrap,
        TransportMode::Standard,
        fixture.request("/api/client/v1/catalog", "synthetic-token"),
        &pool,
    )
    .await
    .unwrap();
    assert!(oversized.into_bytes(2).await.is_err());
    bytes(&fixture, &pool, &bootstrap, TransportMode::Standard).await;
    fixture.release.add_permits(2);
    assert_eq!(surviving.into_bytes(64).await.unwrap(), b"synthetic-body");
    assert_eq!(fixture.binds.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.accepted.load(Ordering::SeqCst), 5);
    assert_eq!(
        pool.slots.lock().unwrap()[0].active.load(Ordering::SeqCst),
        0
    );
    drop(pool);
    fixture.close().await;
}

// AC: @creator-community-replication ac-3
#[tokio::test]
async fn failed_request_connection_never_replays_post_and_next_request_succeeds() {
    let fixture = Fixture::new().await;
    let pool = fixture.pool();
    let bootstrap = fixture.bootstrap();
    bytes(&fixture, &pool, &bootstrap, TransportMode::Standard).await;
    let mut post = fixture.request("/api/client/v1/disconnect-post", "synthetic-token");
    post.method = "POST".into();
    post.body = b"synthetic-mutation".to_vec();
    assert!(
        request_with_pool(&bootstrap, TransportMode::Standard, post, &pool)
            .await
            .is_err()
    );
    assert_eq!(fixture.binds.load(Ordering::SeqCst), 1);
    bytes(&fixture, &pool, &bootstrap, TransportMode::Standard).await;
    assert_eq!(fixture.binds.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.accepted.load(Ordering::SeqCst), 3);
    assert_eq!(
        fixture
            .heads
            .lock()
            .unwrap()
            .iter()
            .filter(|head| head["method"] == "POST")
            .count(),
        1
    );
    drop(pool);
    fixture.close().await;
}

// AC: @creator-community-replication ac-3
#[tokio::test]
async fn cache_is_bounded_and_idle_eviction_never_closes_live_body() {
    let fixture = Fixture::new().await;
    let pool = fixture.pool();
    let bootstrap = fixture.bootstrap();
    let live = request_with_pool(
        &bootstrap,
        TransportMode::Standard,
        fixture.request("/api/client/v1/held", "synthetic-token"),
        &pool,
    )
    .await
    .unwrap();
    let slot = pool.slots.lock().unwrap()[0].clone();
    let endpoint = slot
        .session
        .lock()
        .await
        .as_ref()
        .unwrap()
        .owner
        .endpoint
        .clone();
    drop(slot);
    for index in 0..MAX_SESSIONS + 2 {
        let other = HubBootstrap {
            relay_url: format!("https://relay-{index}.fixture.invalid"),
            ..bootstrap.clone()
        };
        bytes(&fixture, &pool, &other, TransportMode::Standard).await;
        assert!(pool.slots.lock().unwrap().len() <= MAX_SESSIONS);
    }
    pool.prune(Instant::now() + SESSION_IDLE + Duration::from_secs(1));
    assert_eq!(pool.slots.lock().unwrap().len(), 1);
    assert!(!endpoint.is_closed());
    fixture.release.add_permits(1);
    assert_eq!(live.into_bytes(64).await.unwrap(), b"synthetic-body");
    pool.prune(Instant::now() + SESSION_IDLE + Duration::from_secs(1));
    assert!(pool.slots.lock().unwrap().is_empty());
    timeout(Duration::from_secs(5), endpoint.closed())
        .await
        .unwrap();
    timeout(Duration::from_secs(5), async {
        while pool.endpoints.available_permits() != MAX_SESSIONS * 2 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    drop(pool);
    fixture.close().await;
}

// AC: @creator-community-replication ac-3
#[tokio::test]
async fn cancelled_connection_setup_releases_lease_and_endpoint_budget() {
    let fixture = Fixture::new().await;
    let original = fixture.pool();
    let delegate = original.binder.as_ref().unwrap().clone();
    let (started, observed) = tokio::sync::oneshot::channel();
    let started = Arc::new(Mutex::new(Some(started)));
    let mut configured = SessionPool::new();
    configured.binder = Some(Arc::new(move |peer, relay, mode, permit| {
        let delegate = delegate.clone();
        let started = started.lock().unwrap().take();
        Box::pin(async move {
            if let Some(started) = started {
                let endpoint = ephemeral().await;
                let owner = EndpointOwner {
                    endpoint,
                    permit: Some(permit),
                };
                let _ = started.send(owner.endpoint.clone());
                std::future::pending::<()>().await;
                drop(owner);
                unreachable!();
            }
            delegate(peer, relay, mode, permit).await
        })
    }));
    let pool = Arc::new(configured);
    let bootstrap = fixture.bootstrap();
    let waiting = {
        let pool = pool.clone();
        let bootstrap = bootstrap.clone();
        let request = fixture.request("/api/client/v1/catalog", "synthetic-token");
        tokio::spawn(async move {
            request_with_pool(&bootstrap, TransportMode::Standard, request, &pool).await
        })
    };
    let endpoint = timeout(Duration::from_secs(5), observed)
        .await
        .unwrap()
        .unwrap();
    waiting.abort();
    assert!(matches!(waiting.await, Err(error) if error.is_cancelled()));
    timeout(Duration::from_secs(5), endpoint.closed())
        .await
        .unwrap();
    assert_eq!(
        pool.slots.lock().unwrap()[0].active.load(Ordering::SeqCst),
        0
    );
    bytes(&fixture, &pool, &bootstrap, TransportMode::Standard).await;
    assert_eq!(fixture.binds.load(Ordering::SeqCst), 1);
    drop(endpoint);
    drop(original);
    drop(pool);
    fixture.close().await;
}

// AC: @creator-community-replication ac-5
#[tokio::test]
async fn invalid_response_frames_and_headers_do_not_damage_other_connections() {
    let fixture = Fixture::new().await;
    let pool = fixture.pool();
    let bootstrap = fixture.bootstrap();
    let surviving = request_with_pool(
        &bootstrap,
        TransportMode::Standard,
        fixture.request("/api/client/v1/held", "synthetic-token"),
        &pool,
    )
    .await
    .unwrap();
    for (path, expected) in [
        ("/api/client/v1/bad-frame", "frame length"),
        ("/api/client/v1/bad-header", "Unsafe or duplicated"),
    ] {
        let result = request_with_pool(
            &bootstrap,
            TransportMode::Standard,
            fixture.request(path, "synthetic-token"),
            &pool,
        )
        .await;
        assert!(matches!(result, Err(error) if error.contains(expected)));
    }
    bytes(&fixture, &pool, &bootstrap, TransportMode::Standard).await;
    fixture.release.add_permits(1);
    assert_eq!(surviving.into_bytes(64).await.unwrap(), b"synthetic-body");
    assert_eq!(fixture.binds.load(Ordering::SeqCst), 1);
    assert_eq!(fixture.accepted.load(Ordering::SeqCst), 4);
    drop(pool);
    fixture.close().await;
}

// AC: @creator-community-replication ac-3
#[tokio::test]
async fn closed_endpoint_is_rebound_and_background_reaper_has_weak_lifetime() {
    let fixture = Fixture::new().await;
    let pool = fixture.pool();
    let bootstrap = fixture.bootstrap();
    bytes(&fixture, &pool, &bootstrap, TransportMode::Standard).await;
    let slot = pool.slots.lock().unwrap()[0].clone();
    let endpoint = slot
        .session
        .lock()
        .await
        .as_ref()
        .unwrap()
        .owner
        .endpoint
        .clone();
    endpoint.close().await;
    bytes(&fixture, &pool, &bootstrap, TransportMode::Standard).await;
    assert_eq!(fixture.binds.load(Ordering::SeqCst), 2);
    *slot.last_used.lock().unwrap() = Instant::now() - SESSION_IDLE - Duration::from_secs(1);
    drop(slot);
    SessionPool::start_reaper(&pool, Duration::from_millis(10));
    timeout(Duration::from_secs(5), async {
        while !pool.slots.lock().unwrap().is_empty() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let weak = Arc::downgrade(&pool);
    drop(pool);
    timeout(Duration::from_secs(5), async {
        while weak.upgrade().is_some() {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    drop(endpoint);
    fixture.close().await;
}
