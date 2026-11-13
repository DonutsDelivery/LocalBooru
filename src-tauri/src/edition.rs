//! Immutable edition policy shared by native startup, HTTP, and the frontend.

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub struct EditionCapabilities {
    pub edition: &'static str,
    pub community_hubs: bool,
    pub downloaded_executable_addons: bool,
    pub lan_library: bool,
    pub self_updates: bool,
}

pub const fn capabilities() -> EditionCapabilities {
    if cfg!(feature = "store-lite") {
        EditionCapabilities {
            edition: "store-lite",
            community_hubs: false,
            downloaded_executable_addons: false,
            lan_library: true,
            self_updates: false,
        }
    } else {
        EditionCapabilities {
            edition: "full",
            community_hubs: true,
            downloaded_executable_addons: true,
            lan_library: true,
            self_updates: true,
        }
    }
}

pub const UNAVAILABLE: &str = "This feature is not included in DonutMediaCenter Lite";

pub fn require_community() -> Result<(), String> {
    if capabilities().community_hubs {
        Ok(())
    } else {
        Err(UNAVAILABLE.into())
    }
}

pub fn addon_allowed(id: &str) -> bool {
    capabilities().downloaded_executable_addons
        || crate::addons::manifest::get_addon_manifest(id).is_some_and(|manifest| {
            manifest.installation == crate::addons::manifest::AddonInstallation::Builtin
        })
}

pub fn require_addon(id: &str) -> Result<(), String> {
    if addon_allowed(id) {
        Ok(())
    } else {
        Err(UNAVAILABLE.into())
    }
}

/// The online API is the community-provider catalog, not device pairing or LAN
/// library playback. Apply the same policy before forwarding to a full server.
pub fn request_allowed(path: &str) -> bool {
    if capabilities().community_hubs && capabilities().downloaded_executable_addons {
        return true;
    }
    let Ok(url) = reqwest::Url::parse(&format!("http://edition.invalid{path}")) else {
        return false;
    };
    let mut path = url.path();
    while path.starts_with("/remote/") {
        path = &path["/remote".len()..];
    }
    let within = |prefix: &str| {
        path == prefix
            || path
                .strip_prefix(prefix)
                .is_some_and(|tail| tail.starts_with('/'))
    };
    if !capabilities().community_hubs && within("/api/online") {
        return false;
    }
    if !capabilities().downloaded_executable_addons {
        if within("/api/create") && path != "/api/create/workflow" {
            return false;
        }
        if ["/api/app-update", "/api/settings/svp", "/api/cast"]
            .iter()
            .any(|prefix| within(prefix))
        {
            return false;
        }
        if let Some(tail) = path.strip_prefix("/api/addons/") {
            if let Some(id) = tail.split('/').find(|part| !part.is_empty()) {
                return addon_allowed(id);
            }
        }
    }
    true
}

pub async fn enforce(
    request: axum::extract::Request,
    next: axum::middleware::Next,
) -> axum::response::Response {
    use axum::response::IntoResponse;
    if !request_allowed(request.uri().path()) {
        return (
            axum::http::StatusCode::FORBIDDEN,
            axum::Json(serde_json::json!({"detail": UNAVAILABLE})),
        )
            .into_response();
    }
    next.run(request).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn policy_preserves_library_pairing_and_builtin_tools() {
        for path in [
            "/health",
            "/api/edition",
            "/api/images",
            "/api/device-pairing/desktop/start",
            "/api/libraries",
            "/api/music",
            "/api/create/workflow",
            "/api/settings/wd14-sidecar/import",
            "/api/network",
            "/api/share",
            "/remote/api/images/7/file",
            "/api/addons",
            "/api/addons/curation-game/install",
        ] {
            assert!(
                request_allowed(path),
                "blocked ordinary library path: {path}"
            );
        }
        assert!(require_addon("curation-game").is_ok());
    }

    #[test]
    fn policy_covers_remote_proxy_traversal_and_executable_addons() {
        for path in [
            "/api/online/hubs",
            "/api/online/sources/synthetic/browse",
            "/remote/api/online/hubs",
            "/remote/remote/api/online/hubs",
            "/remote/api/images/%2e%2e/online/hubs",
            "/api/addons/donut-create/install",
            "/api/addons/donut-create/api/create/sessions",
            "/api/addons/lada/probe",
            "/api/addons/auto-tagger/update",
            "/remote/api/create/studio/session/ws",
            "/api/app-update/download",
        ] {
            assert_eq!(
                request_allowed(path),
                !cfg!(feature = "store-lite"),
                "wrong policy for {path}"
            );
        }
        assert_eq!(require_community().is_err(), cfg!(feature = "store-lite"));
        assert_eq!(
            require_addon("donut-create").is_err(),
            cfg!(feature = "store-lite")
        );
    }

    #[tokio::test]
    async fn http_policy_blocks_direct_requests_before_any_handler_runs() {
        use axum::{
            body::Body,
            http::{Request, StatusCode},
            middleware,
            routing::any,
            Router,
        };
        use tower::ServiceExt;
        let router = Router::new()
            .route("/api/online/hubs", any(|| async { StatusCode::NO_CONTENT }))
            .route(
                "/remote/api/addons/donut-create/start",
                any(|| async { StatusCode::NO_CONTENT }),
            )
            .route(
                "/api/device-pairing/desktop/start",
                any(|| async { StatusCode::NO_CONTENT }),
            )
            .layer(middleware::from_fn(enforce));
        for path in [
            "/api/online/hubs",
            "/remote/api/addons/donut-create/start",
            "/api/device-pairing/desktop/start",
        ] {
            let response = router
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(path)
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            let allowed = !cfg!(feature = "store-lite") || path.contains("device-pairing");
            assert_eq!(
                response.status(),
                if allowed {
                    StatusCode::NO_CONTENT
                } else {
                    StatusCode::FORBIDDEN
                }
            );
        }
    }

    #[test]
    fn compiled_profile_preserves_lan_and_selects_capabilities() {
        let profile = capabilities();
        assert!(profile.lan_library);
        assert_eq!(profile.community_hubs, !cfg!(feature = "store-lite"));
        assert_eq!(
            profile.downloaded_executable_addons,
            !cfg!(feature = "store-lite")
        );
    }
}
