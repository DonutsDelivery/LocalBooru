// SPDX-License-Identifier: GPL-3.0-only
//! Native hub contract validation. Hub URLs are labels for the authenticated Iroh
//! peer's origin, never permission to fall back to a direct HTTP connection.
use super::{
    models::{RemoteItem, RemoteMediaVariant, RemotePage, SourceCapabilities},
    security,
};
use crate::server::error::AppError;
use reqwest::Url;
use serde_json::Value;

pub fn origin(raw: &str, allow_local: bool) -> Result<String, AppError> {
    let base = security::normalize_base_url(raw, allow_local)?;
    let url = Url::parse(&base).map_err(|_| AppError::BadRequest("Invalid hub origin".into()))?;
    if url.path() != "/" && !url.path().is_empty() {
        return Err(AppError::BadRequest(
            "Hub URL must be an origin without a path".into(),
        ));
    }
    Ok(base)
}
pub fn path(base: &str, raw: &str) -> Result<String, AppError> {
    let origin = Url::parse(base).map_err(|_| AppError::BadRequest("Invalid hub origin".into()))?;
    let url = origin
        .join(raw)
        .map_err(|_| AppError::BadRequest("Invalid hub URL".into()))?;
    if url.origin() != origin.origin()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
        || !url.path().starts_with("/api/client/v1/")
    {
        return Err(AppError::Forbidden(
            "Hub endpoint must remain on its enrolled origin and client API".into(),
        ));
    }
    Ok(format!(
        "{}{}",
        url.path(),
        url.query().map(|q| format!("?{q}")).unwrap_or_default()
    ))
}
pub fn validate_descriptor(base: &str, descriptor: &Value) -> Result<String, AppError> {
    if descriptor.get("protocol").and_then(Value::as_str) != Some("booru-node-client")
        || descriptor.get("version").and_then(Value::as_str) != Some("1.1")
    {
        return Err(AppError::BadRequest(
            "Hub must implement booru-node-client version 1.1".into(),
        ));
    }
    let valid_text = |pointer: &str, max: usize| {
        descriptor
            .pointer(pointer)
            .and_then(Value::as_str)
            .is_some_and(|text| !text.is_empty() && text.len() <= max)
    };
    if !valid_text("/instance/name", 256)
        || !matches!(
            descriptor
                .pointer("/access/catalog")
                .and_then(Value::as_str),
            Some("public" | "members")
        )
        || !matches!(
            descriptor
                .pointer("/access/registration")
                .and_then(Value::as_str),
            Some("open" | "closed")
        )
        || !matches!(
            descriptor
                .pointer("/access/submission")
                .and_then(Value::as_str),
            Some("members" | "closed")
        )
        || descriptor
            .pointer("/access/moderation")
            .and_then(Value::as_str)
            != Some("required")
        || descriptor
            .pointer("/access/retention")
            .and_then(Value::as_str)
            != Some("hosted")
    {
        return Err(AppError::BadRequest(
            "Hub descriptor has invalid identity or policy fields".into(),
        ));
    }
    let methods = descriptor
        .pointer("/authentication/methods")
        .and_then(Value::as_array)
        .ok_or_else(|| AppError::BadRequest("Hub descriptor has no valid login methods".into()))?;
    if methods.len() > 16
        || methods.iter().any(|method| {
            !method
                .as_str()
                .is_some_and(|name| !name.is_empty() && name.len() <= 64)
        })
    {
        return Err(AppError::BadRequest(
            "Hub descriptor has invalid login methods".into(),
        ));
    }
    let pkce = descriptor
        .pointer("/authentication/pkce_methods")
        .and_then(Value::as_array)
        .ok_or_else(|| AppError::BadRequest("Hub descriptor has no valid PKCE methods".into()))?;
    if pkce.len() > 16 || pkce.iter().any(|method| method.as_str().is_none()) {
        return Err(AppError::BadRequest(
            "Hub descriptor has invalid PKCE methods".into(),
        ));
    }
    for flag in [
        "catalog",
        "publish",
        "idempotency",
        "revisions",
        "tombstones",
        "byte_ranges",
        "native_authorization",
    ] {
        if descriptor
            .pointer(&format!("/capabilities/{flag}"))
            .and_then(Value::as_bool)
            .is_none()
        {
            return Err(AppError::BadRequest(
                "Hub descriptor has invalid capability flags".into(),
            ));
        }
    }
    if descriptor
        .pointer("/capabilities/native_authorization")
        .and_then(Value::as_bool)
        == Some(true)
        && (methods.is_empty() || !pkce.iter().any(|method| method.as_str() == Some("S256")))
    {
        return Err(AppError::BadRequest(
            "Hub advertises unavailable native authorization".into(),
        ));
    }
    let instance_id = descriptor
        .pointer("/instance/id")
        .and_then(Value::as_str)
        .filter(|id| uuid::Uuid::parse_str(id).is_ok())
        .ok_or_else(|| {
            AppError::BadRequest("Hub descriptor has no valid instance identity".into())
        })?;
    let advertised = descriptor
        .pointer("/instance/origin")
        .and_then(Value::as_str)
        .ok_or_else(|| AppError::BadRequest("Hub descriptor has no canonical origin".into()))?;
    if advertised.trim_end_matches('/') != base {
        return Err(AppError::Forbidden(
            "Hub descriptor origin differs from the enrolled origin".into(),
        ));
    }
    for (field, expected) in [
        ("authorization_endpoint", "/api/client/v1/authorize"),
        ("token_endpoint", "/api/client/v1/token"),
    ] {
        if descriptor
            .pointer(&format!("/authentication/{field}"))
            .and_then(Value::as_str)
            != Some(expected)
        {
            return Err(AppError::BadRequest(
                "Hub authentication endpoints do not match the native contract".into(),
            ));
        }
    }
    Ok(instance_id.to_string())
}
pub fn capabilities(value: &Value) -> SourceCapabilities {
    let flag = |name: &str| {
        value
            .pointer(&format!("/capabilities/{name}"))
            .and_then(Value::as_bool)
            .unwrap_or(false)
    };
    let mut caps = SourceCapabilities::read_only();
    caps.browse = flag("catalog");
    caps.search = caps.browse;
    caps.fetch_detail = caps.browse;
    caps.fetch_original = caps.browse;
    caps.authenticate = flag("native_authorization");
    caps.upload = flag("publish");
    caps.edit_metadata = flag("revisions");
    caps.delete_remote = flag("tombstones");
    caps.moderation_state = caps.upload;
    caps.byte_ranges = flag("byte_ranges");
    caps
}
pub fn catalog_path(query: &str, page: u32, per_page: u32) -> Result<String, AppError> {
    let url = Url::parse_with_params(
        "https://hub.invalid/api/client/v1/catalog",
        &[
            ("collection", "community".to_string()),
            ("tags", query.to_string()),
            ("page", page.to_string()),
            ("per_page", per_page.to_string()),
        ],
    )
    .map_err(|_| AppError::BadRequest("Invalid catalog search".into()))?;
    Ok(format!(
        "{}?{}",
        url.path(),
        url.query().unwrap_or_default()
    ))
}
pub fn catalog_item(source_id: &str, base: &str, row: &Value) -> Result<RemoteItem, AppError> {
    let text = |key: &str| row.get(key).and_then(Value::as_str).map(str::to_string);
    let id = text("remote_post_id")
        .ok_or_else(|| AppError::ServiceUnavailable("Invalid hub catalog item".into()))?;
    if id.is_empty() || id.len() > 20 || !id.bytes().all(|ch| ch.is_ascii_digit()) {
        return Err(AppError::ServiceUnavailable(
            "Invalid numeric hub catalog identity".into(),
        ));
    }
    let canonical = text("canonical_url")
        .ok_or_else(|| AppError::ServiceUnavailable("Invalid hub catalog URL".into()))?;
    let origin = Url::parse(base).map_err(|_| AppError::BadRequest("Invalid hub origin".into()))?;
    let canonical_url = origin
        .join(&canonical)
        .map_err(|_| AppError::ServiceUnavailable("Invalid canonical URL".into()))?;
    if canonical_url.origin() != origin.origin()
        || !canonical_url.username().is_empty()
        || canonical_url.password().is_some()
    {
        return Err(AppError::Forbidden(
            "Hub canonical URL belongs to another origin".into(),
        ));
    }
    let variants = row
        .get("variants")
        .and_then(Value::as_array)
        .ok_or_else(|| AppError::ServiceUnavailable("Invalid hub media variants".into()))?;
    let mut media = Vec::new();
    for variant in variants {
        let raw = variant
            .get("asset_url")
            .and_then(Value::as_str)
            .ok_or_else(|| AppError::ServiceUnavailable("Invalid hub asset URL".into()))?;
        let asset_path = path(base, raw)?;
        if !asset_path.starts_with("/api/client/v1/media/") {
            return Err(AppError::Forbidden(
                "Hub assets must use protected client media endpoints".into(),
            ));
        }
        media.push(RemoteMediaVariant {
            kind: variant
                .get("kind")
                .and_then(Value::as_str)
                .unwrap_or("original")
                .to_string(),
            asset_url: format!("{base}{asset_path}"),
            mime_type: variant
                .get("mime_type")
                .and_then(Value::as_str)
                .map(str::to_string),
            width: variant
                .get("width")
                .and_then(Value::as_u64)
                .and_then(|n| u32::try_from(n).ok()),
            height: variant
                .get("height")
                .and_then(Value::as_u64)
                .and_then(|n| u32::try_from(n).ok()),
        });
    }
    Ok(RemoteItem {
        source_id: source_id.into(),
        remote_post_id: id,
        canonical_url: canonical_url.to_string(),
        remote_revision: text("remote_revision"),
        title: text("caption"),
        caption: text("caption"),
        alt_text: text("alt_text"),
        content_warning: text("content_warning"),
        creator: row
            .get("creator")
            .filter(|value| !value.is_null())
            .map(|value| serde_json::from_value(value.clone()))
            .transpose()
            .map_err(|_| AppError::ServiceUnavailable("Invalid public hub creator".into()))?,
        tags: row
            .get("tags")
            .and_then(Value::as_array)
            .map(|t| {
                t.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default(),
        rating: text("rating"),
        source_url: None,
        width: media.iter().find_map(|m| m.width),
        height: media.iter().find_map(|m| m.height),
        duration: None,
        media,
    })
}
pub fn catalog_page(
    source_id: &str,
    base: &str,
    value: Value,
    page: u32,
    per_page: u32,
) -> Result<RemotePage, AppError> {
    let rows = value
        .get("items")
        .and_then(Value::as_array)
        .ok_or_else(|| AppError::ServiceUnavailable("Invalid hub catalog page".into()))?;
    Ok(RemotePage {
        items: rows
            .iter()
            .map(|r| catalog_item(source_id, base, r))
            .collect::<Result<_, _>>()?,
        page,
        per_page,
        total: value.get("total").and_then(Value::as_u64),
        stale: false,
        next_cursor: value
            .get("has_more")
            .and_then(Value::as_bool)
            .filter(|v| *v)
            .map(|_| (page + 1).to_string()),
    })
}
