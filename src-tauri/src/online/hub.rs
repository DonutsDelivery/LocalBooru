// SPDX-License-Identifier: GPL-3.0-only
//! Native hub contract validation. Hub URLs are labels for the authenticated Iroh
//! peer's origin, never permission to fall back to a direct HTTP connection.
use super::{
    models::{RemoteItem, RemoteMediaVariant, RemotePage, SourceCapabilities},
    security,
};
use crate::server::error::AppError;
use reqwest::Url;
use serde::{Deserialize, Serialize};
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
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct CatalogQuery {
    pub q: Option<String>,
    pub creator_id: Option<String>,
    #[serde(default)]
    pub offline_only: bool,
    pub page: Option<u32>,
    pub per_page: Option<u32>,
    pub media_type: Option<String>,
    pub rating: Option<String>,
    pub sort: Option<String>,
    pub timeframe: Option<String>,
    #[serde(default)]
    pub favorites_only: bool,
    pub min_age: Option<i32>,
    pub max_age: Option<i32>,
    pub min_width: Option<i32>,
    pub min_height: Option<i32>,
    pub orientation: Option<String>,
    pub min_duration: Option<f64>,
    pub max_duration: Option<f64>,
}
impl CatalogQuery {
    pub fn normalized(mut self) -> Result<Self, AppError> {
        self.q = Some(
            self.q
                .unwrap_or_default()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" "),
        );
        if self
            .q
            .as_ref()
            .is_some_and(|q| q.len() > 1000 || q.split_whitespace().count() > 32)
        {
            return Err(AppError::BadRequest(
                "Use at most 32 remote tag terms and 1000 bytes".into(),
            ));
        }
        if self.creator_id.as_deref().is_some_and(|id| {
            id.is_empty()
                || id.len() > 128
                || !id
                    .bytes()
                    .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
        }) {
            return Err(AppError::BadRequest(
                "Invalid source creator identity".into(),
            ));
        }
        self.page = Some(self.page.unwrap_or(1).clamp(1, 10000));
        self.per_page = Some(self.per_page.unwrap_or(40).clamp(1, 100));
        if self.media_type.as_deref() == Some("all") {
            self.media_type = None;
        }
        if self
            .media_type
            .as_deref()
            .is_some_and(|value| !["image", "video"].contains(&value))
        {
            return Err(AppError::BadRequest(
                "media_type must be image or video".into(),
            ));
        }
        if let Some(ratings) = self.rating.take() {
            let values = ratings
                .split(',')
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .collect::<std::collections::BTreeSet<_>>();
            if values
                .iter()
                .any(|value| !["pg", "pg13", "r", "x", "xxx"].contains(value))
            {
                return Err(AppError::BadRequest("Invalid remote rating filter".into()));
            }
            if !values.is_empty() {
                self.rating = Some(values.into_iter().collect::<Vec<_>>().join(","));
            }
        }
        self.sort = Some(self.sort.unwrap_or_else(|| "newest".into()));
        if self.sort.as_deref().is_some_and(|value| {
            ![
                "newest",
                "oldest",
                "filename_asc",
                "filename_desc",
                "filesize_largest",
                "filesize_smallest",
                "resolution_high",
                "resolution_low",
                "duration_longest",
                "duration_shortest",
                "random",
                "popular",
                "views",
            ]
            .contains(&value)
        }) {
            return Err(AppError::BadRequest(
                "Unsupported remote catalog sort".into(),
            ));
        }
        self.timeframe = match self.timeframe.as_deref() {
            Some("all") | None => None,
            Some("today") => Some("day".into()),
            _ => self.timeframe,
        };
        if self
            .timeframe
            .as_deref()
            .is_some_and(|value| !["day", "week", "month", "year"].contains(&value))
        {
            return Err(AppError::BadRequest("Unsupported remote timeframe".into()));
        }
        if self.min_age.is_some() || self.max_age.is_some() {
            return Err(AppError::Forbidden("Remote detected subject ages are moderation-only and cannot be filtered by a catalog grant".into()));
        }
        if self.min_width.is_some_and(|n| n < 1)
            || self.min_height.is_some_and(|n| n < 1)
            || self.min_duration.is_some_and(|n| !n.is_finite() || n < 0.0)
            || self.max_duration.is_some_and(|n| !n.is_finite() || n < 0.0)
            || self
                .min_duration
                .zip(self.max_duration)
                .is_some_and(|(min, max)| min > max)
        {
            return Err(AppError::BadRequest("Invalid remote filter range".into()));
        }
        if self
            .orientation
            .as_deref()
            .is_some_and(|value| !["landscape", "portrait", "square"].contains(&value))
        {
            return Err(AppError::BadRequest("Invalid remote orientation".into()));
        }
        Ok(self)
    }
    pub fn uses_hub_filters(&self) -> bool {
        self.creator_id.is_some()
            || self.offline_only
            || self.media_type.is_some()
            || self.rating.is_some()
            || self.sort.as_deref().is_some_and(|sort| sort != "newest")
            || self.timeframe.is_some()
            || self.favorites_only
            || self.min_width.is_some()
            || self.min_height.is_some()
            || self.orientation.is_some()
            || self.min_duration.is_some()
            || self.max_duration.is_some()
    }
}
pub fn catalog_path(query: &str, page: u32, per_page: u32) -> Result<String, AppError> {
    catalog_path_for(
        &CatalogQuery {
            q: Some(query.into()),
            page: Some(page),
            per_page: Some(per_page),
            ..Default::default()
        }
        .normalized()?,
    )
}
pub fn catalog_path_for(query: &CatalogQuery) -> Result<String, AppError> {
    let mut pairs = vec![
        ("collection", "community".to_string()),
        ("tags", query.q.clone().unwrap_or_default()),
        ("page", query.page.unwrap_or(1).to_string()),
        ("per_page", query.per_page.unwrap_or(40).to_string()),
        (
            "sort",
            query.sort.clone().unwrap_or_else(|| "newest".into()),
        ),
    ];
    for (name, value) in [
        ("creator_id", &query.creator_id),
        ("media_type", &query.media_type),
        ("rating", &query.rating),
        ("timeframe", &query.timeframe),
        ("orientation", &query.orientation),
    ] {
        if let Some(value) = value {
            pairs.push((name, value.clone()));
        }
    }
    if query.favorites_only {
        pairs.push(("favorites_only", "true".into()));
    }
    for (name, value) in [
        ("min_width", query.min_width),
        ("min_height", query.min_height),
    ] {
        if let Some(value) = value {
            pairs.push((name, value.to_string()));
        }
    }
    for (name, value) in [
        ("min_duration", query.min_duration),
        ("max_duration", query.max_duration),
    ] {
        if let Some(value) = value {
            pairs.push((name, value.to_string()));
        }
    }
    let url = Url::parse_with_params("https://hub.invalid/api/client/v1/catalog", pairs)
        .map_err(|_| AppError::BadRequest("Invalid catalog search".into()))?;
    Ok(format!(
        "{}?{}",
        url.path(),
        url.query().unwrap_or_default()
    ))
}
pub fn catalog_tags_path(
    query: &CatalogQuery,
    prefix: &str,
    limit: u32,
) -> Result<String, AppError> {
    if prefix.len() > 100 {
        return Err(AppError::BadRequest("Tag prefix exceeds 100 bytes".into()));
    }
    let catalog = Url::parse(&format!("https://hub.invalid{}", catalog_path_for(query)?))
        .map_err(|_| AppError::BadRequest("Invalid catalog facet filters".into()))?;
    let mut pairs = catalog
        .query_pairs()
        .into_owned()
        .filter(|(name, _)| !matches!(name.as_str(), "collection" | "page" | "per_page" | "sort"))
        .collect::<Vec<_>>();
    pairs.push(("query".into(), prefix.into()));
    pairs.push(("limit".into(), limit.clamp(1, 50).to_string()));
    let url = Url::parse_with_params("https://hub.invalid/api/client/v1/catalog/tags", pairs)
        .map_err(|_| AppError::BadRequest("Invalid catalog facet request".into()))?;
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
        let sha256 = variant
            .get("sha256")
            .and_then(Value::as_str)
            .map(str::to_ascii_lowercase);
        if sha256.as_ref().is_some_and(|digest| {
            digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit())
        }) {
            return Err(AppError::ServiceUnavailable(
                "Invalid authoritative variant digest".into(),
            ));
        }
        let byte_size = variant.get("byte_size").and_then(Value::as_u64);
        media.push(RemoteMediaVariant {
            kind: variant
                .get("kind")
                .and_then(Value::as_str)
                .unwrap_or("original")
                .to_string(),
            asset_url: format!("{base}{asset_path}"),
            sha256,
            byte_size,
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
        duration: row
            .get("duration")
            .and_then(Value::as_f64)
            .filter(|duration| duration.is_finite() && *duration >= 0.0),
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
        available_ratings: value
            .get("available_ratings")
            .and_then(Value::as_array)
            .map(|ratings| {
                ratings
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default(),
        stale: false,
        offline: false,
        partial: false,
        next_cursor: value
            .get("has_more")
            .and_then(Value::as_bool)
            .filter(|v| *v)
            .map(|_| (page + 1).to_string()),
    })
}

#[cfg(test)]
mod catalog_filter_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn catalog_filters_are_forwarded_in_the_server_query() {
        let query = CatalogQuery {
            q: Some("  cat   -dog ".into()),
            page: Some(3),
            per_page: Some(60),
            media_type: Some("video".into()),
            rating: Some("pg13,pg,pg13".into()),
            sort: Some("filename_asc".into()),
            timeframe: Some("today".into()),
            favorites_only: true,
            min_width: Some(1920),
            min_height: Some(1080),
            orientation: Some("landscape".into()),
            min_duration: Some(0.5),
            max_duration: Some(120.0),
            ..Default::default()
        }
        .normalized()
        .unwrap();
        let path = catalog_path_for(&query).unwrap();
        let url = Url::parse(&format!("https://hub.invalid{path}")).unwrap();
        let pairs = url
            .query_pairs()
            .into_owned()
            .collect::<std::collections::HashMap<_, _>>();
        for (name, value) in [
            ("tags", "cat -dog"),
            ("page", "3"),
            ("per_page", "60"),
            ("media_type", "video"),
            ("rating", "pg,pg13"),
            ("sort", "filename_asc"),
            ("timeframe", "day"),
            ("favorites_only", "true"),
            ("min_width", "1920"),
            ("min_height", "1080"),
            ("orientation", "landscape"),
            ("min_duration", "0.5"),
            ("max_duration", "120"),
        ] {
            assert_eq!(pairs.get(name).map(String::as_str), Some(value), "{name}");
        }
        assert!(!pairs.contains_key("min_age"));
    }

    #[test]
    fn equivalent_queries_share_normalization_and_distinct_filters_do_not() {
        let left = CatalogQuery {
            q: Some("cat  -dog".into()),
            rating: Some("pg13,pg,pg13".into()),
            timeframe: Some("today".into()),
            ..Default::default()
        }
        .normalized()
        .unwrap();
        let right = CatalogQuery {
            q: Some("cat -dog".into()),
            rating: Some("pg,pg13".into()),
            timeframe: Some("day".into()),
            page: Some(1),
            per_page: Some(40),
            sort: Some("newest".into()),
            ..Default::default()
        }
        .normalized()
        .unwrap();
        assert_eq!(
            serde_json::to_string(&left).unwrap(),
            serde_json::to_string(&right).unwrap()
        );
        assert_eq!(
            catalog_path_for(&left).unwrap(),
            catalog_path_for(&right).unwrap()
        );
        for changed in [
            CatalogQuery {
                media_type: Some("image".into()),
                ..left.clone()
            },
            CatalogQuery {
                favorites_only: true,
                ..left.clone()
            },
            CatalogQuery {
                min_width: Some(100),
                ..left.clone()
            },
            CatalogQuery {
                min_height: Some(100),
                ..left.clone()
            },
            CatalogQuery {
                orientation: Some("portrait".into()),
                ..left.clone()
            },
            CatalogQuery {
                min_duration: Some(10.0),
                ..left.clone()
            },
            CatalogQuery {
                max_duration: Some(60.0),
                ..left.clone()
            },
            CatalogQuery {
                sort: Some("oldest".into()),
                ..left.clone()
            },
        ] {
            assert_ne!(
                catalog_path_for(&left).unwrap(),
                catalog_path_for(&changed).unwrap()
            );
            assert_ne!(
                serde_json::to_string(&left).unwrap(),
                serde_json::to_string(&changed).unwrap()
            );
        }
    }

    #[test]
    fn catalog_grant_cannot_filter_private_detected_subject_ages() {
        for query in [
            CatalogQuery {
                min_age: Some(0),
                ..Default::default()
            },
            CatalogQuery {
                max_age: Some(90),
                ..Default::default()
            },
        ] {
            assert!(matches!(query.normalized(), Err(AppError::Forbidden(_))));
        }
        for query in [
            CatalogQuery {
                min_width: Some(0),
                ..Default::default()
            },
            CatalogQuery {
                min_duration: Some(f64::NAN),
                ..Default::default()
            },
            CatalogQuery {
                min_duration: Some(10.0),
                max_duration: Some(1.0),
                ..Default::default()
            },
            CatalogQuery {
                sort: Some("folder_asc".into()),
                ..Default::default()
            },
            CatalogQuery {
                rating: Some("invalid".into()),
                ..Default::default()
            },
        ] {
            assert!(matches!(query.normalized(), Err(AppError::BadRequest(_))));
        }
    }

    #[test]
    fn filtered_server_page_preserves_authoritative_total_and_rating_facets() {
        let page = catalog_page("source","https://hub.example",json!({
            "items":[{"remote_post_id":"7","canonical_url":"/post/7","rating":"pg","duration":90.5,"variants":[{"kind":"original","asset_url":"/api/client/v1/media/7/original","mime_type":"video/mp4"}]}],
            "total":61,"has_more":true,"available_ratings":["pg","pg13"]
        }),1,60).unwrap();
        assert_eq!(page.items.len(), 1);
        assert_eq!(page.total, Some(61));
        assert_eq!(page.items[0].duration, Some(90.5));
        assert_eq!(page.next_cursor.as_deref(), Some("2"));
        assert_eq!(page.available_ratings, vec!["pg", "pg13"]);
        assert_eq!(
            page.items[0].media[0].mime_type.as_deref(),
            Some("video/mp4")
        );
    }
}
