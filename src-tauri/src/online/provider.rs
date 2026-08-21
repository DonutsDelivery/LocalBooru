use super::models::{RemoteItem, RemoteMediaVariant, RemotePage, SourceCapabilities};
use crate::server::error::AppError;
use serde_json::Value;
pub fn capabilities(provider: &str) -> SourceCapabilities {
    let mut c = SourceCapabilities::read_only();
    if provider == "fake" || provider == "donutbooru" {
        c.authenticate = true;
        c.upload = true;
        c.edit_metadata = provider == "donutbooru";
        c.delete_remote = provider == "donutbooru";
        c.moderation_state = true;
    }
    c
}
pub fn list_path(
    provider: &str,
    query: &str,
    page: u32,
    per_page: u32,
) -> Result<String, AppError> {
    let q = query.replace(' ', "+");
    match provider {
        "donutbooru" => Ok(format!(
            "/images?page={page}&per_page={per_page}&sort_by=date&sort_order=desc&tags={q}"
        )),
        "danbooru" => Ok(format!("/posts.json?page={page}&limit={per_page}&tags={q}")),
        _ => Err(AppError::BadRequest("Unsupported provider family".into())),
    }
}
pub fn normalize_page(
    provider: &str,
    source_id: &str,
    base: &str,
    value: Value,
    page: u32,
    per_page: u32,
) -> Result<RemotePage, AppError> {
    let (rows, total) = match provider {
        "donutbooru" => (
            value
                .get("images")
                .and_then(Value::as_array)
                .cloned()
                .ok_or_else(|| {
                    AppError::ServiceUnavailable("Invalid DonutBooru response".into())
                })?,
            value.get("total").and_then(Value::as_u64),
        ),
        "danbooru" => (
            value
                .as_array()
                .cloned()
                .ok_or_else(|| AppError::ServiceUnavailable("Invalid Danbooru response".into()))?,
            None,
        ),
        _ => return Err(AppError::BadRequest("Unsupported provider family".into())),
    };
    let items = rows
        .iter()
        .filter_map(|r| normalize_item(provider, source_id, base, r))
        .collect();
    Ok(RemotePage {
        items,
        page,
        per_page,
        total,
        stale: false,
        next_cursor: None,
    })
}
fn tags(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::String(s)) => s.split_whitespace().map(str::to_string).collect(),
        Some(Value::Array(a)) => a
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        _ => vec![],
    }
}
fn abs(base: &str, v: Option<&str>) -> Option<String> {
    v.map(|s| {
        if s.starts_with("http://") || s.starts_with("https://") {
            s.into()
        } else {
            format!("{base}{}{}", if s.starts_with('/') { "" } else { "/" }, s)
        }
    })
}
fn normalize_item(provider: &str, source_id: &str, base: &str, r: &Value) -> Option<RemoteItem> {
    let id = r.get("id")?.to_string().trim_matches('"').to_string();
    let (
        canonical,
        original,
        sample,
        preview,
        item_tags,
        rating,
        width,
        height,
        duration,
        source_url,
    ) = match provider {
        "donutbooru" => (
            format!("{base}/images/{id}"),
            abs(base, r.get("url").and_then(Value::as_str)),
            abs(base, r.get("sample_url").and_then(Value::as_str)),
            abs(base, r.get("thumbnail_url").and_then(Value::as_str)),
            tags(r.get("tags")),
            r.get("rating").and_then(Value::as_str).map(str::to_string),
            r.get("width").and_then(Value::as_u64).map(|v| v as u32),
            r.get("height").and_then(Value::as_u64).map(|v| v as u32),
            r.get("duration").and_then(Value::as_f64),
            r.get("source_url")
                .and_then(Value::as_str)
                .map(str::to_string),
        ),
        "danbooru" => (
            format!("{base}/posts/{id}"),
            abs(base, r.get("file_url").and_then(Value::as_str)),
            abs(base, r.get("large_file_url").and_then(Value::as_str)),
            abs(base, r.get("preview_file_url").and_then(Value::as_str)),
            tags(r.get("tag_string")),
            r.get("rating").and_then(Value::as_str).map(str::to_string),
            r.get("image_width")
                .and_then(Value::as_u64)
                .map(|v| v as u32),
            r.get("image_height")
                .and_then(Value::as_u64)
                .map(|v| v as u32),
            r.get("duration").and_then(Value::as_f64),
            r.get("source").and_then(Value::as_str).map(str::to_string),
        ),
        _ => return None,
    };
    let media = [
        ("original", original),
        ("sample", sample),
        ("thumbnail", preview),
    ]
    .into_iter()
    .filter_map(|(kind, url)| {
        url.map(|asset_url| RemoteMediaVariant {
            kind: kind.into(),
            asset_url,
            mime_type: None,
            width,
            height,
        })
    })
    .collect();
    Some(RemoteItem {
        source_id: source_id.into(),
        remote_post_id: id,
        canonical_url: canonical,
        remote_revision: None,
        title: None,
        tags: item_tags,
        rating,
        source_url,
        width,
        height,
        duration,
        media,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn two_dialects() {
        let a = normalize_page(
            "donutbooru",
            "a",
            "https://a.test",
            serde_json::json!({"images":[{"id":1,"url":"/a.jpg","tags":["cat"]}],"total":1}),
            1,
            20,
        )
        .unwrap();
        let b=normalize_page("danbooru","b","https://b.test",serde_json::json!([{"id":2,"file_url":"https://cdn.test/b.jpg","tag_string":"dog safe"}]),1,20).unwrap();
        assert_eq!(a.items[0].remote_post_id, "1");
        assert_eq!(b.items[0].tags, vec!["dog", "safe"]);
    }
}
