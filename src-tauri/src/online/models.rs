use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceCapabilities {
    pub browse: bool,
    pub search: bool,
    pub fetch_detail: bool,
    pub fetch_original: bool,
    pub authenticate: bool,
    pub upload: bool,
    pub edit_metadata: bool,
    pub delete_remote: bool,
    pub favorite: bool,
    pub moderation_state: bool,
    pub byte_ranges: bool,
}
impl SourceCapabilities {
    pub fn read_only() -> Self {
        Self {
            browse: true,
            search: true,
            fetch_detail: true,
            fetch_original: true,
            authenticate: false,
            upload: false,
            edit_metadata: false,
            delete_remote: false,
            favorite: false,
            moderation_state: false,
            byte_ranges: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RemoteSource {
    pub source_id: String,
    pub provider_family: String,
    pub display_name: String,
    pub normalized_base_url: String,
    pub policy_profile: String,
    pub enabled: bool,
    pub allow_local_network: bool,
    pub capabilities: SourceCapabilities,
    pub last_probe_at: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RemoteMediaVariant {
    pub kind: String,
    pub asset_url: String,
    pub mime_type: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RemoteItem {
    pub source_id: String,
    pub remote_post_id: String,
    pub canonical_url: String,
    pub remote_revision: Option<String>,
    pub title: Option<String>,
    pub tags: Vec<String>,
    pub rating: Option<String>,
    pub source_url: Option<String>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub duration: Option<f64>,
    pub media: Vec<RemoteMediaVariant>,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RemotePage {
    pub items: Vec<RemoteItem>,
    pub page: u32,
    pub per_page: u32,
    pub total: Option<u64>,
    pub stale: bool,
    pub next_cursor: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublicationSnapshot {
    pub library_id: String,
    pub directory_id: i64,
    pub image_id: i64,
    pub content_sha256: Option<String>,
    pub caption: Option<String>,
    pub alt_text: Option<String>,
    pub content_warning: Option<String>,
    pub tags: Vec<String>,
    pub rating: Option<String>,
}
pub const PUBLICATION_STATES: &[&str] = &[
    "queued",
    "uploading",
    "submitted",
    "pending_moderation",
    "published",
    "rejected",
    "failed_retryable",
    "failed_terminal",
    "retract_requested",
    "retracted",
    "remote_missing",
];

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn remote_item_has_no_local_identity_fields() {
        let item = RemoteItem {
            source_id: "s".into(),
            remote_post_id: "42".into(),
            canonical_url: "https://example.test/posts/42".into(),
            remote_revision: None,
            title: None,
            tags: vec![],
            rating: None,
            source_url: None,
            width: None,
            height: None,
            duration: None,
            media: vec![],
        };
        let v = serde_json::to_value(item).unwrap();
        for field in [
            "directory_id",
            "library_id",
            "file_path",
            "image_id",
            "file_hash",
        ] {
            assert!(v.get(field).is_none());
        }
    }
    #[test]
    fn writes_default_disabled() {
        let c = SourceCapabilities::read_only();
        assert!(c.browse && c.search);
        assert!(!c.upload && !c.delete_remote && !c.edit_metadata);
    }
}
