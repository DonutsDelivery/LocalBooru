use serde::{Deserialize, Serialize};

// ─── Enums ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Rating {
    Pg,
    Pg13,
    R,
    X,
    Xxx,
}

impl Rating {
    pub fn as_str(&self) -> &'static str {
        match self {
            Rating::Pg => "pg",
            Rating::Pg13 => "pg13",
            Rating::R => "r",
            Rating::X => "x",
            Rating::Xxx => "xxx",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "pg" => Some(Rating::Pg),
            "pg13" => Some(Rating::Pg13),
            "r" => Some(Rating::R),
            "x" => Some(Rating::X),
            "xxx" => Some(Rating::Xxx),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TagCategory {
    General,
    Character,
    Copyright,
    Artist,
    Meta,
}

impl TagCategory {
    pub fn as_str(&self) -> &'static str {
        match self {
            TagCategory::General => "general",
            TagCategory::Character => "character",
            TagCategory::Copyright => "copyright",
            TagCategory::Artist => "artist",
            TagCategory::Meta => "meta",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "general" => Some(TagCategory::General),
            "character" => Some(TagCategory::Character),
            "copyright" => Some(TagCategory::Copyright),
            "artist" => Some(TagCategory::Artist),
            "meta" => Some(TagCategory::Meta),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum TaskStatus {
    Pending,
    Processing,
    Completed,
    Failed,
    Cancelled,
}

impl TaskStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskStatus::Pending => "pending",
            TaskStatus::Processing => "processing",
            TaskStatus::Completed => "completed",
            TaskStatus::Failed => "failed",
            TaskStatus::Cancelled => "cancelled",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "pending" => Some(TaskStatus::Pending),
            "processing" => Some(TaskStatus::Processing),
            "completed" => Some(TaskStatus::Completed),
            "failed" => Some(TaskStatus::Failed),
            "cancelled" => Some(TaskStatus::Cancelled),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum TaskType {
    Tag,
    ScanDirectory,
    VerifyFiles,
    Upload,
    AgeDetect,
    ExtractMetadata,
}

impl TaskType {
    pub fn as_str(&self) -> &'static str {
        match self {
            TaskType::Tag => "tag",
            TaskType::ScanDirectory => "scan_directory",
            TaskType::VerifyFiles => "verify_files",
            TaskType::Upload => "upload",
            TaskType::AgeDetect => "age_detect",
            TaskType::ExtractMetadata => "extract_metadata",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "tag" => Some(TaskType::Tag),
            "scan_directory" => Some(TaskType::ScanDirectory),
            "verify_files" => Some(TaskType::VerifyFiles),
            "upload" => Some(TaskType::Upload),
            "age_detect" => Some(TaskType::AgeDetect),
            "extract_metadata" => Some(TaskType::ExtractMetadata),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum FileStatus {
    Available,
    Missing,
    DriveOffline,
    Unknown,
}

impl FileStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            FileStatus::Available => "available",
            FileStatus::Missing => "missing",
            FileStatus::DriveOffline => "drive_offline",
            FileStatus::Unknown => "unknown",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "available" => Some(FileStatus::Available),
            "missing" => Some(FileStatus::Missing),
            "drive_offline" => Some(FileStatus::DriveOffline),
            "unknown" => Some(FileStatus::Unknown),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum AccessLevel {
    Localhost,
    LocalNetwork,
    Public,
}

impl AccessLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            AccessLevel::Localhost => "localhost",
            AccessLevel::LocalNetwork => "local_network",
            AccessLevel::Public => "public",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "localhost" => Some(AccessLevel::Localhost),
            "local_network" => Some(AccessLevel::LocalNetwork),
            "public" => Some(AccessLevel::Public),
            _ => None,
        }
    }
}

// ─── Row structs ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Image {
    pub id: i64,
    pub filename: String,
    pub original_filename: Option<String>,
    pub file_hash: String,
    pub perceptual_hash: Option<String>,
    pub width: Option<i32>,
    pub height: Option<i32>,
    pub file_size: Option<i64>,
    pub duration: Option<f64>,
    pub rating: String,
    pub prompt: Option<String>,
    pub negative_prompt: Option<String>,
    pub model_name: Option<String>,
    pub sampler: Option<String>,
    pub seed: Option<String>,
    pub steps: Option<i32>,
    pub cfg_scale: Option<f64>,
    pub source_url: Option<String>,
    pub num_faces: Option<i32>,
    pub min_detected_age: Option<i32>,
    pub max_detected_age: Option<i32>,
    pub detected_ages: Option<String>,
    pub age_detection_data: Option<String>,
    pub is_favorite: bool,
    pub import_source: Option<String>,
    pub view_count: i32,
    pub created_at: String,
    pub updated_at: Option<String>,
    pub file_created_at: Option<String>,
    pub file_modified_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageFile {
    pub id: i64,
    pub image_id: i64,
    pub original_path: String,
    pub file_exists: bool,
    pub file_status: String,
    pub last_verified_at: Option<String>,
    pub watch_directory_id: Option<i64>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tag {
    pub id: i64,
    pub name: String,
    pub category: String,
    pub post_count: i32,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TagAlias {
    pub alias: String,
    pub target_tag_id: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchDirectory {
    pub id: i64,
    pub path: String,
    pub name: Option<String>,
    pub enabled: bool,
    pub recursive: bool,
    pub auto_tag: bool,
    pub auto_age_detect: bool,
    pub last_scanned_at: Option<String>,
    pub created_at: String,
    pub comfyui_prompt_node_ids: Option<String>,
    pub comfyui_negative_node_ids: Option<String>,
    pub metadata_format: String,
    pub parent_path: Option<String>,
    pub public_access: bool,
    pub show_images: bool,
    pub show_videos: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskQueueItem {
    pub id: i64,
    pub task_type: String,
    pub payload: Option<String>,
    pub status: String,
    pub priority: i32,
    pub attempts: i32,
    pub error_message: Option<String>,
    pub created_at: String,
    pub started_at: Option<String>,
    pub completed_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Collection {
    pub id: i64,
    pub name: String,
    pub description: Option<String>,
    pub cover_image_id: Option<i64>,
    pub item_count: i32,
    pub created_at: String,
    pub updated_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionItem {
    pub id: i64,
    pub collection_id: i64,
    pub image_id: i64,
    pub sort_order: i32,
    pub added_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WatchHistoryEntry {
    pub id: i64,
    pub image_id: i64,
    pub playback_position: f64,
    pub duration: f64,
    pub completed: bool,
    pub last_watched: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub id: i64,
    pub username: String,
    pub password_hash: String,
    pub is_active: bool,
    pub access_level: String,
    pub can_write: bool,
    pub created_at: String,
    pub last_login: Option<String>,
}

/// Image file record from a per-directory database.
/// Unlike [`ImageFile`], directory DBs do not have a `watch_directory_id` column
/// since the directory is implicit from the database file itself.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectoryImageFile {
    pub id: i64,
    pub image_id: i64,
    pub original_path: String,
    pub file_exists: bool,
    pub file_status: String,
    pub last_verified_at: Option<String>,
    pub created_at: String,
}
