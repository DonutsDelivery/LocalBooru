"""
LocalBooru database models - simplified single-user version of DonutBooru

Architecture:
- Main database (library.db): Global models using `Base`
  - WatchDirectory, Tag, TagAlias, TaskQueue, Settings, User, BooruInstance, ExternalUpload
- Per-directory databases (directories/{id}.db): Directory-local models using `DirectoryBase`
  - DirectoryImage, DirectoryImageFile, directory_image_tags
"""
from sqlalchemy import Column, Integer, String, Text, Float, Boolean, DateTime, ForeignKey, Table, Enum, UniqueConstraint
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy.sql import func
from .database import Base
import enum


# =============================================================================
# Directory-specific database base
# =============================================================================
# Models using DirectoryBase are stored in per-directory databases
DirectoryBase = declarative_base()


class Rating(str, enum.Enum):
    """5-tier rating system"""
    pg = "pg"           # Child-friendly, fully clothed
    pg13 = "pg13"       # Dresses, sportswear, shows skin
    r = "r"             # Cleavage, underwear, swimsuits
    x = "x"             # Nudity, visible nipples/genitals
    xxx = "xxx"         # Explicit sexual content


class TagCategory(str, enum.Enum):
    general = "general"
    character = "character"
    copyright = "copyright"
    artist = "artist"
    meta = "meta"


class TaggerModel(str, enum.Enum):
    vit_v3 = "vit-v3"              # Fastest, good quality
    eva02_large_v3 = "eva02-large-v3"  # Slowest, most tags
    swinv2_v3 = "swinv2-v3"        # Medium speed


class TaskStatus(str, enum.Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class TaskType(str, enum.Enum):
    tag = "tag"
    scan_directory = "scan_directory"
    verify_files = "verify_files"
    upload = "upload"
    age_detect = "age_detect"
    extract_metadata = "extract_metadata"  # Extract AI generation metadata


class UploadStatus(str, enum.Enum):
    pending = "pending"
    uploaded = "uploaded"
    failed = "failed"
    deleted = "deleted"


class AccessLevel(str, enum.Enum):
    """Network access levels for authorization"""
    localhost = "localhost"        # Only from 127.0.0.1 / ::1
    local_network = "local_network"  # LAN (192.168.x.x, 10.x.x.x, 172.16-31.x.x)
    public = "public"              # Internet / all IPs


# Association table for image-tag many-to-many relationship
image_tags = Table(
    "image_tags",
    Base.metadata,
    Column("image_id", Integer, ForeignKey("images.id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id", Integer, ForeignKey("tags.id", ondelete="CASCADE"), primary_key=True),
    Column("confidence", Float, nullable=True),  # AI confidence score
    Column("is_manual", Boolean, default=False)  # User-added vs AI-generated
)


class Image(Base):
    """Core image model - simplified for single user local storage"""
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)  # Stored filename (hash-based)
    original_filename = Column(String(255), nullable=True)  # Original upload name
    file_hash = Column(String(64), unique=True, nullable=False, index=True)  # SHA256
    perceptual_hash = Column(String(16), nullable=True, index=True)  # pHash for visual duplicate detection
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    file_size = Column(Integer, nullable=True)
    duration = Column(Float, nullable=True)  # Video duration in seconds
    rating = Column(Enum(Rating), default=Rating.pg, index=True)

    # AI generation metadata
    prompt = Column(Text, nullable=True)
    negative_prompt = Column(Text, nullable=True)
    model_name = Column(String(255), nullable=True)
    sampler = Column(String(100), nullable=True)
    seed = Column(String(50), nullable=True)
    steps = Column(Integer, nullable=True)
    cfg_scale = Column(Float, nullable=True)

    # Source URL (if from Civitai, etc.)
    source_url = Column(Text, nullable=True)

    # Age detection results (for realistic/photorealistic images)
    num_faces = Column(Integer, nullable=True)  # Number of detected faces
    min_detected_age = Column(Integer, nullable=True, index=True)  # Youngest face
    max_detected_age = Column(Integer, nullable=True, index=True)  # Oldest face
    detected_ages = Column(Text, nullable=True)  # JSON: [25, 32, 8]
    age_detection_data = Column(Text, nullable=True)  # Full JSON with all face data

    # Local library features
    is_favorite = Column(Boolean, default=False, index=True)
    import_source = Column(String(500), nullable=True)  # Directory path or "manual"

    # View tracking
    view_count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    file_created_at = Column(DateTime(timezone=True), nullable=True, index=True)  # File creation time
    file_modified_at = Column(DateTime(timezone=True), nullable=True, index=True)  # File modification time

    # Relationships
    tags = relationship("Tag", secondary=image_tags, back_populates="images")
    files = relationship("ImageFile", back_populates="image", cascade="all, delete-orphan")
    external_uploads = relationship("ExternalUpload", back_populates="image", cascade="all, delete-orphan")

    @property
    def url(self):
        """Get URL to serve the image"""
        # For reference-based storage, we serve from original path via API
        return f"/api/images/{self.id}/file"

    @property
    def thumbnail_url(self):
        """Get thumbnail URL"""
        return f"/api/images/{self.id}/thumbnail"


class FileStatus(str, enum.Enum):
    """File availability status"""
    available = "available"      # File exists and is accessible
    missing = "missing"          # File was deleted/moved (confirmed)
    drive_offline = "drive_offline"  # Parent drive/directory is unavailable
    unknown = "unknown"          # Not yet verified


class ImageFile(Base):
    """Track original file locations (reference, not copy)"""
    __tablename__ = "image_files"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id", ondelete="CASCADE"), nullable=False)
    original_path = Column(Text, nullable=False, unique=True, index=True)  # Absolute path to original file
    file_exists = Column(Boolean, default=True, index=True)  # Legacy - kept for compatibility
    file_status = Column(Enum(FileStatus), default=FileStatus.available, index=True)
    last_verified_at = Column(DateTime(timezone=True), nullable=True)
    watch_directory_id = Column(Integer, ForeignKey("watch_directories.id", ondelete="SET NULL"), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    image = relationship("Image", back_populates="files")
    watch_directory = relationship("WatchDirectory", back_populates="image_files")


class Tag(Base):
    __tablename__ = "tags"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    category = Column(Enum(TagCategory), default=TagCategory.general, index=True)
    post_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    images = relationship("Image", secondary=image_tags, back_populates="tags")
    aliases = relationship("TagAlias", back_populates="target_tag")


class TagAlias(Base):
    __tablename__ = "tag_aliases"

    alias = Column(String(100), primary_key=True)
    target_tag_id = Column(Integer, ForeignKey("tags.id"), nullable=False)

    # Relationships
    target_tag = relationship("Tag", back_populates="aliases")


class WatchDirectory(Base):
    """Directories to watch for new images"""
    __tablename__ = "watch_directories"

    id = Column(Integer, primary_key=True, index=True)
    path = Column(Text, nullable=False, unique=True)
    name = Column(String(255), nullable=True)  # User-friendly name
    enabled = Column(Boolean, default=True)
    recursive = Column(Boolean, default=True)  # Scan subdirectories
    auto_tag = Column(Boolean, default=True)  # Auto-tag imported images
    auto_age_detect = Column(Boolean, default=False)  # Auto-detect ages on import
    last_scanned_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # ComfyUI metadata configuration
    comfyui_prompt_node_ids = Column(Text, nullable=True)  # JSON: ["272", "15"]
    comfyui_negative_node_ids = Column(Text, nullable=True)  # JSON: ["273"]
    metadata_format = Column(String(50), default="auto")  # auto, a1111, comfyui, none

    # Network access control
    public_access = Column(Boolean, default=False)  # Allow public network access to this directory

    # Media type filtering
    show_images = Column(Boolean, default=True)  # Show image files in this directory
    show_videos = Column(Boolean, default=True)  # Show video files in this directory

    # Relationships
    image_files = relationship("ImageFile", back_populates="watch_directory")


class BooruInstance(Base):
    """Federated booru instances for uploads"""
    __tablename__ = "booru_instances"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    base_url = Column(String(500), nullable=False)  # e.g., "https://donutbooru.com"
    instance_type = Column(String(50), default="donutbooru")  # donutbooru, danbooru, gelbooru, etc.
    auth_method = Column(String(50), default="discord")  # discord, api_key, none
    auth_token = Column(Text, nullable=True)  # Encrypted token/API key
    is_enabled = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    uploads = relationship("ExternalUpload", back_populates="booru")


class ExternalUpload(Base):
    """Track uploads to external boorus"""
    __tablename__ = "external_uploads"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id", ondelete="CASCADE"), nullable=False)
    booru_id = Column(Integer, ForeignKey("booru_instances.id", ondelete="CASCADE"), nullable=False)
    external_id = Column(String(100), nullable=True)  # ID on remote booru
    external_url = Column(Text, nullable=True)  # URL on external booru
    uploaded_at = Column(DateTime(timezone=True), nullable=True)
    status = Column(Enum(UploadStatus), default=UploadStatus.pending)
    error_message = Column(Text, nullable=True)

    # Relationships
    image = relationship("Image", back_populates="external_uploads")
    booru = relationship("BooruInstance", back_populates="uploads")


class TaskQueue(Base):
    """Background task queue"""
    __tablename__ = "task_queue"

    id = Column(Integer, primary_key=True, index=True)
    task_type = Column(Enum(TaskType), nullable=False, index=True)
    payload = Column(Text, nullable=True)  # JSON payload
    status = Column(Enum(TaskStatus), default=TaskStatus.pending, index=True)
    priority = Column(Integer, default=0)  # Higher = more urgent
    attempts = Column(Integer, default=0)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)


class Collection(Base):
    """User-created groupings of images"""
    __tablename__ = "collections"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    cover_image_id = Column(Integer, ForeignKey("images.id", ondelete="SET NULL"), nullable=True)
    item_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    cover_image = relationship("Image", foreign_keys=[cover_image_id])
    items = relationship("CollectionItem", back_populates="collection", cascade="all, delete-orphan")


class CollectionItem(Base):
    """Items within a collection with ordering"""
    __tablename__ = "collection_items"

    id = Column(Integer, primary_key=True, index=True)
    collection_id = Column(Integer, ForeignKey("collections.id", ondelete="CASCADE"), nullable=False, index=True)
    image_id = Column(Integer, ForeignKey("images.id", ondelete="CASCADE"), nullable=False)
    sort_order = Column(Integer, default=0)
    added_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    collection = relationship("Collection", back_populates="items")
    image = relationship("Image")

    __table_args__ = (
        UniqueConstraint('collection_id', 'image_id', name='uq_collection_image'),
    )


class WatchHistory(Base):
    """Track video watch progress for resume functionality"""
    __tablename__ = "watch_history"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id", ondelete="CASCADE"), nullable=False, unique=True, index=True)
    playback_position = Column(Float, default=0.0)  # Seconds into video
    duration = Column(Float, default=0.0)  # Total video duration
    completed = Column(Boolean, default=False, index=True)
    last_watched = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    image = relationship("Image")


class Settings(Base):
    """App settings stored in database"""
    __tablename__ = "settings"

    key = Column(String(100), primary_key=True)
    value = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class User(Base):
    """User accounts for network access authentication"""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)  # Format: salt:hash (PBKDF2)
    is_active = Column(Boolean, default=True)
    access_level = Column(Enum(AccessLevel), default=AccessLevel.local_network)  # Minimum level user can access from
    can_write = Column(Boolean, default=False)  # Override read-only restriction for remote access
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)


# =============================================================================
# Per-Directory Database Models
# =============================================================================
# These models are stored in per-directory databases (directories/{id}.db)
# They use DirectoryBase instead of Base

# Association table for image-tag many-to-many in directory databases
# Note: tag_id references global tags in main DB (not enforced via FK since cross-DB)
directory_image_tags = Table(
    "image_tags",
    DirectoryBase.metadata,
    Column("image_id", Integer, ForeignKey("images.id", ondelete="CASCADE"), primary_key=True),
    Column("tag_id", Integer, primary_key=True),  # References Tag.id in main DB
    Column("confidence", Float, nullable=True),  # AI confidence score
    Column("is_manual", Boolean, default=False)  # User-added vs AI-generated
)


class DirectoryImage(DirectoryBase):
    """
    Image model for per-directory databases.

    Same fields as the main Image model but stored in directory-specific DBs.
    """
    __tablename__ = "images"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)  # Stored filename (hash-based)
    original_filename = Column(String(255), nullable=True)  # Original upload name
    file_hash = Column(String(64), unique=True, nullable=False, index=True)  # SHA256/xxhash
    perceptual_hash = Column(String(16), nullable=True, index=True)  # pHash for visual duplicate detection
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    file_size = Column(Integer, nullable=True)
    duration = Column(Float, nullable=True)  # Video duration in seconds
    rating = Column(Enum(Rating), default=Rating.pg, index=True)

    # AI generation metadata
    prompt = Column(Text, nullable=True)
    negative_prompt = Column(Text, nullable=True)
    model_name = Column(String(255), nullable=True)
    sampler = Column(String(100), nullable=True)
    seed = Column(String(50), nullable=True)
    steps = Column(Integer, nullable=True)
    cfg_scale = Column(Float, nullable=True)

    # Source URL (if from Civitai, etc.)
    source_url = Column(Text, nullable=True)

    # Age detection results
    num_faces = Column(Integer, nullable=True)
    min_detected_age = Column(Integer, nullable=True, index=True)
    max_detected_age = Column(Integer, nullable=True, index=True)
    detected_ages = Column(Text, nullable=True)  # JSON: [25, 32, 8]
    age_detection_data = Column(Text, nullable=True)

    # Local library features
    is_favorite = Column(Boolean, default=False, index=True)
    import_source = Column(String(500), nullable=True)

    # View tracking
    view_count = Column(Integer, default=0)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    file_created_at = Column(DateTime(timezone=True), nullable=True, index=True)
    file_modified_at = Column(DateTime(timezone=True), nullable=True, index=True)

    # Relationships within directory DB
    files = relationship("DirectoryImageFile", back_populates="image", cascade="all, delete-orphan")

    @property
    def url(self):
        """Get URL to serve the image"""
        return f"/api/images/{self.id}/file"

    @property
    def thumbnail_url(self):
        """Get thumbnail URL"""
        return f"/api/images/{self.id}/thumbnail"


class DirectoryImageFile(DirectoryBase):
    """
    Track original file locations for images in a directory database.

    Note: No watch_directory_id since each DB is already per-directory.
    """
    __tablename__ = "image_files"

    id = Column(Integer, primary_key=True, index=True)
    image_id = Column(Integer, ForeignKey("images.id", ondelete="CASCADE"), nullable=False)
    original_path = Column(Text, nullable=False, unique=True, index=True)
    file_exists = Column(Boolean, default=True, index=True)
    file_status = Column(Enum(FileStatus), default=FileStatus.available, index=True)
    last_verified_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationship
    image = relationship("DirectoryImage", back_populates="files")
