use std::collections::{BTreeMap, BTreeSet};
use std::io::{Read, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use futures_util::TryStreamExt;
use reqwest::Url;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::lada::{
    self, LadaBackend, LadaBackendCompatibility, LadaBackendPreference, LadaDeployment,
    LadaReadiness, LadaReadinessStatus, LADA_ADDON_VERSION, LADA_DEPLOYMENT_FILE,
    LADA_MODEL_REVISION, LADA_PROBE_TIMEOUT, LADA_PROTOCOL_VERSION, LADA_UPSTREAM_REVISION,
};

pub const LADA_LICENSE: &str = "AGPL-3.0-only";
pub const LADA_SOURCE_URL: &str =
    "https://github.com/DonutsDelivery/localbooru-lada-addon/tree/v0.1.0";
pub const LADA_RELEASE_MANIFEST_URL: &str = "https://github.com/DonutsDelivery/localbooru-lada-addon/releases/download/v0.1.0/release-manifest.json";
const LADA_RELEASE_BASE_URL: &str =
    "https://github.com/DonutsDelivery/localbooru-lada-addon/releases/download/v0.1.0";
pub const LADA_LOCAL_MANIFEST_ENV: &str = "LOCALBOORU_LADA_RELEASE_MANIFEST";
const COMMON_PACKAGE: &str = "linux_x86_64_common";
const MODEL_PACKAGE: &str = "model_bundle";
const MAX_ARTIFACT_BYTES: u64 = 20 * 1024 * 1024 * 1024;
const MAX_INSTALLED_BYTES: u64 = 30 * 1024 * 1024 * 1024;
const DOWNLOAD_CHUNK_BYTES: usize = 1024 * 1024;
const MAX_MANIFEST_BYTES: u64 = 2 * 1024 * 1024;
const CANCEL_POLL_INTERVAL: Duration = Duration::from_millis(500);
const MAX_ARCHIVE_ENTRIES: usize = 1_000_000;
const LADA_MODEL_PROBE_TIMEOUT_SECS: f64 = 90.0;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct LadaArtifact {
    pub url: String,
    pub sha256: String,
    pub size: u64,
    pub installed_size: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct LadaReleaseIdentity {
    pub repository: String,
    pub revision: String,
    pub license: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct LadaModel {
    pub name: String,
    pub role: String,
    pub variant: String,
    #[serde(default)]
    pub default: bool,
    pub size: u64,
    pub sha256: String,
    pub source_url: String,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct LadaReleaseManifest {
    pub schema_version: u32,
    pub addon_id: String,
    pub version: String,
    pub protocol_version: u32,
    pub license: String,
    pub source_url: String,
    pub upstream: LadaReleaseIdentity,
    pub model_repository: LadaReleaseIdentity,
    pub models: Vec<LadaModel>,
    pub backend_compatibility: LadaBackendCompatibility,
    pub packages: BTreeMap<String, LadaArtifact>,
    pub corresponding_source: LadaArtifact,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LadaInstallStage {
    Resolving,
    Downloading,
    Installing,
    Validating,
    Probing,
    Activating,
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
pub struct LadaInstallProgress {
    pub stage: LadaInstallStage,
    pub completed_bytes: u64,
    pub total_bytes: u64,
    pub package: Option<String>,
}

#[derive(Clone, Debug)]
enum ManifestSource {
    Remote(Url),
    Local(PathBuf),
}

pub struct LadaInstallOutcome {
    pub readiness: LadaReadiness,
    pub deployment: LadaDeployment,
}

fn expected_models() -> BTreeMap<&'static str, (&'static str, u64, &'static str, &'static str, bool)>
{
    BTreeMap::from([
        (
            "lada_mosaic_detection_model_v2.pt",
            (
                "056756fcab250bcdf0833e75aac33e2197b8809b0ab8c16e14722dcec94269b5",
                45_153_839,
                "detection",
                "v2",
                false,
            ),
        ),
        (
            "lada_mosaic_detection_model_v4_accurate.pt",
            (
                "c244d7e49d8f88e264b8dc15f91fb21f5908ad8fb6f300b7bc88462d0801bc1f",
                45_136_630,
                "detection",
                "v4-accurate",
                false,
            ),
        ),
        (
            "lada_mosaic_detection_model_v4_fast.pt",
            (
                "9a6b660d1d3e3797d39515e08b0e72fcc59815f38279faa7a4ab374ab2c1e3b4",
                5_981_796,
                "detection",
                "v4-fast",
                true,
            ),
        ),
        (
            "lada_mosaic_restoration_model_generic_v1.2.pth",
            (
                "d404152576ce64fb5b2f315c03062709dac4f5f8548934866cd01c823c8104ee",
                78_441_770,
                "restoration",
                "basicvsrpp-v1.2",
                true,
            ),
        ),
    ])
}

fn validate_sha256(value: &str, label: &str) -> Result<(), String> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("{} has an invalid SHA-256 digest", label));
    }
    Ok(())
}

fn validate_https(value: &str, label: &str) -> Result<Url, String> {
    let url = Url::parse(value).map_err(|error| format!("Invalid {} URL: {}", label, error))?;
    if url.scheme() != "https" || url.host_str().is_none() {
        return Err(format!("{} URL must use HTTPS", label));
    }
    Ok(url)
}

pub fn validate_release_manifest(manifest: &LadaReleaseManifest) -> Result<(), String> {
    if manifest.schema_version != 1
        || manifest.addon_id != "lada"
        || manifest.version != LADA_ADDON_VERSION
        || manifest.protocol_version != LADA_PROTOCOL_VERSION
        || manifest.license != LADA_LICENSE
        || manifest.source_url != LADA_SOURCE_URL
        || manifest.upstream.repository != "https://github.com/ladaapp/lada"
        || manifest.upstream.revision != LADA_UPSTREAM_REVISION
        || manifest.upstream.license != LADA_LICENSE
        || manifest.model_repository.repository != "https://huggingface.co/ladaapp/lada"
        || manifest.model_repository.revision != LADA_MODEL_REVISION
        || manifest.model_repository.license != LADA_LICENSE
    {
        return Err(
            "The LADA release manifest does not match LocalBooru's trusted release identity".into(),
        );
    }
    validate_https(&manifest.source_url, "source")?;
    validate_https(&manifest.upstream.repository, "upstream repository")?;
    validate_https(&manifest.model_repository.repository, "model repository")?;

    let expected = expected_models();
    if manifest.models.len() != expected.len() {
        return Err("The LADA release manifest has an unexpected model set".into());
    }
    let mut names = BTreeSet::new();
    let mut default_detection = 0;
    let mut default_restoration = 0;
    for model in &manifest.models {
        if Path::new(&model.name)
            .file_name()
            .and_then(|name| name.to_str())
            != Some(&model.name)
            || !names.insert(model.name.as_str())
        {
            return Err(
                "The LADA release manifest contains an unsafe or duplicate model name".into(),
            );
        }
        let Some((sha256, size, role, variant, default)) = expected.get(model.name.as_str()) else {
            return Err(format!("Unexpected LADA model: {}", model.name));
        };
        let expected_source = format!(
            "https://huggingface.co/ladaapp/lada/resolve/{}/{}",
            LADA_MODEL_REVISION, model.name
        );
        if model.sha256 != *sha256
            || model.size != *size
            || model.role != *role
            || model.variant != *variant
            || model.default != *default
            || model.source_url != expected_source
        {
            return Err(format!("LADA model identity mismatch: {}", model.name));
        }
        validate_https(&model.source_url, "model source")?;
        if model.default && model.role == "detection" {
            default_detection += 1;
        }
        if model.default && model.role == "restoration" {
            default_restoration += 1;
        }
    }
    if default_detection != 1 || default_restoration != 1 {
        return Err(
            "The LADA manifest must select one default detection and restoration model".into(),
        );
    }

    let cuda_compatible = matches!(
        (
            manifest.backend_compatibility.cuda.variant.as_str(),
            manifest.backend_compatibility.cuda.minimum_driver_major,
        ),
        ("cu128", 570) | ("cu126", 560)
    );
    if manifest.backend_compatibility.cuda.package != "linux_x86_64_cuda"
        || !cuda_compatible
        || manifest.backend_compatibility.xpu.package != "linux_x86_64_xpu"
        || manifest.backend_compatibility.xpu.kernel_drivers != ["i915", "xe"]
        || !manifest.backend_compatibility.xpu.requires_render_node
    {
        return Err("The LADA release manifest has incompatible backend metadata".into());
    }

    let required_packages = BTreeSet::from([
        COMMON_PACKAGE,
        manifest.backend_compatibility.cuda.package.as_str(),
        manifest.backend_compatibility.xpu.package.as_str(),
        MODEL_PACKAGE,
    ]);
    if manifest
        .packages
        .keys()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        != required_packages
    {
        return Err("The LADA release manifest has an unexpected package topology".into());
    }
    let expected_filenames = BTreeMap::from([
        (COMMON_PACKAGE, "linux-x86_64-common.tar.zst"),
        ("linux_x86_64_cuda", "linux-x86_64-cuda.tar.zst"),
        ("linux_x86_64_xpu", "linux-x86_64-xpu.tar.zst"),
        (MODEL_PACKAGE, "models.tar.zst"),
    ]);
    for (name, artifact) in &manifest.packages {
        validate_https(&artifact.url, &format!("{} package", name))?;
        let expected_filename = expected_filenames[name.as_str()];
        if artifact.url != format!("{}/{}", LADA_RELEASE_BASE_URL, expected_filename) {
            return Err(format!("{} has an unexpected package URL", name));
        }
        validate_sha256(&artifact.sha256, name)?;
        if artifact.size == 0 || artifact.size > MAX_ARTIFACT_BYTES {
            return Err(format!("{} has an invalid download size", name));
        }
        let installed = artifact
            .installed_size
            .ok_or_else(|| format!("{} is missing its installed size", name))?;
        if installed == 0 || installed > MAX_INSTALLED_BYTES {
            return Err(format!("{} has an invalid installed size", name));
        }
    }
    validate_https(&manifest.corresponding_source.url, "corresponding source")?;
    if manifest.corresponding_source.url != format!("{}/source.tar.zst", LADA_RELEASE_BASE_URL) {
        return Err("The LADA corresponding source has an unexpected URL".into());
    }
    validate_sha256(
        &manifest.corresponding_source.sha256,
        "corresponding_source",
    )?;
    if manifest.corresponding_source.size == 0
        || manifest.corresponding_source.size > MAX_ARTIFACT_BYTES
    {
        return Err("The LADA corresponding source has an invalid download size".into());
    }
    Ok(())
}

fn manifest_source() -> Result<ManifestSource, String> {
    if let Some(path) = std::env::var_os(LADA_LOCAL_MANIFEST_ENV) {
        let path = PathBuf::from(path);
        if !path.is_absolute() {
            return Err(format!(
                "{} must be an absolute path",
                LADA_LOCAL_MANIFEST_ENV
            ));
        }
        return Ok(ManifestSource::Local(path));
    }
    Ok(ManifestSource::Remote(
        Url::parse(LADA_RELEASE_MANIFEST_URL).expect("pinned LADA manifest URL"),
    ))
}

async fn send_request(
    request: reqwest::RequestBuilder,
    cancellation: &AtomicBool,
    label: &str,
) -> Result<reqwest::Response, String> {
    let request = request.send();
    tokio::pin!(request);
    loop {
        tokio::select! {
            response = &mut request => {
                return response
                    .map_err(|error| format!("Failed to download {}: {}", label, error))?
                    .error_for_status()
                    .map_err(|error| format!("Failed to download {}: {}", label, error));
            }
            _ = tokio::time::sleep(CANCEL_POLL_INTERVAL) => {
                if cancellation.load(Ordering::SeqCst) {
                    return Err("LADA installation was cancelled".into());
                }
            }
        }
    }
}

async fn response_bytes_bounded(
    mut response: reqwest::Response,
    cancellation: &AtomicBool,
    label: &str,
    limit: u64,
) -> Result<Vec<u8>, String> {
    if response.url().scheme() != "https" {
        return Err(format!("{} download redirected away from HTTPS", label));
    }
    if response.content_length().is_some_and(|size| size > limit) {
        return Err(format!("{} exceeded its maximum size", label));
    }
    let mut bytes = Vec::new();
    loop {
        if cancellation.load(Ordering::SeqCst) {
            return Err("LADA installation was cancelled".into());
        }
        let chunk = match tokio::time::timeout(CANCEL_POLL_INTERVAL, response.chunk()).await {
            Ok(result) => result.map_err(|error| format!("Failed to read {}: {}", label, error))?,
            Err(_) => continue,
        };
        let Some(chunk) = chunk else {
            break;
        };
        let new_len = bytes
            .len()
            .checked_add(chunk.len())
            .filter(|length| *length as u64 <= limit)
            .ok_or_else(|| format!("{} exceeded its maximum size", label))?;
        bytes.reserve(new_len - bytes.len());
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

async fn load_manifest(
    client: &reqwest::Client,
    cancellation: &AtomicBool,
) -> Result<(LadaReleaseManifest, ManifestSource), String> {
    let source = manifest_source()?;
    let bytes = match &source {
        ManifestSource::Local(path) => {
            let metadata = tokio::fs::metadata(path).await.map_err(|error| {
                format!("Failed to inspect local LADA release manifest: {}", error)
            })?;
            if metadata.len() > MAX_MANIFEST_BYTES {
                return Err("Local LADA release manifest exceeded its maximum size".into());
            }
            tokio::fs::read(path)
                .await
                .map_err(|error| format!("Failed to read local LADA release manifest: {}", error))?
        }
        ManifestSource::Remote(url) => {
            let response = send_request(
                client.get(url.clone()),
                cancellation,
                "LADA release manifest",
            )
            .await?;
            response_bytes_bounded(
                response,
                cancellation,
                "LADA release manifest",
                MAX_MANIFEST_BYTES,
            )
            .await?
        }
    };
    let manifest = serde_json::from_slice(&bytes)
        .map_err(|error| format!("Failed to parse LADA release manifest: {}", error))?;
    validate_release_manifest(&manifest)?;
    Ok((manifest, source))
}

fn artifact_local_path(source: &Path, artifact: &LadaArtifact) -> Result<PathBuf, String> {
    let directory = source
        .parent()
        .ok_or_else(|| "Local LADA manifest has no parent directory".to_string())?;
    let url = Url::parse(&artifact.url)
        .map_err(|error| format!("Invalid LADA artifact URL: {}", error))?;
    let filename = url
        .path_segments()
        .and_then(|segments| segments.last())
        .filter(|name| !name.is_empty())
        .ok_or_else(|| "LADA artifact URL has no filename".to_string())?;
    if Path::new(filename)
        .file_name()
        .and_then(|name| name.to_str())
        != Some(filename)
    {
        return Err("LADA artifact URL has an unsafe filename".into());
    }
    Ok(directory.join(filename))
}

async fn copy_and_hash<R>(
    mut reader: R,
    destination: &Path,
    artifact: &LadaArtifact,
    cancellation: &AtomicBool,
    mut completed: u64,
    total: u64,
    package: &str,
    progress: &(dyn Fn(LadaInstallProgress) + Send + Sync),
) -> Result<u64, String>
where
    R: tokio::io::AsyncRead + Unpin,
{
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let mut output = tokio::fs::File::create(destination)
        .await
        .map_err(|error| format!("Failed to create LADA package: {}", error))?;
    let mut digest = Sha256::new();
    let mut size = 0u64;
    let mut buffer = vec![0u8; DOWNLOAD_CHUNK_BYTES];
    loop {
        if cancellation.load(Ordering::SeqCst) {
            return Err("LADA installation was cancelled".into());
        }
        let count = match tokio::time::timeout(CANCEL_POLL_INTERVAL, reader.read(&mut buffer)).await
        {
            Ok(result) => {
                result.map_err(|error| format!("Failed to read LADA package: {}", error))?
            }
            Err(_) => continue,
        };
        if count == 0 {
            break;
        }
        size = size
            .checked_add(count as u64)
            .filter(|size| *size <= artifact.size)
            .ok_or_else(|| format!("{} exceeded its declared size", package))?;
        digest.update(&buffer[..count]);
        output
            .write_all(&buffer[..count])
            .await
            .map_err(|error| format!("Failed to write LADA package: {}", error))?;
        progress(LadaInstallProgress {
            stage: LadaInstallStage::Downloading,
            completed_bytes: completed + size,
            total_bytes: total,
            package: Some(package.into()),
        });
    }
    output
        .sync_all()
        .await
        .map_err(|error| format!("Failed to sync LADA package: {}", error))?;
    if size != artifact.size
        || format!("{:x}", digest.finalize()) != artifact.sha256.to_ascii_lowercase()
    {
        return Err(format!("{} failed size or SHA-256 verification", package));
    }
    completed += size;
    Ok(completed)
}

async fn download_artifact(
    client: &reqwest::Client,
    source: &ManifestSource,
    artifact: &LadaArtifact,
    destination: &Path,
    cancellation: &AtomicBool,
    completed: u64,
    total: u64,
    package: &str,
    progress: &(dyn Fn(LadaInstallProgress) + Send + Sync),
) -> Result<u64, String> {
    match source {
        ManifestSource::Local(manifest_path) => {
            let path = artifact_local_path(manifest_path, artifact)?;
            let canonical_parent = manifest_path
                .parent()
                .and_then(|parent| std::fs::canonicalize(parent).ok())
                .ok_or_else(|| "Local LADA bundle directory is unavailable".to_string())?;
            let canonical = std::fs::canonicalize(&path)
                .map_err(|error| format!("Local LADA package is unavailable: {}", error))?;
            if !canonical.starts_with(canonical_parent) {
                return Err("Local LADA package escaped its bundle directory".into());
            }
            let input = tokio::fs::File::open(canonical)
                .await
                .map_err(|error| format!("Failed to open local LADA package: {}", error))?;
            copy_and_hash(
                input,
                destination,
                artifact,
                cancellation,
                completed,
                total,
                package,
                progress,
            )
            .await
        }
        ManifestSource::Remote(_) => {
            let url = validate_https(&artifact.url, "artifact")?;
            let response = send_request(client.get(url), cancellation, package).await?;
            if response.url().scheme() != "https" {
                return Err(format!("{} download redirected away from HTTPS", package));
            }
            if response
                .content_length()
                .is_some_and(|size| size != artifact.size)
            {
                return Err(format!(
                    "{} download size did not match its manifest",
                    package
                ));
            }
            let stream = response.bytes_stream().map_err(std::io::Error::other);
            let reader = tokio_util::io::StreamReader::new(stream);
            copy_and_hash(
                reader,
                destination,
                artifact,
                cancellation,
                completed,
                total,
                package,
                progress,
            )
            .await
        }
    }
}

fn safe_archive_path(path: &Path) -> bool {
    !path.as_os_str().is_empty()
        && path
            .components()
            .all(|component| matches!(component, Component::Normal(_)))
}

fn extract_archive(
    archive: &Path,
    destination: &Path,
    limit: u64,
    expected_root: &str,
    cancellation: &AtomicBool,
) -> Result<(), String> {
    let input = std::fs::File::open(archive)
        .map_err(|error| format!("Failed to open LADA package: {}", error))?;
    let decoder = zstd::Decoder::new(input)
        .map_err(|error| format!("Failed to decompress LADA package: {}", error))?;
    let mut archive = tar::Archive::new(decoder);
    let mut extracted = 0u64;
    let mut entry_count = 0usize;
    let mut files = BTreeSet::new();
    for entry in archive
        .entries()
        .map_err(|error| format!("Failed to read LADA package: {}", error))?
    {
        if cancellation.load(Ordering::SeqCst) {
            return Err("LADA installation was cancelled".into());
        }
        entry_count += 1;
        if entry_count > MAX_ARCHIVE_ENTRIES {
            return Err("LADA package contains too many entries".into());
        }
        let mut entry = entry.map_err(|error| format!("Failed to read LADA package: {}", error))?;
        let relative = entry
            .path()
            .map_err(|error| format!("Invalid path in LADA package: {}", error))?
            .into_owned();
        let root_matches = relative
            .components()
            .next()
            .is_some_and(|component| component.as_os_str() == expected_root);
        if !safe_archive_path(&relative) || !root_matches {
            return Err(format!(
                "LADA package contains a path outside its {} layer",
                expected_root
            ));
        }
        let kind = entry.header().entry_type();
        if kind.is_dir() {
            std::fs::create_dir_all(destination.join(&relative))
                .map_err(|error| format!("Failed to create LADA package directory: {}", error))?;
            continue;
        }
        if !kind.is_file() || !files.insert(relative.clone()) {
            return Err("LADA package contains links, special files, or duplicate entries".into());
        }
        extracted = extracted
            .checked_add(entry.size())
            .filter(|size| *size <= limit)
            .ok_or_else(|| "LADA package exceeded its declared installed size".to_string())?;
        let target = destination.join(&relative);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("Failed to create LADA package directory: {}", error))?;
        }
        let mut output = std::fs::File::create(&target)
            .map_err(|error| format!("Failed to extract LADA package: {}", error))?;
        let mut buffer = [0u8; DOWNLOAD_CHUNK_BYTES];
        loop {
            if cancellation.load(Ordering::SeqCst) {
                return Err("LADA installation was cancelled".into());
            }
            let count = entry
                .read(&mut buffer)
                .map_err(|error| format!("Failed to extract LADA package: {}", error))?;
            if count == 0 {
                break;
            }
            output
                .write_all(&buffer[..count])
                .map_err(|error| format!("Failed to extract LADA package: {}", error))?;
        }
        output
            .flush()
            .map_err(|error| format!("Failed to flush LADA package: {}", error))?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = entry.header().mode().unwrap_or(0o644) & 0o777;
            std::fs::set_permissions(&target, std::fs::Permissions::from_mode(mode))
                .map_err(|error| format!("Failed to set LADA package permissions: {}", error))?;
        }
    }
    Ok(())
}

fn hash_file(path: &Path) -> Result<(String, u64), String> {
    let mut input = std::fs::File::open(path)
        .map_err(|error| format!("Failed to open {}: {}", path.display(), error))?;
    let mut digest = Sha256::new();
    let mut size = 0u64;
    let mut buffer = [0u8; DOWNLOAD_CHUNK_BYTES];
    loop {
        let count = input
            .read(&mut buffer)
            .map_err(|error| format!("Failed to verify {}: {}", path.display(), error))?;
        if count == 0 {
            break;
        }
        size += count as u64;
        digest.update(&buffer[..count]);
    }
    Ok((format!("{:x}", digest.finalize()), size))
}

fn validate_models(root: &Path, models: &[LadaModel]) -> Result<(), String> {
    for model in models {
        let path = root.join("models").join(&model.name);
        let (sha256, size) = hash_file(&path)?;
        if sha256 != model.sha256 || size != model.size {
            return Err(format!(
                "Installed LADA model failed verification: {}",
                model.name
            ));
        }
    }
    Ok(())
}

fn probe_config(
    root: &Path,
    manifest: &LadaReleaseManifest,
    preference: LadaBackendPreference,
) -> serde_json::Value {
    serde_json::json!({
        "protocol_version": LADA_PROTOCOL_VERSION,
        "upstream_revision": LADA_UPSTREAM_REVISION,
        "expected_upstream_revision": LADA_UPSTREAM_REVISION,
        "model_revision": LADA_MODEL_REVISION,
        "requested_backend": match preference {
            LadaBackendPreference::Auto => "auto",
            LadaBackendPreference::Cuda => "cuda",
            LadaBackendPreference::Xpu => "xpu",
        },
        "fp16": true,
        "model_probe_size": 256,
        "model_probe_frames": 2,
        "max_probe_seconds": LADA_MODEL_PROBE_TIMEOUT_SECS,
        "models": manifest.models.iter().map(|model| serde_json::json!({
            "name": model.name,
            "role": model.role,
            "variant": model.variant,
            "default": model.default,
            "size": model.size,
            "sha256": model.sha256,
            "path": root.join("models").join(&model.name),
        })).collect::<Vec<_>>(),
    })
}

fn deployment_for(
    root: &Path,
    manifest: &LadaReleaseManifest,
    backend: LadaBackend,
    package: &str,
) -> LadaDeployment {
    LadaDeployment {
        addon_version: manifest.version.clone(),
        protocol_version: manifest.protocol_version,
        upstream_revision: manifest.upstream.revision.clone(),
        model_revision: manifest.model_repository.revision.clone(),
        backend_compatibility: manifest.backend_compatibility.clone(),
        selected_backend: backend,
        selected_package: package.into(),
        artifact_sha256: [COMMON_PACKAGE, package, MODEL_PACKAGE]
            .into_iter()
            .map(|name| (name.to_string(), manifest.packages[name].sha256.clone()))
            .collect(),
        executable: root.join("runtime/bin/localbooru-lada-sidecar"),
        probe_config: root.join("probe.json"),
    }
}

fn write_synced(path: &Path, contents: &[u8], label: &str) -> Result<(), String> {
    let mut file = std::fs::File::create(path)
        .map_err(|error| format!("Failed to create {}: {}", label, error))?;
    file.write_all(contents)
        .map_err(|error| format!("Failed to write {}: {}", label, error))?;
    file.sync_all()
        .map_err(|error| format!("Failed to sync {}: {}", label, error))
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<(), String> {
    std::fs::File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| format!("Failed to sync LADA installation directory: {}", error))
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<(), String> {
    Ok(())
}

fn write_deployment(
    root: &Path,
    manifest: &LadaReleaseManifest,
    deployment: &LadaDeployment,
    preference: LadaBackendPreference,
) -> Result<(), String> {
    let deployment_root = deployment
        .probe_config
        .parent()
        .ok_or_else(|| "LADA probe configuration has no deployment directory".to_string())?;
    write_synced(
        &root.join("probe.json"),
        &serde_json::to_vec_pretty(&probe_config(deployment_root, manifest, preference))
            .map_err(|error| format!("Failed to serialize LADA probe configuration: {}", error))?,
        "LADA probe configuration",
    )?;
    write_synced(
        &root.join("release-manifest.json"),
        &serde_json::to_vec_pretty(manifest)
            .map_err(|error| format!("Failed to serialize LADA release manifest: {}", error))?,
        "LADA release manifest",
    )?;
    write_synced(
        &root.join(LADA_DEPLOYMENT_FILE),
        &serde_json::to_vec_pretty(deployment)
            .map_err(|error| format!("Failed to serialize LADA deployment: {}", error))?,
        "LADA deployment",
    )?;
    sync_directory(root)
}

fn activate(staging: &Path, active: &Path, backup: &Path) -> Result<(), String> {
    let parent = active
        .parent()
        .ok_or_else(|| "LADA deployment has no managed parent directory".to_string())?;
    if backup.exists() {
        std::fs::remove_dir_all(backup)
            .map_err(|error| format!("Failed to clear old LADA rollback data: {}", error))?;
        sync_directory(parent)?;
    }
    let had_active = active.exists();
    if had_active {
        std::fs::rename(active, backup)
            .map_err(|error| format!("Failed to preserve the active LADA deployment: {}", error))?;
        sync_directory(parent)?;
    }
    if let Err(error) = std::fs::rename(staging, active) {
        if had_active {
            let _ = std::fs::rename(backup, active);
            let _ = sync_directory(parent);
        }
        return Err(format!(
            "Failed to activate the verified LADA deployment: {}",
            error
        ));
    }
    if let Err(error) = sync_directory(parent) {
        log::warn!(
            "Failed to sync the activated LADA deployment directory: {}",
            error
        );
    }
    if had_active {
        if let Err(error) = std::fs::remove_dir_all(backup) {
            log::warn!("Failed to remove old LADA deployment backup: {}", error);
        } else if let Err(error) = sync_directory(parent) {
            log::warn!(
                "Failed to sync removal of the old LADA deployment backup: {}",
                error
            );
        }
    }
    Ok(())
}

pub fn recover_interrupted_install(addons_base: &Path) -> Result<(), String> {
    let Ok(entries) = std::fs::read_dir(addons_base) else {
        return Ok(());
    };
    let mut backups = Vec::new();
    let mut staging = Vec::new();
    for entry in entries.filter_map(Result::ok) {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with(".lada-backup-") {
            backups.push(entry.path());
        } else if name.starts_with(".lada-staging-") {
            staging.push(entry.path());
        }
    }

    let active = addons_base.join("lada");
    if !active.exists() {
        match backups.len() {
            0 => {}
            1 => std::fs::rename(&backups[0], &active).map_err(|error| {
                format!(
                    "Failed to restore the interrupted LADA deployment: {}",
                    error
                )
            })?,
            _ => {
                return Err(
                    "Multiple LADA rollback deployments exist; refusing to discard them".into(),
                )
            }
        }
    }
    for path in backups {
        if path.exists() {
            std::fs::remove_dir_all(path)
                .map_err(|error| format!("Failed to remove stale LADA rollback data: {}", error))?;
        }
    }
    for path in staging {
        std::fs::remove_dir_all(path)
            .map_err(|error| format!("Failed to remove stale LADA staging data: {}", error))?;
    }
    sync_directory(addons_base)
}

pub fn cleanup_staging(addons_base: &Path) -> Result<(), String> {
    let Ok(entries) = std::fs::read_dir(addons_base) else {
        return Ok(());
    };
    for entry in entries.filter_map(Result::ok) {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.starts_with(".lada-staging-") || name.starts_with(".lada-backup-") {
            std::fs::remove_dir_all(entry.path()).map_err(|error| {
                format!("Failed to remove stale LADA installation data: {}", error)
            })?;
        }
    }
    sync_directory(addons_base)
}

pub async fn install(
    addons_base: &Path,
    preference: LadaBackendPreference,
    accepted_license: bool,
    cancellation: Arc<AtomicBool>,
    probe_timeout: Duration,
    progress: &(dyn Fn(LadaInstallProgress) + Send + Sync),
) -> Result<LadaInstallOutcome, String> {
    if !accepted_license {
        return Err(format!(
            "You must accept {} before installing LADA",
            LADA_LICENSE
        ));
    }
    if !cfg!(all(target_os = "linux", target_arch = "x86_64")) {
        return Err("LADA video restoration currently requires Linux x86_64".into());
    }
    if cancellation.load(Ordering::SeqCst) {
        return Err("LADA installation was cancelled".into());
    }
    std::fs::create_dir_all(addons_base)
        .map_err(|error| format!("Failed to create managed add-on directory: {}", error))?;
    recover_interrupted_install(addons_base)?;
    progress(LadaInstallProgress {
        stage: LadaInstallStage::Resolving,
        completed_bytes: 0,
        total_bytes: 0,
        package: None,
    });

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(30 * 60))
        .build()
        .map_err(|error| format!("Failed to create LADA download client: {}", error))?;
    let (manifest, source) = load_manifest(&client, &cancellation).await?;
    let detection = lada::detect_host_accelerators();
    let selection =
        lada::select_backend_for_release(&detection, preference, &manifest.backend_compatibility)
            .map_err(|error| error.reason)?;
    let package_names = [COMMON_PACKAGE, selection.package.as_str(), MODEL_PACKAGE];
    let total = package_names.iter().try_fold(0u64, |total, name| {
        total
            .checked_add(manifest.packages[*name].size)
            .ok_or_else(|| "LADA download size overflow".to_string())
    })?;
    if total > MAX_ARTIFACT_BYTES {
        return Err("The selected LADA packages exceed the maximum download size".into());
    }
    let installed_total = package_names.iter().try_fold(0u64, |total, name| {
        total
            .checked_add(manifest.packages[*name].installed_size.unwrap())
            .ok_or_else(|| "LADA installed size overflow".to_string())
    })?;
    if installed_total > MAX_INSTALLED_BYTES {
        return Err("The selected LADA packages exceed the maximum installed size".into());
    }

    let token = uuid::Uuid::new_v4();
    let staging = addons_base.join(format!(".lada-staging-{}", token));
    let backup = addons_base.join(format!(".lada-backup-{}", token));
    let downloads = staging.join(".downloads");
    std::fs::create_dir_all(&downloads)
        .map_err(|error| format!("Failed to create LADA staging directory: {}", error))?;

    let result = async {
        let mut completed = 0u64;
        let mut archives = Vec::new();
        for name in package_names {
            let artifact = &manifest.packages[name];
            let archive = downloads.join(format!("{}.tar.zst", name));
            completed = download_artifact(
                &client,
                &source,
                artifact,
                &archive,
                &cancellation,
                completed,
                total,
                name,
                progress,
            )
            .await?;
            let expected_root = if name == MODEL_PACKAGE {
                "models"
            } else {
                "runtime"
            };
            archives.push((
                name.to_string(),
                archive,
                artifact.installed_size.unwrap(),
                expected_root,
            ));
        }
        if cancellation.load(Ordering::SeqCst) {
            return Err("LADA installation was cancelled".into());
        }
        progress(LadaInstallProgress {
            stage: LadaInstallStage::Installing,
            completed_bytes: completed,
            total_bytes: total,
            package: None,
        });
        let extraction_root = staging.clone();
        let extraction_cancellation = cancellation.clone();
        tokio::task::spawn_blocking(move || {
            for (_, archive, installed_size, expected_root) in archives {
                extract_archive(
                    &archive,
                    &extraction_root,
                    installed_size,
                    expected_root,
                    &extraction_cancellation,
                )?;
            }
            Ok::<_, String>(())
        })
        .await
        .map_err(|error| format!("LADA extraction task failed: {}", error))??;
        std::fs::remove_dir_all(&downloads)
            .map_err(|error| format!("Failed to remove verified LADA archives: {}", error))?;

        if cancellation.load(Ordering::SeqCst) {
            return Err("LADA installation was cancelled".into());
        }
        progress(LadaInstallProgress {
            stage: LadaInstallStage::Validating,
            completed_bytes: total,
            total_bytes: total,
            package: None,
        });
        validate_models(&staging, &manifest.models)?;
        let stage_deployment =
            deployment_for(&staging, &manifest, selection.backend, &selection.package);
        write_deployment(&staging, &manifest, &stage_deployment, preference)?;

        progress(LadaInstallProgress {
            stage: LadaInstallStage::Probing,
            completed_bytes: total,
            total_bytes: total,
            package: None,
        });
        let probe = lada::probe_and_persist(&staging, &stage_deployment, preference, probe_timeout);
        tokio::pin!(probe);
        let readiness = loop {
            tokio::select! {
                readiness = &mut probe => break readiness,
                _ = tokio::time::sleep(CANCEL_POLL_INTERVAL) => {
                    if cancellation.load(Ordering::SeqCst) {
                        return Err("LADA installation was cancelled".into());
                    }
                }
            }
        };
        if readiness.status != LadaReadinessStatus::Ready {
            return Err(readiness
                .reason
                .clone()
                .unwrap_or_else(|| "The LADA accelerator and model probe failed".into()));
        }
        if cancellation.load(Ordering::SeqCst) {
            return Err("LADA installation was cancelled".into());
        }

        progress(LadaInstallProgress {
            stage: LadaInstallStage::Activating,
            completed_bytes: total,
            total_bytes: total,
            package: None,
        });
        let active = addons_base.join("lada");
        let final_deployment =
            deployment_for(&active, &manifest, selection.backend, &selection.package);
        write_deployment(&staging, &manifest, &final_deployment, preference)?;
        if cancellation.load(Ordering::SeqCst) {
            return Err("LADA installation was cancelled".into());
        }
        activate(&staging, &active, &backup)?;
        Ok(LadaInstallOutcome {
            readiness,
            deployment: final_deployment,
        })
    }
    .await;

    if result.is_err() {
        let _ = std::fs::remove_dir_all(&staging);
        if backup.exists() && !addons_base.join("lada").exists() {
            let _ = std::fs::rename(&backup, addons_base.join("lada"));
        }
        let _ = std::fs::remove_dir_all(&backup);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_archive(path: &Path, entry_path: &str, contents: &[u8]) {
        let output = std::fs::File::create(path).unwrap();
        let encoder = zstd::Encoder::new(output, 1).unwrap();
        let mut builder = tar::Builder::new(encoder);
        let mut header = tar::Header::new_gnu();
        header.set_mode(0o644);
        header.set_size(contents.len() as u64);
        header.set_cksum();
        builder
            .append_data(&mut header, entry_path, contents)
            .unwrap();
        let encoder = builder.into_inner().unwrap();
        encoder.finish().unwrap();
    }

    fn valid_manifest() -> LadaReleaseManifest {
        let models = expected_models()
            .into_iter()
            .map(|(name, (sha256, size, role, variant, default))| LadaModel {
                name: name.into(),
                role: role.into(),
                variant: variant.into(),
                default,
                size,
                sha256: sha256.into(),
                source_url: format!(
                    "https://huggingface.co/ladaapp/lada/resolve/{}/{}",
                    LADA_MODEL_REVISION, name
                ),
            })
            .collect();
        let artifact = |filename: &str| {
            LadaArtifact {
            url: format!(
                "https://github.com/DonutsDelivery/localbooru-lada-addon/releases/download/v0.1.0/{}",
                filename
            ),
            sha256: "a".repeat(64),
            size: 1,
            installed_size: Some(1),
        }
        };
        LadaReleaseManifest {
            schema_version: 1,
            addon_id: "lada".into(),
            version: LADA_ADDON_VERSION.into(),
            protocol_version: LADA_PROTOCOL_VERSION,
            license: LADA_LICENSE.into(),
            source_url: LADA_SOURCE_URL.into(),
            upstream: LadaReleaseIdentity {
                repository: "https://github.com/ladaapp/lada".into(),
                revision: LADA_UPSTREAM_REVISION.into(),
                license: LADA_LICENSE.into(),
            },
            model_repository: LadaReleaseIdentity {
                repository: "https://huggingface.co/ladaapp/lada".into(),
                revision: LADA_MODEL_REVISION.into(),
                license: LADA_LICENSE.into(),
            },
            models,
            backend_compatibility: LadaBackendCompatibility::default(),
            packages: BTreeMap::from([
                (
                    COMMON_PACKAGE.into(),
                    artifact("linux-x86_64-common.tar.zst"),
                ),
                (
                    "linux_x86_64_cuda".into(),
                    artifact("linux-x86_64-cuda.tar.zst"),
                ),
                (
                    "linux_x86_64_xpu".into(),
                    artifact("linux-x86_64-xpu.tar.zst"),
                ),
                (MODEL_PACKAGE.into(), artifact("models.tar.zst")),
            ]),
            corresponding_source: artifact("source.tar.zst"),
        }
    }

    // AC: @lada-managed-install ac-verified-activation
    #[test]
    fn trusted_manifest_rejects_identity_model_and_package_changes() {
        let manifest = valid_manifest();
        validate_release_manifest(&manifest).unwrap();

        let mut changed = manifest.clone();
        changed.upstream.revision = "untrusted".into();
        assert!(validate_release_manifest(&changed).is_err());
        let mut changed = manifest.clone();
        changed.models[0].sha256 = "b".repeat(64);
        assert!(validate_release_manifest(&changed).is_err());
        let mut changed = manifest.clone();
        changed.models[0].source_url = "https://example.com/model.pt".into();
        assert!(validate_release_manifest(&changed).is_err());
        let mut changed = manifest.clone();
        changed.packages.get_mut(COMMON_PACKAGE).unwrap().url =
            "https://example.com/linux-x86_64-common.tar.zst".into();
        assert!(validate_release_manifest(&changed).is_err());
        let mut changed = manifest;
        changed.packages.remove(MODEL_PACKAGE);
        assert!(validate_release_manifest(&changed).is_err());
    }

    // AC: @lada-managed-install ac-verified-activation
    #[test]
    fn final_deployment_metadata_is_written_inside_staging() {
        let root = std::env::temp_dir().join(format!(
            "localbooru-lada-final-metadata-test-{}",
            uuid::Uuid::new_v4()
        ));
        let staging = root.join("staging");
        let active = root.join("lada");
        std::fs::create_dir_all(&staging).unwrap();
        let manifest = valid_manifest();
        let deployment = deployment_for(&active, &manifest, LadaBackend::Cuda, "linux_x86_64_cuda");

        write_deployment(
            &staging,
            &manifest,
            &deployment,
            LadaBackendPreference::Auto,
        )
        .unwrap();

        assert!(staging.join("probe.json").is_file());
        assert!(!active.exists());
        let persisted = LadaDeployment::load(&staging).unwrap();
        assert_eq!(
            persisted.executable,
            active.join("runtime/bin/localbooru-lada-sidecar")
        );
        assert_eq!(persisted.probe_config, active.join("probe.json"));
        let config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(staging.join("probe.json")).unwrap()).unwrap();
        assert_eq!(
            config["max_probe_seconds"].as_f64(),
            Some(LADA_MODEL_PROBE_TIMEOUT_SECS)
        );
        assert!(LADA_PROBE_TIMEOUT.as_secs_f64() > LADA_MODEL_PROBE_TIMEOUT_SECS);
        assert!(config["models"].as_array().unwrap().iter().all(|model| {
            Path::new(model["path"].as_str().unwrap()).starts_with(active.join("models"))
        }));
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @lada-managed-install ac-atomic-rollback
    #[tokio::test]
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    async fn installation_observes_a_preexisting_cancellation_signal() {
        let root = std::env::temp_dir().join(format!(
            "localbooru-lada-pre-cancel-test-{}",
            uuid::Uuid::new_v4()
        ));
        let cancellation = Arc::new(AtomicBool::new(true));

        let error = match install(
            &root,
            LadaBackendPreference::Auto,
            true,
            cancellation,
            Duration::from_secs(1),
            &|_| {},
        )
        .await
        {
            Ok(_) => panic!("pre-cancelled installation unexpectedly succeeded"),
            Err(error) => error,
        };

        assert!(error.contains("cancelled"));
        assert!(!root.exists());
    }

    // AC: @lada-managed-install ac-atomic-rollback
    #[test]
    fn failed_activation_restores_existing_deployment() {
        let root = std::env::temp_dir().join(format!(
            "localbooru-lada-activation-test-{}",
            uuid::Uuid::new_v4()
        ));
        let active = root.join("lada");
        let staging = root.join("staging");
        let backup = root.join("backup");
        std::fs::create_dir_all(&active).unwrap();
        std::fs::write(active.join("identity"), "old").unwrap();
        std::fs::create_dir_all(&staging).unwrap();
        std::fs::write(staging.join("identity"), "new").unwrap();
        std::fs::create_dir_all(&backup).unwrap();
        std::fs::write(backup.join("block"), "block").unwrap();

        // Removing a non-empty backup succeeds, so force activation failure by replacing staging with a missing path.
        std::fs::remove_dir_all(&staging).unwrap();
        assert!(activate(&staging, &active, &backup).is_err());
        assert_eq!(
            std::fs::read_to_string(active.join("identity")).unwrap(),
            "old"
        );
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @lada-managed-install ac-atomic-rollback
    #[test]
    fn interrupted_activation_restores_the_only_working_backup() {
        let root = std::env::temp_dir().join(format!(
            "localbooru-lada-recovery-test-{}",
            uuid::Uuid::new_v4()
        ));
        let backup = root.join(".lada-backup-interrupted");
        std::fs::create_dir_all(&backup).unwrap();
        std::fs::write(backup.join("identity"), "working").unwrap();
        std::fs::create_dir_all(root.join(".lada-staging-interrupted")).unwrap();

        recover_interrupted_install(&root).unwrap();

        assert_eq!(
            std::fs::read_to_string(root.join("lada/identity")).unwrap(),
            "working"
        );
        assert!(!root.join(".lada-backup-interrupted").exists());
        assert!(!root.join(".lada-staging-interrupted").exists());
        let _ = std::fs::remove_dir_all(root);
    }

    // AC: @lada-managed-install ac-verified-activation
    #[test]
    fn extraction_enforces_layer_roots_and_observes_cancellation() {
        let root = std::env::temp_dir().join(format!(
            "localbooru-lada-extraction-test-{}",
            uuid::Uuid::new_v4()
        ));
        std::fs::create_dir_all(&root).unwrap();
        let archive = root.join("runtime.tar.zst");
        let destination = root.join("destination");
        write_archive(&archive, "runtime/bin/tool", b"tool");

        let active = AtomicBool::new(false);
        extract_archive(&archive, &destination, 4, "runtime", &active).unwrap();
        assert_eq!(
            std::fs::read(destination.join("runtime/bin/tool")).unwrap(),
            b"tool"
        );

        let wrong_root = root.join("wrong-root");
        assert!(extract_archive(&archive, &wrong_root, 4, "models", &active).is_err());
        let cancelled = AtomicBool::new(true);
        assert!(
            extract_archive(&archive, &root.join("cancelled"), 4, "runtime", &cancelled)
                .unwrap_err()
                .contains("cancelled")
        );
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn archive_paths_reject_parent_absolute_and_empty_components() {
        assert!(safe_archive_path(Path::new("runtime/bin/tool")));
        assert!(!safe_archive_path(Path::new("../escape")));
        assert!(!safe_archive_path(Path::new("/absolute")));
        assert!(!safe_archive_path(Path::new("")));
    }

    // AC: @lada-managed-install ac-clean-uninstall
    #[test]
    fn stale_installation_cleanup_only_removes_lada_transaction_directories() {
        let root = std::env::temp_dir().join(format!(
            "localbooru-lada-cleanup-test-{}",
            uuid::Uuid::new_v4()
        ));
        for name in [".lada-staging-one", ".lada-backup-two", "other-addon"] {
            std::fs::create_dir_all(root.join(name)).unwrap();
        }
        cleanup_staging(&root).unwrap();
        assert!(!root.join(".lada-staging-one").exists());
        assert!(!root.join(".lada-backup-two").exists());
        assert!(root.join("other-addon").exists());
        let _ = std::fs::remove_dir_all(root);
    }
}
