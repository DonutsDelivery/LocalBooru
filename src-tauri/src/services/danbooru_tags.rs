//! Danbooru tag list used to turn generation-prompt words into library tags.
//!
//! Ported from the v0.3.33 Python metadata extractor: the list is downloaded
//! once, cached as `danbooru_tags.txt` in the data dir (reused for a week) and
//! kept in memory for an hour. Every failure is soft: callers get an empty set
//! (or the stale cache) and a log line, never an error.

use std::collections::{BTreeSet, HashSet};
use std::path::Path;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use regex::Regex;

pub const DANBOORU_TAGS_URL: &str = "https://gist.githubusercontent.com/pythongosssss/1d3efa6050356a08cea975183088159a/raw/a18fb2f94f9156cf4476b0c24a09544d6c0baec6/danbooru-tags.txt";
const CACHE_FILE: &str = "danbooru_tags.txt";
const MEMORY_TTL: Duration = Duration::from_secs(60 * 60);
const FILE_MAX_AGE: Duration = Duration::from_secs(7 * 24 * 60 * 60);
/// After a failed download, wait this long before trying again so an offline
/// machine does not stall every queued metadata task on a 30 second timeout.
const FAILURE_RETRY: Duration = Duration::from_secs(5 * 60);
const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(30);

struct Cached {
    tags: Arc<HashSet<String>>,
    loaded_at: Instant,
    ttl: Duration,
}

static MEMORY: tokio::sync::Mutex<Option<Cached>> = tokio::sync::Mutex::const_new(None);

fn parse_tag_list(text: &str) -> HashSet<String> {
    text.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(str::to_string)
        .collect()
}

async fn read_cache_file(path: &Path, max_age: Option<Duration>) -> Option<HashSet<String>> {
    if let Some(max_age) = max_age {
        let age = tokio::fs::metadata(path)
            .await
            .ok()?
            .modified()
            .ok()?
            .elapsed()
            .ok()?;
        if age >= max_age {
            return None;
        }
    }
    let tags = parse_tag_list(&tokio::fs::read_to_string(path).await.ok()?);
    (!tags.is_empty()).then_some(tags)
}

async fn download(client: &reqwest::Client, url: &str) -> Result<String, String> {
    let response = client
        .get(url)
        .timeout(DOWNLOAD_TIMEOUT)
        .send()
        .await
        .map_err(|e| e.to_string())?;
    if !response.status().is_success() {
        return Err(format!("HTTP {}", response.status()));
    }
    response.text().await.map_err(|e| e.to_string())
}

/// Load the tag list from the on-disk cache or `url`.
///
/// Returns the tags and whether they are current. `false` means the download
/// failed and the tags are a stale cache (or empty).
async fn load(client: &reqwest::Client, url: &str, cache_path: &Path) -> (HashSet<String>, bool) {
    if let Some(tags) = read_cache_file(cache_path, Some(FILE_MAX_AGE)).await {
        return (tags, true);
    }

    match download(client, url).await {
        Ok(text) => {
            let tags = parse_tag_list(&text);
            if !tags.is_empty() {
                let tmp = cache_path.with_extension("txt.tmp");
                let written = match tokio::fs::write(&tmp, &text).await {
                    Ok(()) => tokio::fs::rename(&tmp, cache_path).await,
                    Err(error) => Err(error),
                };
                if let Err(error) = written {
                    log::warn!("[MetadataExtractor] Could not cache Danbooru tags: {error}");
                }
                return (tags, true);
            }
            log::warn!("[MetadataExtractor] Downloaded Danbooru tag list was empty");
        }
        Err(error) => {
            log::warn!("[MetadataExtractor] Failed to download Danbooru tags: {error}");
        }
    }

    (
        read_cache_file(cache_path, None).await.unwrap_or_default(),
        false,
    )
}

/// The Danbooru tag set, or an empty set when it cannot be loaded.
pub async fn get_tags(client: &reqwest::Client, data_dir: &Path) -> Arc<HashSet<String>> {
    let mut memory = MEMORY.lock().await;
    if let Some(cached) = memory.as_ref() {
        if cached.loaded_at.elapsed() < cached.ttl {
            return cached.tags.clone();
        }
    }
    let (tags, current) = load(client, DANBOORU_TAGS_URL, &data_dir.join(CACHE_FILE)).await;
    let tags = Arc::new(tags);
    *memory = Some(Cached {
        tags: tags.clone(),
        loaded_at: Instant::now(),
        ttl: if current { MEMORY_TTL } else { FAILURE_RETRY },
    });
    tags
}

fn cached_regex(cell: &'static OnceLock<Regex>, pattern: &str) -> &'static Regex {
    cell.get_or_init(|| Regex::new(pattern).expect("static regex"))
}

/// Return the Danbooru tags (underscore form, sorted, deduplicated) that occur
/// as comma/bracket separated words in `prompt`.
pub fn match_prompt_tags(prompt: &str, valid_tags: &HashSet<String>) -> Vec<String> {
    static SEPARATORS: OnceLock<Regex> = OnceLock::new();
    static LEADING_WEIGHT: OnceLock<Regex> = OnceLock::new();
    static TRAILING_WEIGHT: OnceLock<Regex> = OnceLock::new();

    if prompt.is_empty() || valid_tags.is_empty() {
        return Vec::new();
    }

    let lookup = |token: &str| -> Option<String> {
        [
            token.to_string(),
            token.replace(' ', "_"),
            token.replace('_', " "),
        ]
        .into_iter()
        .find(|candidate| valid_tags.contains(candidate))
        .map(|candidate| candidate.replace(' ', "_"))
    };

    let prompt = prompt.to_lowercase();
    let mut found = BTreeSet::new();
    for token in cached_regex(&SEPARATORS, r"[,()\[\]<>:|]+").split(&prompt) {
        let token = token.trim();
        if token.chars().count() < 2 {
            continue;
        }
        // Exact words first so tags that start with a digit ("1girl") survive;
        // the v0.3.33 extractor stripped digits before matching.
        if let Some(tag) = lookup(token) {
            found.insert(tag);
            continue;
        }
        // Drop prompt weights such as "1.5" or a trailing count.
        let stripped = cached_regex(&LEADING_WEIGHT, r"^\d+\.?\d*\s*").replace(token, "");
        let stripped = cached_regex(&TRAILING_WEIGHT, r"\s*\d+\.?\d*$")
            .replace(stripped.trim(), "")
            .trim()
            .to_string();
        if stripped.is_empty() || stripped == token {
            continue;
        }
        if let Some(tag) = lookup(&stripped) {
            found.insert(tag);
        }
    }
    found.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tag_set(tags: &[&str]) -> HashSet<String> {
        tags.iter().map(|tag| tag.to_string()).collect()
    }

    #[test]
    fn matches_comma_separated_prompt_words() {
        let valid = tag_set(&["1girl", "blue_eyes", "solo", "long hair"]);
        let found = match_prompt_tags("1Girl, Blue Eyes, solo, masterpiece, long_hair", &valid);
        assert_eq!(found, vec!["1girl", "blue_eyes", "long_hair", "solo"]);
    }

    #[test]
    fn handles_weights_brackets_and_lora_syntax() {
        let valid = tag_set(&["smile", "hat", "red_scarf"]);
        let found = match_prompt_tags(
            "(smile:1.2), [hat], ((red scarf)), <lora:style_x:0.8>, 1.5",
            &valid,
        );
        assert_eq!(found, vec!["hat", "red_scarf", "smile"]);
    }

    #[test]
    fn strips_numeric_prefix_and_suffix_only_after_exact_miss() {
        let valid = tag_set(&["cat", "dog"]);
        let found = match_prompt_tags("1.5 cat, dog 2, 3 unknown, x", &valid);
        assert_eq!(found, vec!["cat", "dog"]);
    }

    #[test]
    fn ignores_unknown_words_and_empty_inputs() {
        let valid = tag_set(&["cat"]);
        assert!(match_prompt_tags("dragon, castle", &valid).is_empty());
        assert!(match_prompt_tags("", &valid).is_empty());
        assert!(match_prompt_tags("cat", &HashSet::new()).is_empty());
    }

    #[test]
    fn parses_tag_list_lines() {
        let tags = parse_tag_list("cat\r\n dog \n\nblue_eyes\n");
        assert_eq!(tags, tag_set(&["cat", "dog", "blue_eyes"]));
    }

    #[tokio::test]
    async fn fresh_cache_file_is_used_without_network() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(CACHE_FILE);
        std::fs::write(&path, "cat\ndog\n").unwrap();
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let (tags, current) = load(&client, "http://127.0.0.1:1/unreachable", &path).await;
        assert!(current);
        assert_eq!(tags, tag_set(&["cat", "dog"]));
    }

    #[tokio::test]
    async fn failed_download_falls_back_to_stale_cache_then_to_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(CACHE_FILE);
        let client = reqwest::Client::builder().no_proxy().build().unwrap();
        let url = "http://127.0.0.1:1/unreachable";

        let (tags, current) = load(&client, url, &path).await;
        assert!(!current);
        assert!(tags.is_empty());
        assert!(!path.exists());

        std::fs::write(&path, "cat\n").unwrap();
        let old = std::time::SystemTime::now() - FILE_MAX_AGE - Duration::from_secs(60);
        std::fs::File::options()
            .write(true)
            .open(&path)
            .unwrap()
            .set_modified(old)
            .unwrap();
        let (tags, current) = load(&client, url, &path).await;
        assert!(!current);
        assert_eq!(tags, tag_set(&["cat"]));
    }
}
