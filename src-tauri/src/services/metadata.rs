//! AI generation metadata extraction service.
//!
//! Extracts prompts and generation parameters from PNG metadata.
//! Supports Automatic1111 (tEXt "parameters") and ComfyUI (tEXt "prompt" JSON).
//!
//! PNG chunks are parsed natively: 4-byte length (big-endian), 4-byte type,
//! `length` bytes of data, 4-byte CRC. tEXt chunks use keyword\0value format.
//! iTXt chunks use keyword\0compression_flag\0compression_method\0language\0translated_keyword\0text.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use rusqlite::Connection;
use serde::{Deserialize, Serialize};

// ─── Data types ──────────────────────────────────────────────────────────────

/// Metadata extracted from an Automatic1111 / ComfyUI generated image.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationMetadata {
    pub prompt: Option<String>,
    pub negative_prompt: Option<String>,
    pub model_name: Option<String>,
    pub sampler: Option<String>,
    pub seed: Option<String>,
    pub steps: Option<i32>,
    pub cfg_scale: Option<f64>,
    pub source_format: Option<String>, // "a1111", "comfyui"
}

/// Result of a metadata extraction attempt.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractionResult {
    pub status: String, // "success", "no_metadata", "config_mismatch", "error"
    pub metadata: Option<GenerationMetadata>,
    pub message: Option<String>,
    pub has_comfyui_data: bool,
}

impl ExtractionResult {
    fn success(metadata: GenerationMetadata) -> Self {
        Self {
            status: "success".into(),
            metadata: Some(metadata),
            message: None,
            has_comfyui_data: false,
        }
    }

    fn no_metadata() -> Self {
        Self {
            status: "no_metadata".into(),
            metadata: None,
            message: None,
            has_comfyui_data: false,
        }
    }

    fn config_mismatch(message: &str) -> Self {
        Self {
            status: "config_mismatch".into(),
            metadata: None,
            message: Some(message.into()),
            has_comfyui_data: true,
        }
    }

    fn error(message: &str) -> Self {
        Self {
            status: "error".into(),
            metadata: None,
            message: Some(message.into()),
            has_comfyui_data: false,
        }
    }
}

// ─── PNG chunk parsing ───────────────────────────────────────────────────────

/// PNG file signature (8 bytes).
const PNG_SIGNATURE: [u8; 8] = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];

/// Extract all tEXt and iTXt chunks from a PNG file.
///
/// Returns a map of keyword -> text value.
/// Non-PNG files return an empty map.
pub fn extract_png_text_chunks(file_path: &str) -> Result<HashMap<String, String>, String> {
    let path = Path::new(file_path);
    if !path.exists() {
        return Err(format!("File not found: {}", file_path));
    }

    // Only process PNG files
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase());
    if ext.as_deref() != Some("png") {
        return Ok(HashMap::new());
    }

    let file = File::open(path).map_err(|e| format!("Failed to open file: {}", e))?;
    let mut reader = BufReader::new(file);

    // Verify PNG signature
    let mut sig = [0u8; 8];
    reader
        .read_exact(&mut sig)
        .map_err(|e| format!("Failed to read PNG signature: {}", e))?;
    if sig != PNG_SIGNATURE {
        return Ok(HashMap::new()); // Not a valid PNG
    }

    let mut chunks = HashMap::new();

    // Read chunks until IEND or EOF
    loop {
        // Read chunk length (4 bytes, big-endian)
        let mut len_buf = [0u8; 4];
        if reader.read_exact(&mut len_buf).is_err() {
            break;
        }
        let chunk_len = u32::from_be_bytes(len_buf) as usize;

        // Read chunk type (4 bytes)
        let mut type_buf = [0u8; 4];
        if reader.read_exact(&mut type_buf).is_err() {
            break;
        }
        let chunk_type = std::str::from_utf8(&type_buf).unwrap_or("");

        if chunk_type == "IEND" {
            break;
        }

        if chunk_type == "tEXt" || chunk_type == "iTXt" {
            // Read chunk data
            let mut data = vec![0u8; chunk_len];
            if reader.read_exact(&mut data).is_err() {
                break;
            }

            // Skip CRC (4 bytes)
            let mut crc_buf = [0u8; 4];
            let _ = reader.read_exact(&mut crc_buf);

            if chunk_type == "tEXt" {
                parse_text_chunk(&data, &mut chunks);
            } else {
                parse_itxt_chunk(&data, &mut chunks);
            }
        } else {
            // Skip chunk data + CRC
            let skip_len = chunk_len + 4;
            let mut skip_buf = vec![0u8; skip_len];
            if reader.read_exact(&mut skip_buf).is_err() {
                break;
            }
        }
    }

    Ok(chunks)
}

/// Parse a tEXt chunk: keyword\0value (both Latin-1, but we treat as UTF-8).
fn parse_text_chunk(data: &[u8], chunks: &mut HashMap<String, String>) {
    if let Some(null_pos) = data.iter().position(|&b| b == 0) {
        let keyword = String::from_utf8_lossy(&data[..null_pos]).to_string();
        let value = String::from_utf8_lossy(&data[null_pos + 1..]).to_string();
        if !keyword.is_empty() {
            chunks.insert(keyword, value);
        }
    }
}

/// Parse an iTXt chunk: keyword\0compression_flag\0compression_method\0language\0translated_keyword\0text
fn parse_itxt_chunk(data: &[u8], chunks: &mut HashMap<String, String>) {
    // Find keyword (null-terminated)
    let keyword_end = match data.iter().position(|&b| b == 0) {
        Some(pos) => pos,
        None => return,
    };
    let keyword = String::from_utf8_lossy(&data[..keyword_end]).to_string();
    if keyword.is_empty() {
        return;
    }

    let mut offset = keyword_end + 1;

    // compression_flag (1 byte)
    if offset >= data.len() {
        return;
    }
    let compression_flag = data[offset];
    offset += 1;

    // compression_method (1 byte)
    if offset >= data.len() {
        return;
    }
    // let _compression_method = data[offset];
    offset += 1;

    // language tag (null-terminated)
    if let Some(null_pos) = data[offset..].iter().position(|&b| b == 0) {
        offset += null_pos + 1;
    } else {
        return;
    }

    // translated keyword (null-terminated)
    if let Some(null_pos) = data[offset..].iter().position(|&b| b == 0) {
        offset += null_pos + 1;
    } else {
        return;
    }

    // Remaining data is the text
    if offset < data.len() {
        if compression_flag == 0 {
            // Uncompressed
            let text = String::from_utf8_lossy(&data[offset..]).to_string();
            chunks.insert(keyword, text);
        }
        // compressed iTXt (compression_flag == 1) uses zlib; rare for SD metadata.
        // We skip compressed chunks since AI generators typically use uncompressed tEXt.
    }
}

// ─── Format detection ────────────────────────────────────────────────────────

/// Check if the chunk map contains Automatic1111 metadata ("parameters" key).
pub fn has_a1111_metadata(chunks: &HashMap<String, String>) -> bool {
    chunks.contains_key("parameters")
}

/// Check if the chunk map contains ComfyUI metadata ("prompt" key with valid JSON).
pub fn has_comfyui_metadata(chunks: &HashMap<String, String>) -> bool {
    if let Some(prompt_str) = chunks.get("prompt") {
        serde_json::from_str::<serde_json::Value>(prompt_str).is_ok()
    } else {
        false
    }
}

// ─── A1111 parsing ───────────────────────────────────────────────────────────

/// Parse Automatic1111's "parameters" text chunk.
///
/// Format:
/// ```text
/// positive prompt text
/// Negative prompt: negative prompt text
/// Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 12345, Size: 512x512, Model: model_name
/// ```
pub fn parse_a1111_metadata(parameters: &str) -> GenerationMetadata {
    let mut metadata = GenerationMetadata {
        source_format: Some("a1111".into()),
        ..Default::default()
    };

    // Split by "Negative prompt:" to get positive and rest
    let parts: Vec<&str> = parameters.splitn(2, "Negative prompt:").collect();
    metadata.prompt = Some(parts[0].trim().to_string());

    if parts.len() > 1 {
        let negative_and_params = parts[1];

        // Find where the key-value pairs start (look for pattern like "\nSteps: ")
        let params_line;
        if let Some(m) = find_params_start(negative_and_params) {
            metadata.negative_prompt = Some(negative_and_params[..m].trim().to_string());
            params_line = &negative_and_params[m..];
        } else {
            metadata.negative_prompt = Some(negative_and_params.trim().to_string());
            params_line = "";
        }

        // Parse key-value pairs
        if !params_line.is_empty() {
            parse_a1111_kvs(params_line, &mut metadata);
        }
    } else {
        // No negative prompt section — look for params at the end of prompt text
        // Check if the prompt itself ends with key-value params
        let prompt_text = parts[0];
        if let Some(m) = find_params_start(prompt_text) {
            metadata.prompt = Some(prompt_text[..m].trim().to_string());
            parse_a1111_kvs(&prompt_text[m..], &mut metadata);
        }
    }

    metadata
}

/// Find the byte offset where A1111 key-value parameters start.
/// Looks for "\nSteps: <digit>" pattern first, then any "\n<Key>: " pattern.
fn find_params_start(text: &str) -> Option<usize> {
    // Primary: look for "\nSteps: <digit>"
    for (i, _) in text.match_indices("\nSteps:") {
        let after = &text[i + 7..]; // len("\nSteps:") == 7
        let after_trimmed = after.trim_start();
        if after_trimmed.starts_with(|c: char| c.is_ascii_digit()) {
            return Some(i + 1); // +1 to skip the newline, keep "Steps:..."
        }
    }

    // Fallback: any line starting with a capitalized key followed by ": "
    for (i, _) in text.match_indices('\n') {
        let rest = &text[i + 1..];
        if rest.len() >= 3 {
            let first = rest.as_bytes()[0];
            if first.is_ascii_uppercase() {
                // Check for "Key: " pattern
                if let Some(colon_pos) = rest.find(": ") {
                    let key = &rest[..colon_pos];
                    // Key should be a single word or two (e.g., "CFG scale")
                    if key.len() <= 20
                        && key
                            .chars()
                            .all(|c| c.is_ascii_alphanumeric() || c == ' ' || c == '_')
                    {
                        return Some(i + 1);
                    }
                }
            }
        }
    }

    None
}

/// Parse key-value pairs from an A1111 parameters line.
fn parse_a1111_kvs(params_line: &str, metadata: &mut GenerationMetadata) {
    // Steps
    if let Some(val) = extract_kv_int(params_line, "Steps") {
        metadata.steps = Some(val);
    }

    // Sampler
    if let Some(val) = extract_kv_str(params_line, "Sampler") {
        metadata.sampler = Some(val);
    }

    // CFG scale
    if let Some(val) = extract_kv_float(params_line, "CFG scale") {
        metadata.cfg_scale = Some(val);
    }

    // Seed
    if let Some(val) = extract_kv_str(params_line, "Seed") {
        // Seed should be numeric
        let trimmed = val.trim();
        if trimmed.chars().all(|c| c.is_ascii_digit()) {
            metadata.seed = Some(trimmed.to_string());
        }
    }

    // Model
    if let Some(val) = extract_kv_str(params_line, "Model") {
        metadata.model_name = Some(val);
    }
}

/// Extract a string value for a given key from "Key: value, Key2: value2" format.
fn extract_kv_str(line: &str, key: &str) -> Option<String> {
    let pattern = format!("{}: ", key);
    if let Some(start) = line.find(&pattern) {
        let value_start = start + pattern.len();
        let rest = &line[value_start..];
        // Value ends at the next comma or newline
        let end = rest
            .find(',')
            .or_else(|| rest.find('\n'))
            .unwrap_or(rest.len());
        let val = rest[..end].trim().to_string();
        if !val.is_empty() {
            return Some(val);
        }
    }
    None
}

/// Extract an integer value for a given key.
fn extract_kv_int(line: &str, key: &str) -> Option<i32> {
    extract_kv_str(line, key).and_then(|s| s.parse().ok())
}

/// Extract a float value for a given key.
fn extract_kv_float(line: &str, key: &str) -> Option<f64> {
    extract_kv_str(line, key).and_then(|s| s.parse().ok())
}

// ─── ComfyUI parsing ─────────────────────────────────────────────────────────

/// Parse ComfyUI's "prompt" JSON chunk.
///
/// The JSON is a dict of node_id -> node_data. Prompt text is extracted from
/// nodes whose IDs match the configured lists. Text is typically found in
/// `inputs.text`, `inputs.string`, `inputs.prompt`, or SDXL fields like
/// `inputs.clip_l` / `inputs.clip_g`.
///
/// KSampler nodes are also scanned for seed, steps, cfg, and sampler_name.
pub fn parse_comfyui_metadata(
    prompt_json: &str,
    node_ids: &[String],
    negative_node_ids: &[String],
) -> (GenerationMetadata, bool) {
    let mut metadata = GenerationMetadata {
        source_format: Some("comfyui".into()),
        ..Default::default()
    };

    let prompt_data: serde_json::Value = match serde_json::from_str(prompt_json) {
        Ok(v) => v,
        Err(_) => return (metadata, false),
    };

    let obj = match prompt_data.as_object() {
        Some(o) => o,
        None => return (metadata, false),
    };

    // Extract positive prompts from configured node IDs
    let text_keys = ["text", "string", "prompt", "clip_l", "clip_g", "positive"];
    let neg_text_keys = ["text", "string", "prompt", "clip_l", "clip_g", "negative"];

    let mut positive_texts: Vec<String> = Vec::new();
    for node_id in node_ids {
        if let Some(node) = obj.get(node_id.as_str()) {
            if let Some(inputs) = node.get("inputs").and_then(|i| i.as_object()) {
                for &key in &text_keys {
                    if let Some(text) = inputs.get(key).and_then(|v| v.as_str()) {
                        let trimmed = text.trim();
                        if !trimmed.is_empty() {
                            positive_texts.push(trimmed.to_string());
                            break;
                        }
                    }
                }
            }
        }
    }

    // Extract negative prompts from configured node IDs
    let mut negative_texts: Vec<String> = Vec::new();
    for node_id in negative_node_ids {
        if let Some(node) = obj.get(node_id.as_str()) {
            if let Some(inputs) = node.get("inputs").and_then(|i| i.as_object()) {
                for &key in &neg_text_keys {
                    if let Some(text) = inputs.get(key).and_then(|v| v.as_str()) {
                        let trimmed = text.trim();
                        if !trimmed.is_empty() {
                            negative_texts.push(trimmed.to_string());
                            break;
                        }
                    }
                }
            }
        }
    }

    if !positive_texts.is_empty() {
        metadata.prompt = Some(positive_texts.join("\n"));
    }
    if !negative_texts.is_empty() {
        metadata.negative_prompt = Some(negative_texts.join("\n"));
    }

    // Scan all nodes for KSampler parameters (seed, steps, cfg, sampler)
    for (_node_id, node) in obj {
        let class_type = node
            .get("class_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if class_type.contains("KSampler") || class_type.to_lowercase().contains("sampler") {
            if let Some(inputs) = node.get("inputs").and_then(|i| i.as_object()) {
                if metadata.seed.is_none() {
                    if let Some(seed_val) = inputs.get("seed").and_then(|v| v.as_f64()) {
                        metadata.seed = Some((seed_val as i64).to_string());
                    }
                }
                if metadata.steps.is_none() {
                    if let Some(steps_val) = inputs.get("steps").and_then(|v| v.as_f64()) {
                        metadata.steps = Some(steps_val as i32);
                    }
                }
                if metadata.cfg_scale.is_none() {
                    if let Some(cfg_val) = inputs.get("cfg").and_then(|v| v.as_f64()) {
                        metadata.cfg_scale = Some(cfg_val);
                    }
                }
                if metadata.sampler.is_none() {
                    if let Some(sampler_name) =
                        inputs.get("sampler_name").and_then(|v| v.as_str())
                    {
                        metadata.sampler = Some(sampler_name.to_string());
                    }
                }
            }
        }
    }

    let found_prompts = metadata.prompt.is_some() || metadata.negative_prompt.is_some();
    (metadata, found_prompts)
}

// ─── Main extraction logic ───────────────────────────────────────────────────

/// Auto-detect format and extract metadata from a PNG file.
///
/// `format_hint` can be "auto", "a1111", "comfyui", or "none".
pub fn extract_metadata(
    file_path: &str,
    comfyui_prompt_node_ids: Option<&[String]>,
    comfyui_negative_node_ids: Option<&[String]>,
    format_hint: &str,
) -> ExtractionResult {
    let chunks = match extract_png_text_chunks(file_path) {
        Ok(c) => c,
        Err(e) => return ExtractionResult::error(&e),
    };

    if chunks.is_empty() {
        return ExtractionResult::no_metadata();
    }

    if format_hint == "none" {
        return ExtractionResult::no_metadata();
    }

    // Auto-detect or use hint
    let detected_format = if format_hint != "auto" {
        Some(format_hint.to_string())
    } else if has_a1111_metadata(&chunks) {
        Some("a1111".into())
    } else if has_comfyui_metadata(&chunks) {
        Some("comfyui".into())
    } else {
        None
    };

    let detected_format = match detected_format {
        Some(f) => f,
        None => return ExtractionResult::no_metadata(),
    };

    // Parse A1111 format
    if detected_format == "a1111" {
        if let Some(parameters) = chunks.get("parameters") {
            let metadata = parse_a1111_metadata(parameters);
            if metadata.prompt.is_some() || metadata.negative_prompt.is_some() {
                return ExtractionResult::success(metadata);
            }
        }
        return ExtractionResult::no_metadata();
    }

    // Parse ComfyUI format
    if detected_format == "comfyui" {
        if let Some(prompt_json) = chunks.get("prompt") {
            let prompt_ids = comfyui_prompt_node_ids.unwrap_or(&[]);
            let negative_ids = comfyui_negative_node_ids.unwrap_or(&[]);

            let (metadata, found_prompts) =
                parse_comfyui_metadata(prompt_json, prompt_ids, negative_ids);

            if found_prompts {
                return ExtractionResult::success(metadata);
            }

            // ComfyUI metadata exists but no prompts extracted
            if !prompt_ids.is_empty() || !negative_ids.is_empty() {
                return ExtractionResult::config_mismatch(
                    "ComfyUI metadata found but configured nodes yielded no prompts",
                );
            } else {
                return ExtractionResult::config_mismatch(
                    "ComfyUI metadata found but no node mapping configured",
                );
            }
        }
    }

    ExtractionResult::no_metadata()
}

/// Extract metadata from a file and save it to the database.
///
/// Reads the image file, detects the AI generation format, parses the metadata,
/// and updates the images table with prompt, negative_prompt, model_name,
/// sampler, seed, steps, and cfg_scale.
pub fn extract_and_save_metadata(
    conn: &Connection,
    image_id: i64,
    file_path: &str,
    comfyui_node_ids: Option<&[String]>,
    comfyui_negative_node_ids: Option<&[String]>,
    format_hint: &str,
) -> Result<ExtractionResult, String> {
    let result = extract_metadata(file_path, comfyui_node_ids, comfyui_negative_node_ids, format_hint);

    if result.status != "success" {
        return Ok(result);
    }

    let metadata = match &result.metadata {
        Some(m) => m,
        None => return Ok(result),
    };

    // Build UPDATE dynamically to only set non-null fields
    let mut sets: Vec<String> = Vec::new();
    let mut values: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

    if let Some(ref prompt) = metadata.prompt {
        sets.push(format!("prompt = ?{}", sets.len() + 1));
        values.push(Box::new(prompt.clone()));
    }
    if let Some(ref negative_prompt) = metadata.negative_prompt {
        sets.push(format!("negative_prompt = ?{}", sets.len() + 1));
        values.push(Box::new(negative_prompt.clone()));
    }
    if let Some(ref model_name) = metadata.model_name {
        sets.push(format!("model_name = ?{}", sets.len() + 1));
        values.push(Box::new(model_name.clone()));
    }
    if let Some(ref sampler) = metadata.sampler {
        sets.push(format!("sampler = ?{}", sets.len() + 1));
        values.push(Box::new(sampler.clone()));
    }
    if let Some(ref seed) = metadata.seed {
        sets.push(format!("seed = ?{}", sets.len() + 1));
        values.push(Box::new(seed.clone()));
    }
    if let Some(steps) = metadata.steps {
        sets.push(format!("steps = ?{}", sets.len() + 1));
        values.push(Box::new(steps));
    }
    if let Some(cfg_scale) = metadata.cfg_scale {
        sets.push(format!("cfg_scale = ?{}", sets.len() + 1));
        values.push(Box::new(cfg_scale));
    }

    if sets.is_empty() {
        return Ok(result);
    }

    // Add image_id as the final parameter
    let id_param = sets.len() + 1;
    let sql = format!(
        "UPDATE images SET {} WHERE id = ?{}",
        sets.join(", "),
        id_param
    );
    values.push(Box::new(image_id));

    let params: Vec<&dyn rusqlite::types::ToSql> = values.iter().map(|v| v.as_ref()).collect();
    conn.execute(&sql, params.as_slice())
        .map_err(|e| format!("Database update failed: {}", e))?;

    Ok(result)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_a1111_basic() {
        let text = "beautiful landscape, mountains\nNegative prompt: ugly, blurry\nSteps: 20, Sampler: Euler a, CFG scale: 7.5, Seed: 12345, Model: sd_xl_base";
        let meta = parse_a1111_metadata(text);
        assert_eq!(meta.prompt.as_deref(), Some("beautiful landscape, mountains"));
        assert_eq!(meta.negative_prompt.as_deref(), Some("ugly, blurry"));
        assert_eq!(meta.steps, Some(20));
        assert_eq!(meta.sampler.as_deref(), Some("Euler a"));
        assert_eq!(meta.cfg_scale, Some(7.5));
        assert_eq!(meta.seed.as_deref(), Some("12345"));
        assert_eq!(meta.model_name.as_deref(), Some("sd_xl_base"));
    }

    #[test]
    fn test_parse_a1111_no_negative() {
        let text = "a cat sitting on a mat\nSteps: 30, Sampler: DPM++ 2M, CFG scale: 12, Seed: 99999";
        let meta = parse_a1111_metadata(text);
        assert_eq!(meta.prompt.as_deref(), Some("a cat sitting on a mat"));
        assert!(meta.negative_prompt.is_none());
        assert_eq!(meta.steps, Some(30));
        assert_eq!(meta.seed.as_deref(), Some("99999"));
    }

    #[test]
    fn test_parse_a1111_prompt_only() {
        let text = "just a prompt with no params";
        let meta = parse_a1111_metadata(text);
        assert_eq!(meta.prompt.as_deref(), Some("just a prompt with no params"));
        assert!(meta.negative_prompt.is_none());
        assert!(meta.steps.is_none());
    }

    #[test]
    fn test_parse_comfyui_basic() {
        let json = r#"{
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "a beautiful sunset", "clip": ["4", 0]}
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "ugly, bad quality", "clip": ["5", 0]}
            },
            "6": {
                "class_type": "KSampler",
                "inputs": {"seed": 42, "steps": 25, "cfg": 8.0, "sampler_name": "euler"}
            }
        }"#;

        let node_ids = vec!["3".to_string()];
        let neg_ids = vec!["4".to_string()];
        let (meta, found) = parse_comfyui_metadata(json, &node_ids, &neg_ids);

        assert!(found);
        assert_eq!(meta.prompt.as_deref(), Some("a beautiful sunset"));
        assert_eq!(meta.negative_prompt.as_deref(), Some("ugly, bad quality"));
        assert_eq!(meta.seed.as_deref(), Some("42"));
        assert_eq!(meta.steps, Some(25));
        assert_eq!(meta.cfg_scale, Some(8.0));
        assert_eq!(meta.sampler.as_deref(), Some("euler"));
    }

    #[test]
    fn test_parse_comfyui_no_matching_nodes() {
        let json = r#"{"3": {"class_type": "CLIPTextEncode", "inputs": {"text": "hello"}}}"#;
        let node_ids = vec!["99".to_string()];
        let neg_ids: Vec<String> = vec![];
        let (meta, found) = parse_comfyui_metadata(json, &node_ids, &neg_ids);
        assert!(!found);
        assert!(meta.prompt.is_none());
    }

    #[test]
    fn test_has_a1111_metadata() {
        let mut chunks = HashMap::new();
        chunks.insert("parameters".to_string(), "some text".to_string());
        assert!(has_a1111_metadata(&chunks));
        assert!(!has_comfyui_metadata(&chunks));
    }

    #[test]
    fn test_has_comfyui_metadata() {
        let mut chunks = HashMap::new();
        chunks.insert("prompt".to_string(), r#"{"3": {}}"#.to_string());
        assert!(has_comfyui_metadata(&chunks));
        assert!(!has_a1111_metadata(&chunks));
    }

    #[test]
    fn test_has_comfyui_metadata_invalid_json() {
        let mut chunks = HashMap::new();
        chunks.insert("prompt".to_string(), "not json".to_string());
        assert!(!has_comfyui_metadata(&chunks));
    }

    #[test]
    fn test_extract_kv_str() {
        let line = "Steps: 20, Sampler: Euler a, CFG scale: 7.5, Seed: 12345";
        assert_eq!(extract_kv_str(line, "Steps"), Some("20".to_string()));
        assert_eq!(extract_kv_str(line, "Sampler"), Some("Euler a".to_string()));
        assert_eq!(extract_kv_str(line, "CFG scale"), Some("7.5".to_string()));
        assert_eq!(extract_kv_str(line, "Seed"), Some("12345".to_string()));
        assert_eq!(extract_kv_str(line, "Missing"), None);
    }
}
