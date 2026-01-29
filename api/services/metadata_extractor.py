"""
AI generation metadata extraction service.
Extracts prompts and generation parameters from PNG/WebP metadata.
Supports Automatic1111 (tEXt 'parameters') and ComfyUI (tEXt 'prompt'/'workflow').
"""
import json
import re
import time
import httpx
from PIL import Image
from pathlib import Path
from typing import Optional, Set
from dataclasses import dataclass, field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models import (
    Image as ImageModel, Tag, image_tags, TagCategory,
    DirectoryImage, directory_image_tags
)
from ..database import get_data_dir, directory_db_manager

# Danbooru tags cache
DANBOORU_TAGS_URL = "https://gist.githubusercontent.com/pythongosssss/1d3efa6050356a08cea975183088159a/raw/a18fb2f94f9156cf4476b0c24a09544d6c0baec6/danbooru-tags.txt"
_danbooru_tags_cache: Optional[Set[str]] = None
_danbooru_tags_cache_time: float = 0


@dataclass
class GenerationMetadata:
    """Extracted AI generation metadata"""
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    model_name: Optional[str] = None
    sampler: Optional[str] = None
    seed: Optional[str] = None
    steps: Optional[int] = None
    cfg_scale: Optional[float] = None
    source_format: Optional[str] = None  # 'a1111', 'comfyui', 'unknown'


@dataclass
class ExtractionResult:
    """Result of metadata extraction attempt"""
    status: str  # 'success', 'no_metadata', 'config_mismatch', 'error'
    metadata: Optional[GenerationMetadata] = None
    message: Optional[str] = None
    has_comfyui_data: bool = False  # True if ComfyUI metadata exists but couldn't extract prompts


def extract_png_text_chunks(image_path: str) -> dict[str, str]:
    """Extract all tEXt/iTXt chunks from PNG file"""
    try:
        with Image.open(image_path) as img:
            return dict(img.info) if img.info else {}
    except Exception:
        return {}


def parse_a1111_metadata(parameters_text: str) -> GenerationMetadata:
    """
    Parse Automatic1111 'parameters' format.
    Format:
        positive prompt text
        Negative prompt: negative prompt text
        Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 12345, Size: 512x512, Model: model_name
    """
    metadata = GenerationMetadata(source_format='a1111')

    # Split by "Negative prompt:" to get positive and rest
    parts = parameters_text.split('Negative prompt:', 1)
    metadata.prompt = parts[0].strip()

    if len(parts) > 1:
        # Find where the key-value pairs start (look for pattern like "Steps: ")
        negative_and_params = parts[1]
        param_match = re.search(r'\n(Steps:\s*\d+)', negative_and_params)
        if param_match:
            metadata.negative_prompt = negative_and_params[:param_match.start()].strip()
            params_line = negative_and_params[param_match.start():].strip()
        else:
            # Try to find any key-value pattern
            param_match = re.search(r'\n([A-Z][a-z]+:\s*)', negative_and_params)
            if param_match:
                metadata.negative_prompt = negative_and_params[:param_match.start()].strip()
                params_line = negative_and_params[param_match.start():].strip()
            else:
                metadata.negative_prompt = negative_and_params.strip()
                params_line = ""

        # Parse key-value pairs
        if params_line:
            # Extract specific values using regex
            steps_match = re.search(r'Steps:\s*(\d+)', params_line)
            if steps_match:
                metadata.steps = int(steps_match.group(1))

            sampler_match = re.search(r'Sampler:\s*([^,]+)', params_line)
            if sampler_match:
                metadata.sampler = sampler_match.group(1).strip()

            cfg_match = re.search(r'CFG scale:\s*([\d.]+)', params_line)
            if cfg_match:
                metadata.cfg_scale = float(cfg_match.group(1))

            seed_match = re.search(r'Seed:\s*(\d+)', params_line)
            if seed_match:
                metadata.seed = seed_match.group(1)

            model_match = re.search(r'Model:\s*([^,\n]+)', params_line)
            if model_match:
                metadata.model_name = model_match.group(1).strip()

    return metadata


def parse_comfyui_metadata(
    prompt_json: str,
    workflow_json: Optional[str],
    prompt_node_ids: list[str],
    negative_node_ids: list[str]
) -> tuple[GenerationMetadata, bool]:
    """
    Parse ComfyUI prompt/workflow JSON.
    Uses configured node IDs to find prompt text.

    Returns:
        tuple of (metadata, found_prompts) - found_prompts is False if nodes exist but no text extracted
    """
    metadata = GenerationMetadata(source_format='comfyui')

    try:
        prompt_data = json.loads(prompt_json)
    except json.JSONDecodeError:
        return metadata, False

    # Extract prompts from configured node IDs
    positive_texts = []
    negative_texts = []

    for node_id in prompt_node_ids:
        if node_id in prompt_data:
            node = prompt_data[node_id]
            # Look for text in 'inputs' -> common text field names
            inputs = node.get('inputs', {})
            for key in ['text', 'string', 'prompt', 'clip_l', 'clip_g', 'positive']:
                if key in inputs and isinstance(inputs[key], str) and inputs[key].strip():
                    positive_texts.append(inputs[key])
                    break

    for node_id in negative_node_ids:
        if node_id in prompt_data:
            node = prompt_data[node_id]
            inputs = node.get('inputs', {})
            for key in ['text', 'string', 'prompt', 'clip_l', 'clip_g', 'negative']:
                if key in inputs and isinstance(inputs[key], str) and inputs[key].strip():
                    negative_texts.append(inputs[key])
                    break

    if positive_texts:
        metadata.prompt = '\n'.join(positive_texts)
    if negative_texts:
        metadata.negative_prompt = '\n'.join(negative_texts)

    # Try to extract seed and other params from KSampler nodes
    for node_id, node in prompt_data.items():
        class_type = node.get('class_type', '')
        if 'KSampler' in class_type or 'sampler' in class_type.lower():
            inputs = node.get('inputs', {})
            if 'seed' in inputs and metadata.seed is None:
                seed_val = inputs['seed']
                if isinstance(seed_val, (int, float)):
                    metadata.seed = str(int(seed_val))
            if 'steps' in inputs and metadata.steps is None:
                steps_val = inputs['steps']
                if isinstance(steps_val, (int, float)):
                    metadata.steps = int(steps_val)
            if 'cfg' in inputs and metadata.cfg_scale is None:
                cfg_val = inputs['cfg']
                if isinstance(cfg_val, (int, float)):
                    metadata.cfg_scale = float(cfg_val)
            if 'sampler_name' in inputs and metadata.sampler is None:
                metadata.sampler = inputs['sampler_name']

    # Check if we found any prompts
    found_prompts = bool(metadata.prompt or metadata.negative_prompt)
    return metadata, found_prompts


def discover_comfyui_nodes(image_path: str) -> list[dict]:
    """
    Discover all text-containing nodes in a ComfyUI image.
    Returns list of {node_id, node_type, field, sample_text} for UI configuration.
    """
    chunks = extract_png_text_chunks(image_path)

    if not chunks:
        return []

    # Try both prompt and workflow chunks
    prompt_data = {}
    workflow_data = {}

    if 'prompt' in chunks:
        try:
            prompt_data = json.loads(chunks['prompt'])
        except json.JSONDecodeError:
            pass

    if 'workflow' in chunks:
        try:
            workflow_data = json.loads(chunks['workflow'])
        except json.JSONDecodeError:
            pass

    # If no valid data, return empty
    if not prompt_data and not workflow_data:
        return []

    nodes = []
    seen_node_ids = set()

    # Build node type lookup from workflow
    node_types = {}
    for node in workflow_data.get('nodes', []):
        node_id = str(node.get('id'))
        node_type = node.get('properties', {}).get('Node name for S&R', node.get('type', 'Unknown'))
        node_types[node_id] = node_type

    # Extract text from prompt data (runtime values)
    for node_id, node in prompt_data.items():
        class_type = node.get('class_type', 'Unknown')
        node_type = node_types.get(node_id, class_type)
        inputs = node.get('inputs', {})

        # Check common text field names
        for key in ['text', 'string', 'prompt', 'clip_l', 'clip_g', 'positive', 'negative']:
            if key in inputs and isinstance(inputs[key], str):
                text = inputs[key].strip()
                if len(text) > 5:  # Only show meaningful text
                    nodes.append({
                        'node_id': node_id,
                        'node_type': node_type,
                        'class_type': class_type,
                        'field': key,
                        'sample_text': text[:300] + ('...' if len(text) > 300 else '')
                    })
                    seen_node_ids.add(node_id)
                    break

    # Also check widgets_values from workflow for nodes not yet found
    for node in workflow_data.get('nodes', []):
        node_id = str(node.get('id'))
        if node_id in seen_node_ids:
            continue

        node_type = node.get('properties', {}).get('Node name for S&R', node.get('type', 'Unknown'))
        widgets = node.get('widgets_values', [])

        for i, val in enumerate(widgets):
            if isinstance(val, str) and len(val.strip()) > 20:
                text = val.strip()
                nodes.append({
                    'node_id': node_id,
                    'node_type': node_type,
                    'class_type': node.get('type', 'Unknown'),
                    'field': f'widget_{i}',
                    'sample_text': text[:300] + ('...' if len(text) > 300 else '')
                })
                seen_node_ids.add(node_id)
                break

    return nodes


def has_comfyui_metadata(image_path: str) -> bool:
    """Check if image has ComfyUI metadata"""
    chunks = extract_png_text_chunks(image_path)
    return 'prompt' in chunks or 'workflow' in chunks


def has_a1111_metadata(image_path: str) -> bool:
    """Check if image has A1111 metadata"""
    chunks = extract_png_text_chunks(image_path)
    return 'parameters' in chunks


def extract_metadata(
    image_path: str,
    comfyui_prompt_node_ids: list[str] = None,
    comfyui_negative_node_ids: list[str] = None,
    format_hint: str = 'auto'
) -> ExtractionResult:
    """
    Main extraction function. Auto-detects format or uses hint.

    Returns ExtractionResult with status:
        - 'success': Metadata extracted successfully
        - 'no_metadata': No AI generation metadata found in file
        - 'config_mismatch': ComfyUI metadata exists but configured nodes yielded no prompts
        - 'error': Error during extraction
    """
    try:
        chunks = extract_png_text_chunks(image_path)
    except Exception as e:
        return ExtractionResult(status='error', message=str(e))

    if not chunks:
        return ExtractionResult(status='no_metadata')

    # Auto-detect or use hint
    detected_format = None
    if 'parameters' in chunks:
        detected_format = 'a1111'
    elif 'prompt' in chunks:
        detected_format = 'comfyui'

    if format_hint == 'none':
        return ExtractionResult(status='no_metadata')

    if format_hint != 'auto':
        detected_format = format_hint

    if not detected_format:
        return ExtractionResult(status='no_metadata')

    # Parse A1111 format
    if detected_format == 'a1111' and 'parameters' in chunks:
        metadata = parse_a1111_metadata(chunks['parameters'])
        if metadata.prompt or metadata.negative_prompt:
            return ExtractionResult(status='success', metadata=metadata)
        return ExtractionResult(status='no_metadata')

    # Parse ComfyUI format
    if detected_format == 'comfyui' and 'prompt' in chunks:
        prompt_node_ids = comfyui_prompt_node_ids or []
        negative_node_ids = comfyui_negative_node_ids or []

        metadata, found_prompts = parse_comfyui_metadata(
            chunks['prompt'],
            chunks.get('workflow'),
            prompt_node_ids,
            negative_node_ids
        )

        if found_prompts:
            return ExtractionResult(status='success', metadata=metadata)

        # ComfyUI metadata exists but no prompts extracted with current config
        if prompt_node_ids or negative_node_ids:
            # User configured nodes but they didn't yield anything
            return ExtractionResult(
                status='config_mismatch',
                message='ComfyUI metadata found but configured nodes yielded no prompts',
                has_comfyui_data=True
            )
        else:
            # No configuration set yet
            return ExtractionResult(
                status='config_mismatch',
                message='ComfyUI metadata found but no node mapping configured',
                has_comfyui_data=True
            )

    return ExtractionResult(status='no_metadata')


async def extract_and_save_metadata(
    image_path: str,
    image_id: int,
    db: AsyncSession,
    comfyui_prompt_node_ids: list[str] = None,
    comfyui_negative_node_ids: list[str] = None,
    format_hint: str = 'auto',
    directory_id: int = None
) -> ExtractionResult:
    """Extract metadata and update database"""

    result = extract_metadata(
        image_path,
        comfyui_prompt_node_ids,
        comfyui_negative_node_ids,
        format_hint
    )

    if result.status != 'success' or result.metadata is None:
        return result

    # Update image record in appropriate database
    if directory_id:
        # Image is in a directory database
        dir_db = await directory_db_manager.get_session(directory_id)
        try:
            query = select(DirectoryImage).where(DirectoryImage.id == image_id)
            db_result = await dir_db.execute(query)
            image = db_result.scalar_one_or_none()

            if image:
                metadata = result.metadata
                if metadata.prompt:
                    image.prompt = metadata.prompt
                if metadata.negative_prompt:
                    image.negative_prompt = metadata.negative_prompt
                if metadata.model_name:
                    image.model_name = metadata.model_name
                if metadata.sampler:
                    image.sampler = metadata.sampler
                if metadata.seed:
                    image.seed = metadata.seed
                if metadata.steps:
                    image.steps = metadata.steps
                if metadata.cfg_scale:
                    image.cfg_scale = metadata.cfg_scale

                await dir_db.commit()
        finally:
            await dir_db.close()
    else:
        # Legacy: image is in main database
        query = select(ImageModel).where(ImageModel.id == image_id)
        db_result = await db.execute(query)
        image = db_result.scalar_one_or_none()

        if image:
            metadata = result.metadata
            if metadata.prompt:
                image.prompt = metadata.prompt
            if metadata.negative_prompt:
                image.negative_prompt = metadata.negative_prompt
            if metadata.model_name:
                image.model_name = metadata.model_name
            if metadata.sampler:
                image.sampler = metadata.sampler
            if metadata.seed:
                image.seed = metadata.seed
            if metadata.steps:
                image.steps = metadata.steps
            if metadata.cfg_scale:
                image.cfg_scale = metadata.cfg_scale

            await db.commit()

    return result


def get_danbooru_tags_sync() -> Set[str]:
    """
    Load Danbooru tag list from cache or download.
    Uses a file cache that refreshes weekly.
    """
    global _danbooru_tags_cache, _danbooru_tags_cache_time

    # Return memory cache if recent (1 hour)
    if _danbooru_tags_cache and (time.time() - _danbooru_tags_cache_time) < 3600:
        return _danbooru_tags_cache

    cache_path = get_data_dir() / 'danbooru_tags.txt'

    # Check file cache (refresh weekly)
    if cache_path.exists():
        stat = cache_path.stat()
        age_days = (time.time() - stat.st_mtime) / 86400
        if age_days < 7:
            try:
                tags = set(cache_path.read_text().splitlines())
                _danbooru_tags_cache = tags
                _danbooru_tags_cache_time = time.time()
                return tags
            except Exception:
                pass

    # Download fresh
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(DANBOORU_TAGS_URL)
            if response.status_code == 200:
                cache_path.write_text(response.text)
                tags = set(response.text.splitlines())
                _danbooru_tags_cache = tags
                _danbooru_tags_cache_time = time.time()
                return tags
    except Exception as e:
        print(f"[MetadataExtractor] Failed to download Danbooru tags: {e}")

    # Fallback to file cache if download fails
    if cache_path.exists():
        try:
            tags = set(cache_path.read_text().splitlines())
            _danbooru_tags_cache = tags
            _danbooru_tags_cache_time = time.time()
            return tags
        except Exception:
            pass

    return set()


def extract_tags_from_prompt(prompt: str, valid_tags: Set[str]) -> list[str]:
    """
    Extract valid booru tags from prompt text.
    Only returns tags that exist in the valid_tags set.
    """
    if not prompt or not valid_tags:
        return []

    # Normalize prompt
    prompt_lower = prompt.lower()

    # Split by common prompt separators
    # SD prompts use commas, parentheses, and various other delimiters
    tokens = re.split(r'[,\(\)\[\]<>:\|]+', prompt_lower)

    found_tags = set()
    for token in tokens:
        # Clean up the token
        token = token.strip()

        # Skip empty or very short tokens
        if len(token) < 2:
            continue

        # Remove weight syntax like "1.5" from tokens
        token = re.sub(r'^\d+\.?\d*\s*', '', token).strip()
        token = re.sub(r'\s*\d+\.?\d*$', '', token).strip()

        if not token:
            continue

        # Try direct match
        if token in valid_tags:
            found_tags.add(token)
            continue

        # Try with underscores instead of spaces
        token_underscore = token.replace(' ', '_')
        if token_underscore in valid_tags:
            found_tags.add(token_underscore)
            continue

        # Try with spaces instead of underscores
        token_space = token.replace('_', ' ')
        if token_space in valid_tags:
            found_tags.add(token_space)

    return list(found_tags)


async def add_tags_from_prompt(
    image_id: int,
    prompt: str,
    db: AsyncSession,
    confidence: float = 0.8,
    directory_id: int = None
) -> list[str]:
    """
    Extract tags from prompt and add them to the image.
    Only adds tags that exist in the Danbooru tag list.
    Returns list of tags added.

    Tags are stored in the main DB; associations go to the appropriate DB.
    """
    if not prompt:
        return []

    valid_tags = get_danbooru_tags_sync()
    if not valid_tags:
        return []

    # Extract matching tags
    found_tags = extract_tags_from_prompt(prompt, valid_tags)
    if not found_tags:
        return []

    if directory_id:
        # Image is in a directory database
        dir_db = await directory_db_manager.get_session(directory_id)
        try:
            # Get existing tags for this image from directory DB
            existing_query = (
                select(Tag.name)
                .where(Tag.id.in_(
                    select(directory_image_tags.c.tag_id)
                    .where(directory_image_tags.c.image_id == image_id)
                ))
            )
            existing_result = await db.execute(existing_query)
            existing_tag_names = {row[0] for row in existing_result.all()}

            added_tags = []
            for tag_name in found_tags:
                normalized_name = tag_name.lower().replace(' ', '_')

                if normalized_name in existing_tag_names:
                    continue

                # Find or create tag in main DB
                tag_query = select(Tag).where(Tag.name == normalized_name)
                tag_result = await db.execute(tag_query)
                tag = tag_result.scalar_one_or_none()

                if not tag:
                    tag = Tag(
                        name=normalized_name,
                        category=TagCategory.general,
                        post_count=0
                    )
                    db.add(tag)
                    await db.flush()

                # Add association to directory DB
                await dir_db.execute(
                    directory_image_tags.insert().values(
                        image_id=image_id,
                        tag_id=tag.id,
                        confidence=confidence,
                        is_manual=False
                    )
                )

                tag.post_count += 1
                added_tags.append(normalized_name)
                existing_tag_names.add(normalized_name)

            if added_tags:
                await dir_db.commit()
                await db.commit()

            return added_tags
        finally:
            await dir_db.close()
    else:
        # Legacy: image is in main database
        existing_query = (
            select(Tag.name)
            .join(image_tags, image_tags.c.tag_id == Tag.id)
            .where(image_tags.c.image_id == image_id)
        )
        existing_result = await db.execute(existing_query)
        existing_tag_names = {row[0] for row in existing_result.all()}

        added_tags = []
        for tag_name in found_tags:
            normalized_name = tag_name.lower().replace(' ', '_')

            if normalized_name in existing_tag_names:
                continue

            tag_query = select(Tag).where(Tag.name == normalized_name)
            tag_result = await db.execute(tag_query)
            tag = tag_result.scalar_one_or_none()

            if not tag:
                tag = Tag(
                    name=normalized_name,
                    category=TagCategory.general,
                    post_count=0
                )
                db.add(tag)
                await db.flush()

            await db.execute(
                image_tags.insert().values(
                    image_id=image_id,
                    tag_id=tag.id,
                    confidence=confidence,
                    is_manual=False
                )
            )

            tag.post_count += 1
            added_tags.append(normalized_name)
            existing_tag_names.add(normalized_name)

        if added_tags:
            await db.commit()

        return added_tags


async def extract_and_save_metadata_with_tags(
    image_path: str,
    image_id: int,
    db: AsyncSession,
    comfyui_prompt_node_ids: list[str] = None,
    comfyui_negative_node_ids: list[str] = None,
    format_hint: str = 'auto',
    extract_tags: bool = True,
    directory_id: int = None
) -> tuple[ExtractionResult, list[str]]:
    """
    Extract metadata and optionally match prompt words to booru tags.
    Returns (extraction_result, list_of_added_tags).
    """
    result = await extract_and_save_metadata(
        image_path,
        image_id,
        db,
        comfyui_prompt_node_ids,
        comfyui_negative_node_ids,
        format_hint,
        directory_id=directory_id
    )

    added_tags = []
    if extract_tags and result.status == 'success' and result.metadata:
        prompt = result.metadata.prompt or ''
        if prompt:
            added_tags = await add_tags_from_prompt(
                image_id, prompt, db,
                directory_id=directory_id
            )

    return result, added_tags
