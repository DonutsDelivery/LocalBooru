"""
Age detection service using MiVOLO v2 (state-of-the-art)
Uses InsightFace for face detection, MiVOLO for age/gender estimation
"""
import logging
from pathlib import Path
from typing import Optional
import numpy as np

logger = logging.getLogger(__name__)

# Global model cache
_face_detector = None
_face_detector_type = None  # 'insightface' or 'opencv'
_body_detector = None
_mivolo_model = None
_mivolo_processor = None
_models_loaded = False


def get_models():
    """Get or initialize the face detector, body detector, and MiVOLO model (lazy loading)."""
    global _face_detector, _face_detector_type, _body_detector, _mivolo_model, _mivolo_processor, _models_loaded

    if _models_loaded:
        return _face_detector, _face_detector_type, _body_detector, _mivolo_model, _mivolo_processor

    # Try InsightFace first (best quality), fall back to OpenCV (no compilation needed)
    try:
        from insightface.app import FaceAnalysis
        _face_detector = FaceAnalysis(
            name='buffalo_l',
            providers=['CPUExecutionProvider']
        )
        _face_detector.prepare(ctx_id=-1, det_size=(640, 640))
        _face_detector_type = 'insightface'
        logger.info("InsightFace face detector loaded (CPU mode)")
    except Exception as e:
        logger.warning(f"InsightFace not available ({e}), using OpenCV face detection")
        # Fall back to OpenCV Haar Cascade (built-in, no compilation needed)
        try:
            import cv2
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            _face_detector = cv2.CascadeClassifier(cascade_path)
            _face_detector_type = 'opencv'
            logger.info("OpenCV Haar Cascade face detector loaded")
        except Exception as e2:
            logger.error(f"Failed to load any face detector: {e2}")
            _face_detector = None
            _face_detector_type = None

    try:
        # Load YOLO for body/person detection
        from ultralytics import YOLO
        from .model_downloader import get_model_path, is_model_available, download_model
        import asyncio

        # Get the model path from our model storage
        yolo_model_path = get_model_path("yolov8n") / "yolov8n.pt"

        # If not available, try to download synchronously
        if not is_model_available("yolov8n"):
            logger.info("YOLO model not found, downloading...")
            try:
                # Run async download in sync context
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # We're already in an async context, schedule it
                    asyncio.create_task(download_model("yolov8n"))
                    # Fall back to ultralytics download for now
                    _body_detector = YOLO('yolov8n.pt')
                else:
                    loop.run_until_complete(download_model("yolov8n"))
                    _body_detector = YOLO(str(yolo_model_path))
            except Exception as download_error:
                logger.warning(f"Failed to download YOLO via model_downloader: {download_error}, falling back to ultralytics")
                _body_detector = YOLO('yolov8n.pt')  # Let ultralytics handle download
        else:
            _body_detector = YOLO(str(yolo_model_path))

        logger.info("YOLO body detector loaded")
    except Exception as e:
        logger.error(f"Failed to load YOLO: {e}")
        _body_detector = None

    try:
        # Load MiVOLO v2 for age/gender estimation
        import torch
        from transformers import AutoModelForImageClassification, AutoImageProcessor
        import os

        # Set HuggingFace cache to our packages dir so custom code is found
        packages_dir = os.environ.get('LOCALBOORU_PACKAGES_DIR')
        if packages_dir:
            hf_cache = os.path.join(packages_dir, 'huggingface')
            os.environ['HF_HOME'] = hf_cache
            os.environ['TRANSFORMERS_CACHE'] = hf_cache
            # Add to Python path so custom model code can be imported
            if hf_cache not in sys.path:
                sys.path.insert(0, hf_cache)
            print(f"[AgeDetector] HF cache set to: {hf_cache}", flush=True)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        print(f"[AgeDetector] Loading MiVOLO on {device}...", flush=True)

        _mivolo_model = AutoModelForImageClassification.from_pretrained(
            "iitolstykh/mivolo_v2",
            trust_remote_code=True,
            torch_dtype=dtype
        ).to(device).eval()

        _mivolo_processor = AutoImageProcessor.from_pretrained(
            "iitolstykh/mivolo_v2",
            trust_remote_code=True
        )
        print(f"[AgeDetector] MiVOLO v2 loaded successfully!", flush=True)
        logger.info(f"MiVOLO v2 model loaded ({device})")
    except Exception as e:
        print(f"[AgeDetector] ERROR loading MiVOLO: {e}", flush=True)
        import traceback
        traceback.print_exc()
        logger.error(f"Failed to load MiVOLO: {e}")
        _mivolo_model = None
        _mivolo_processor = None

    _models_loaded = True
    return _face_detector, _face_detector_type, _body_detector, _mivolo_model, _mivolo_processor


def _get_body_for_face(face_bbox, bodies, img_shape):
    """Find the body bounding box that contains or best matches a face.

    Args:
        face_bbox: (x1, y1, x2, y2) of the face
        bodies: List of body bounding boxes from YOLO
        img_shape: (height, width) of the image

    Returns:
        Body bounding box (x1, y1, x2, y2) or None
    """
    if not bodies:
        return None

    fx1, fy1, fx2, fy2 = face_bbox
    face_center_x = (fx1 + fx2) / 2
    face_center_y = (fy1 + fy2) / 2

    best_body = None
    best_score = -1

    for body in bodies:
        bx1, by1, bx2, by2 = body

        # Check if face center is within or near the body bbox
        # Face should be in upper portion of body
        if bx1 <= face_center_x <= bx2:
            # Face center X is within body
            body_height = by2 - by1
            # Face should be in top 40% of body
            if by1 <= face_center_y <= by1 + body_height * 0.4:
                # Calculate overlap/containment score
                body_area = (bx2 - bx1) * (by2 - by1)
                face_area = (fx2 - fx1) * (fy2 - fy1)
                # Prefer bodies that are reasonably sized relative to face
                ratio = body_area / face_area if face_area > 0 else 0
                # Good ratio is 5-50x face size
                if 3 < ratio < 100:
                    score = min(ratio, 50) / 50  # Normalize
                    if score > best_score:
                        best_score = score
                        best_body = body

    return best_body


class FaceInfo:
    """Information about a detected face."""
    def __init__(self, age: int, gender: str, confidence: float, bbox: tuple):
        self.age = age
        self.gender = gender  # 'M' or 'F'
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)

    def to_dict(self) -> dict:
        return {
            'age': self.age,
            'gender': self.gender,
            'confidence': round(self.confidence, 3),
            'bbox': [int(x) for x in self.bbox]
        }


class AgeDetectionResult:
    """Result of age detection on an image."""
    def __init__(self, faces: list[FaceInfo]):
        self.faces = faces
        self.num_faces = len(faces)

    @property
    def ages(self) -> list[int]:
        """List of all detected ages."""
        return [f.age for f in self.faces]

    @property
    def min_age(self) -> Optional[int]:
        """Youngest detected age."""
        return min(self.ages) if self.ages else None

    @property
    def max_age(self) -> Optional[int]:
        """Oldest detected age."""
        return max(self.ages) if self.ages else None

    @property
    def avg_age(self) -> Optional[float]:
        """Average detected age."""
        return sum(self.ages) / len(self.ages) if self.ages else None

    def get_age_tags(self) -> list[str]:
        """Generate age-related tags based on detected faces."""
        tags = []

        if not self.faces:
            return tags

        # Add face count tag
        if self.num_faces == 1:
            tags.append('1_person')
        elif self.num_faces == 2:
            tags.append('2_people')
        elif self.num_faces >= 3:
            tags.append('multiple_people')

        # Add age bracket tags for each unique bracket
        age_brackets = set()
        for face in self.faces:
            bracket = self._get_age_bracket(face.age)
            age_brackets.add(bracket)

        for bracket in age_brackets:
            tags.append(f'age:{bracket}')

        # Add gender tags
        genders = [f.gender for f in self.faces]
        if 'M' in genders:
            tags.append('male')
        if 'F' in genders:
            tags.append('female')

        return tags

    def _get_age_bracket(self, age: int) -> str:
        """Convert numeric age to bracket tag."""
        if age < 13:
            return 'child'
        elif age < 18:
            return 'teen'
        elif age < 30:
            return '20s'
        elif age < 40:
            return '30s'
        elif age < 50:
            return '40s'
        elif age < 60:
            return '50s'
        elif age < 70:
            return '60s'
        else:
            return 'elderly'

    def to_dict(self) -> dict:
        return {
            'num_faces': self.num_faces,
            'faces': [f.to_dict() for f in self.faces],
            'min_age': self.min_age,
            'max_age': self.max_age,
            'avg_age': round(self.avg_age, 1) if self.avg_age else None,
            'age_tags': self.get_age_tags()
        }


# Tags that indicate realistic/photorealistic content
REALISTIC_TAGS = {
    'realistic',
    'photorealistic',
    'photo_(medium)',
    'photography_(artwork)',
    'real_life',
    'real_person',
    '3d',
    '3d_render',
    'hyperrealistic',
}


def should_detect_age(tags: list[str]) -> bool:
    """Check if image should have age detection based on its tags."""
    tag_names = {t.lower() if isinstance(t, str) else t.name.lower() for t in tags}
    return bool(tag_names & REALISTIC_TAGS)


def _estimate_age_gender_mivolo(face_crop, mivolo_model, mivolo_processor, body_crop=None):
    """Use MiVOLO to estimate age and gender from face (and optionally body) crop."""
    import torch

    # Convert BGR to RGB numpy array (MiVOLO expects list of numpy arrays)
    if isinstance(face_crop, np.ndarray):
        face_rgb = face_crop[:, :, ::-1].copy()  # BGR to RGB
    else:
        face_rgb = np.array(face_crop)

    # Process face with MiVOLO processor - expects list of numpy arrays
    device = mivolo_model.device
    dtype = mivolo_model.dtype

    faces_input = mivolo_processor(images=[face_rgb])["pixel_values"]
    faces_input = faces_input.to(dtype=dtype, device=device)

    # Process body if provided, otherwise use zeros placeholder
    # MiVOLO requires both face and body inputs (concatenates them)
    if body_crop is not None:
        if isinstance(body_crop, np.ndarray):
            body_rgb = body_crop[:, :, ::-1].copy()  # BGR to RGB
        else:
            body_rgb = np.array(body_crop)
        body_input = mivolo_processor(images=[body_rgb])["pixel_values"]
        body_input = body_input.to(dtype=dtype, device=device)
    else:
        # Create zero tensor as placeholder when no body detected
        body_input = torch.zeros_like(faces_input)

    with torch.no_grad():
        outputs = mivolo_model(faces_input=faces_input, body_input=body_input)

    # Extract age and gender from outputs
    # MiVOLO v2 uses age_output and gender_class_idx
    age = int(round(outputs.age_output[0].item()))
    gender = 'M' if outputs.gender_class_idx[0].item() == 1 else 'F'

    return age, gender


def is_age_detection_enabled() -> bool:
    """Check if age detection is enabled in settings"""
    try:
        from ..routers.settings import get_setting, AGE_DETECTION_ENABLED, are_required_deps_installed
        enabled = get_setting(AGE_DETECTION_ENABLED, "false") == "true"
        if not enabled:
            return False
        # Also check required dependencies are installed
        return are_required_deps_installed()
    except Exception as e:
        logger.warning(f"Failed to check age detection setting: {e}")
        return False


async def detect_ages(image_path: str | Path) -> Optional[AgeDetectionResult]:
    """
    Detect faces and estimate ages in an image.
    Uses YOLO for body detection + MiVOLO for face+body age estimation.

    Args:
        image_path: Path to the image file

    Returns:
        AgeDetectionResult with detected faces, or None if detection fails
    """
    # Check if age detection is enabled
    if not is_age_detection_enabled():
        logger.debug("Age detection is disabled")
        return None

    import cv2

    face_detector, detector_type, body_detector, mivolo_model, mivolo_processor = get_models()

    if face_detector is None:
        logger.warning("Face detector not available")
        return None

    # If using OpenCV face detection (no InsightFace), we NEED MiVOLO for age estimation
    # Otherwise we'd just save fake data
    if detector_type == 'opencv' and mivolo_model is None:
        print("[AgeDetector] MiVOLO not available with OpenCV detector, skipping age detection", flush=True)
        logger.warning("MiVOLO not available, cannot estimate ages with OpenCV detector")
        return None

    try:
        # Load image with OpenCV
        image_path = Path(image_path)
        if not image_path.exists():
            logger.warning(f"Image not found: {image_path}")
            return None

        img = cv2.imread(str(image_path))
        if img is None:
            logger.warning(f"Failed to load image: {image_path}")
            return None

        h, w = img.shape[:2]

        # Detect bodies using YOLO (class 0 = person)
        bodies = []
        if body_detector is not None:
            try:
                yolo_results = body_detector(img, verbose=False)
                for result in yolo_results:
                    for box in result.boxes:
                        if int(box.cls[0]) == 0:  # person class
                            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
                            bodies.append((bx1, by1, bx2, by2))
                logger.debug(f"YOLO detected {len(bodies)} bodies")
            except Exception as e:
                logger.warning(f"YOLO body detection failed: {e}")

        # Detect faces using available detector
        if detector_type == 'insightface':
            faces = face_detector.get(img)
            if not faces:
                logger.debug(f"No faces detected in {image_path}")
                return AgeDetectionResult([])
            # Convert InsightFace results to common format
            face_boxes = [(face.bbox.astype(int), float(face.det_score), face) for face in faces]
        elif detector_type == 'opencv':
            # OpenCV Haar Cascade detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            detections = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            if len(detections) == 0:
                logger.debug(f"No faces detected in {image_path}")
                return AgeDetectionResult([])
            # Convert to common format: (bbox, confidence, original_face)
            face_boxes = [((x, y, x+w, y+h), 0.9, None) for (x, y, w, h) in detections]
        else:
            logger.warning(f"Unknown face detector type: {detector_type}")
            return None

        face_infos = []
        for bbox_tuple, confidence, original_face in face_boxes:
            x1, y1, x2, y2 = bbox_tuple

            # Expand face bbox slightly for better context
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            fx1 = max(0, x1 - pad_x)
            fy1 = max(0, y1 - pad_y)
            fx2 = min(w, x2 + pad_x)
            fy2 = min(h, y2 + pad_y)

            # Crop face
            face_crop = img[fy1:fy2, fx1:fx2]

            if face_crop.size == 0:
                continue

            # Try to find matching body for this face
            body_crop = None
            body_bbox = _get_body_for_face((x1, y1, x2, y2), bodies, (h, w))
            if body_bbox is not None:
                bx1, by1, bx2, by2 = body_bbox
                body_crop = img[by1:by2, bx1:bx2]
                if body_crop.size == 0:
                    body_crop = None

            # Use MiVOLO for age/gender if available
            if mivolo_model is not None and mivolo_processor is not None:
                try:
                    age, gender = _estimate_age_gender_mivolo(
                        face_crop, mivolo_model, mivolo_processor, body_crop=body_crop
                    )
                except Exception as e:
                    logger.warning(f"MiVOLO failed: {e}")
                    # Fall back to InsightFace if available
                    if original_face is not None and hasattr(original_face, 'age'):
                        age = int(original_face.age)
                        gender = 'M' if original_face.gender == 1 else 'F'
                    else:
                        # No age estimation available with OpenCV
                        age = 25  # Default estimate
                        gender = 'U'
            elif original_face is not None and hasattr(original_face, 'age'):
                # Fall back to InsightFace age/gender
                age = int(original_face.age)
                gender = 'M' if original_face.gender == 1 else 'F'
            else:
                # OpenCV doesn't provide age/gender, MiVOLO not available
                # Skip this face - don't save fake data
                print("[AgeDetector] WARNING: No age model available, skipping face", flush=True)
                logger.warning("No age estimation model available, skipping face")
                continue

            face_infos.append(FaceInfo(
                age=age,
                gender=gender,
                confidence=confidence,
                bbox=(fx1, fy1, fx2, fy2)
            ))

        # Sort by face size (largest first, assuming primary subject)
        face_infos.sort(
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )

        result = AgeDetectionResult(face_infos)
        logger.debug(f"Detected {result.num_faces} faces in {image_path}: ages {result.ages}")
        return result

    except Exception as e:
        logger.error(f"Age detection failed for {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


async def detect_ages_from_array(img_array: np.ndarray) -> Optional[AgeDetectionResult]:
    """
    Detect faces and estimate ages from a numpy array.
    Uses YOLO for body detection + MiVOLO for face+body age estimation.

    Args:
        img_array: BGR numpy array (OpenCV format)

    Returns:
        AgeDetectionResult with detected faces, or None if detection fails
    """
    face_detector, detector_type, body_detector, mivolo_model, mivolo_processor = get_models()

    if face_detector is None or detector_type != 'insightface':
        # This function uses InsightFace-specific API (face_detector.get)
        return None

    try:
        h, w = img_array.shape[:2]

        # Detect bodies using YOLO
        bodies = []
        if body_detector is not None:
            try:
                yolo_results = body_detector(img_array, verbose=False)
                for result in yolo_results:
                    for box in result.boxes:
                        if int(box.cls[0]) == 0:  # person class
                            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
                            bodies.append((bx1, by1, bx2, by2))
            except Exception as e:
                logger.warning(f"YOLO body detection failed: {e}")

        faces = face_detector.get(img_array)

        if not faces:
            return AgeDetectionResult([])

        face_infos = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox
            confidence = float(face.det_score)

            # Expand bbox slightly
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            fx1 = max(0, x1 - pad_x)
            fy1 = max(0, y1 - pad_y)
            fx2 = min(w, x2 + pad_x)
            fy2 = min(h, y2 + pad_y)

            face_crop = img_array[fy1:fy2, fx1:fx2]

            if face_crop.size == 0:
                continue

            # Try to find matching body
            body_crop = None
            body_bbox = _get_body_for_face((x1, y1, x2, y2), bodies, (h, w))
            if body_bbox is not None:
                bx1, by1, bx2, by2 = body_bbox
                body_crop = img_array[by1:by2, bx1:bx2]
                if body_crop.size == 0:
                    body_crop = None

            if mivolo_model is not None and mivolo_processor is not None:
                try:
                    age, gender = _estimate_age_gender_mivolo(
                        face_crop, mivolo_model, mivolo_processor, body_crop=body_crop
                    )
                except Exception as e:
                    logger.warning(f"MiVOLO failed: {e}")
                    age = int(face.age)
                    gender = 'M' if face.gender == 1 else 'F'
            else:
                age = int(face.age)
                gender = 'M' if face.gender == 1 else 'F'

            face_infos.append(FaceInfo(
                age=age,
                gender=gender,
                confidence=confidence,
                bbox=(fx1, fy1, fx2, fy2)
            ))

        face_infos.sort(
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True
        )

        return AgeDetectionResult(face_infos)

    except Exception as e:
        logger.error(f"Age detection from array failed: {e}")
        return None
