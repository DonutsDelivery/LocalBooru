"""
Age Detector Sidecar — Standalone FastAPI app.

Detects faces in images and estimates age/gender using InsightFace + MiVOLO v2.
No database access — returns detection results to the Rust backend.

Detection cascade:
  1. InsightFace face detection (preferred) OR OpenCV Haar Cascade (fallback)
  2. YOLO v8n body/person detection
  3. MiVOLO v2 age/gender estimation from face+body crops

Endpoints:
  GET  /health  → health check + backend availability
  POST /detect  → detect faces and estimate ages in an image
"""

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("age-detector")

# ─── Global model state (lazy-loaded) ────────────────────────────────────────

_face_detector = None
_face_detector_type = None  # 'insightface' or 'opencv'
_body_detector = None
_mivolo_model = None
_mivolo_processor = None
_models_loaded = False


def _load_models():
    """Lazy-load all detection models."""
    global _face_detector, _face_detector_type, _body_detector
    global _mivolo_model, _mivolo_processor, _models_loaded

    if _models_loaded:
        return

    # ── Face detector ─────────────────────────────────────────────────────
    try:
        from insightface.app import FaceAnalysis
        _face_detector = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        _face_detector.prepare(ctx_id=-1, det_size=(640, 640))
        _face_detector_type = "insightface"
        logger.info("InsightFace face detector loaded (CPU)")
    except Exception as e:
        logger.warning(f"InsightFace not available ({e}), trying OpenCV")
        try:
            import cv2
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            _face_detector = cv2.CascadeClassifier(cascade_path)
            _face_detector_type = "opencv"
            logger.info("OpenCV Haar Cascade face detector loaded")
        except Exception as e2:
            logger.error(f"No face detector available: {e2}")
            _face_detector = None
            _face_detector_type = None

    # ── Body detector (YOLO) ──────────────────────────────────────────────
    try:
        from ultralytics import YOLO
        _body_detector = YOLO("yolov8n.pt")
        logger.info("YOLO body detector loaded")
    except Exception as e:
        logger.warning(f"YOLO not available: {e}")
        _body_detector = None

    # ── MiVOLO v2 age/gender ─────────────────────────────────────────────
    try:
        import torch
        from transformers import AutoModelForImageClassification, AutoImageProcessor

        # Allow custom HuggingFace cache dir
        packages_dir = os.environ.get("LOCALBOORU_PACKAGES_DIR")
        if packages_dir:
            import sys
            hf_cache = os.path.join(packages_dir, "huggingface")
            os.environ["HF_HOME"] = hf_cache
            os.environ["TRANSFORMERS_CACHE"] = hf_cache
            if hf_cache not in sys.path:
                sys.path.insert(0, hf_cache)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        logger.info(f"Loading MiVOLO v2 on {device}...")

        _mivolo_model = AutoModelForImageClassification.from_pretrained(
            "iitolstykh/mivolo_v2",
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device).eval()

        _mivolo_processor = AutoImageProcessor.from_pretrained(
            "iitolstykh/mivolo_v2",
            trust_remote_code=True,
        )
        logger.info("MiVOLO v2 loaded successfully")
    except Exception as e:
        logger.error(f"MiVOLO not available: {e}")
        _mivolo_model = None
        _mivolo_processor = None

    _models_loaded = True


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_body_for_face(face_bbox, bodies, img_shape):
    """Find the body bounding box that best matches a face."""
    if not bodies:
        return None

    fx1, fy1, fx2, fy2 = face_bbox
    face_cx = (fx1 + fx2) / 2
    face_cy = (fy1 + fy2) / 2

    best_body = None
    best_score = -1

    for bx1, by1, bx2, by2 in bodies:
        if bx1 <= face_cx <= bx2:
            body_h = by2 - by1
            if by1 <= face_cy <= by1 + body_h * 0.4:
                body_area = (bx2 - bx1) * (by2 - by1)
                face_area = (fx2 - fx1) * (fy2 - fy1)
                ratio = body_area / face_area if face_area > 0 else 0
                if 3 < ratio < 100:
                    score = min(ratio, 50) / 50
                    if score > best_score:
                        best_score = score
                        best_body = (bx1, by1, bx2, by2)

    return best_body


def _estimate_age_gender_mivolo(face_crop, body_crop=None):
    """Use MiVOLO v2 to estimate age and gender from face + optional body crop."""
    import torch

    face_rgb = face_crop[:, :, ::-1].copy()  # BGR to RGB

    device = _mivolo_model.device
    dtype = _mivolo_model.dtype

    faces_input = _mivolo_processor(images=[face_rgb])["pixel_values"]
    faces_input = faces_input.to(dtype=dtype, device=device)

    if body_crop is not None:
        body_rgb = body_crop[:, :, ::-1].copy()
        body_input = _mivolo_processor(images=[body_rgb])["pixel_values"]
        body_input = body_input.to(dtype=dtype, device=device)
    else:
        body_input = torch.zeros_like(faces_input)

    with torch.no_grad():
        outputs = _mivolo_model(faces_input=faces_input, body_input=body_input)

    age = int(round(outputs.age_output[0].item()))
    gender = "M" if outputs.gender_class_idx[0].item() == 1 else "F"
    return age, gender


def _get_age_bracket(age: int) -> str:
    """Convert numeric age to bracket string."""
    if age < 13:
        return "child"
    elif age < 18:
        return "teen"
    elif age < 30:
        return "20s"
    elif age < 40:
        return "30s"
    elif age < 50:
        return "40s"
    elif age < 60:
        return "50s"
    elif age < 70:
        return "60s"
    else:
        return "elderly"


# ─── Core detection ──────────────────────────────────────────────────────────

def detect_ages(image_path: str) -> dict:
    """Detect faces and estimate ages in an image. Returns structured result."""
    import cv2

    if not _models_loaded:
        _load_models()

    if _face_detector is None:
        raise RuntimeError("No face detector available")

    # OpenCV detector alone needs MiVOLO for age estimation
    if _face_detector_type == "opencv" and _mivolo_model is None:
        raise RuntimeError("MiVOLO required for age estimation with OpenCV detector")

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Failed to load image: {image_path}")

    h, w = img.shape[:2]

    # Detect bodies with YOLO
    bodies = []
    if _body_detector is not None:
        try:
            yolo_results = _body_detector(img, verbose=False)
            for result in yolo_results:
                for box in result.boxes:
                    if int(box.cls[0]) == 0:  # person class
                        bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy().astype(int)
                        bodies.append((bx1, by1, bx2, by2))
        except Exception as e:
            logger.warning(f"YOLO detection failed: {e}")

    # Detect faces
    if _face_detector_type == "insightface":
        faces_raw = _face_detector.get(img)
        if not faces_raw:
            return {"num_faces": 0, "faces": [], "min_age": None, "max_age": None, "detected_ages": ""}
        face_boxes = [(face.bbox.astype(int), float(face.det_score), face) for face in faces_raw]
    elif _face_detector_type == "opencv":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        min_face = max(50, min(h, w) // 15)
        detections = _face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=8, minSize=(min_face, min_face)
        )
        if len(detections) == 0:
            return {"num_faces": 0, "faces": [], "min_age": None, "max_age": None, "detected_ages": ""}
        face_boxes = [((x, y, x + fw, y + fh), 0.9, None) for (x, y, fw, fh) in detections]
    else:
        raise RuntimeError(f"Unknown detector type: {_face_detector_type}")

    face_results = []
    for bbox_tuple, confidence, original_face in face_boxes:
        x1, y1, x2, y2 = bbox_tuple

        # Expand bbox slightly for context
        pad_x = int((x2 - x1) * 0.1)
        pad_y = int((y2 - y1) * 0.1)
        fx1, fy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
        fx2, fy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)

        face_crop = img[fy1:fy2, fx1:fx2]
        if face_crop.size == 0:
            continue

        # Find matching body
        body_crop = None
        body_bbox = _get_body_for_face((x1, y1, x2, y2), bodies, (h, w))
        if body_bbox is not None:
            bx1, by1, bx2, by2 = body_bbox
            bc = img[by1:by2, bx1:bx2]
            if bc.size > 0:
                body_crop = bc

        # Age/gender estimation
        if _mivolo_model is not None and _mivolo_processor is not None:
            try:
                age, gender = _estimate_age_gender_mivolo(face_crop, body_crop)
            except Exception as e:
                logger.warning(f"MiVOLO failed: {e}")
                if original_face is not None and hasattr(original_face, "age"):
                    age = int(original_face.age)
                    gender = "M" if original_face.gender == 1 else "F"
                else:
                    continue
        elif original_face is not None and hasattr(original_face, "age"):
            age = int(original_face.age)
            gender = "M" if original_face.gender == 1 else "F"
        else:
            continue

        face_results.append({
            "age": age,
            "gender": gender,
            "confidence": round(confidence, 3),
            "bbox": [int(fx1), int(fy1), int(fx2), int(fy2)],
            "age_bracket": _get_age_bracket(age),
        })

    # Sort by face area descending (primary subject first)
    face_results.sort(
        key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]),
        reverse=True,
    )

    ages = [f["age"] for f in face_results]
    return {
        "num_faces": len(face_results),
        "faces": face_results,
        "min_age": min(ages) if ages else None,
        "max_age": max(ages) if ages else None,
        "detected_ages": ",".join(str(a) for a in ages),
    }


# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="Age Detector Sidecar")


@app.get("/health")
async def health():
    has_insightface = _face_detector_type == "insightface" if _models_loaded else False
    has_mivolo = _mivolo_model is not None if _models_loaded else False
    has_opencv = _face_detector_type == "opencv" if _models_loaded else False

    return {
        "status": "ok",
        "backends": {
            "insightface": has_insightface,
            "mivolo": has_mivolo,
            "opencv": has_opencv,
        },
    }


class DetectRequest(BaseModel):
    file_path: str


@app.post("/detect")
async def detect(req: DetectRequest):
    if not os.path.exists(req.file_path):
        raise HTTPException(status_code=404, detail="Image file not found")

    try:
        result = detect_ages(req.file_path)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Detection failed for {req.file_path}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
