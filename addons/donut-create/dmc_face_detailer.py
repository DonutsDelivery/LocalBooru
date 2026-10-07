"""Owned, request-local routing around the pinned DonutFaceDetailer.

The node resolves DonutNodes only after ComfyUI has registered all packs. It
never patches its classes or shares detector/conditioning state between runs.
"""
import copy
import inspect
import json
import math
import sys

ORDINALS = ["1st", "2nd", "3rd", *[f"{index}th" for index in range(4, 17)]]


def face_prompts(value):
    if not isinstance(value, str) or len(value) > 1024 * 1024:
        raise ValueError("Face descriptions must be a JSON array of at most 16 characters.")
    def unique(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("Duplicate face description field.")
            result[key] = item
        return result
    try:
        rows = json.loads(value, object_pairs_hook=unique)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("Face descriptions must be a JSON array.") from error
    if not isinstance(rows, list) or len(rows) > 16:
        raise ValueError("Face descriptions must be a JSON array of at most 16 characters.")
    names = set()
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {"character", "position", "face_prompt"}
                or not isinstance(row["character"], str) or not row["character"].strip() or len(row["character"]) > 200
                or not isinstance(row["position"], str) or row["position"] not in ["left", "center", "right", *ORDINALS]
                or not isinstance(row["face_prompt"], str) or not row["face_prompt"].strip() or len(row["face_prompt"]) > 8000):
            raise ValueError("Each face description needs a character, resolved position and face-only prompt.")
        name = row["character"].strip().casefold()
        if name in names:
            raise ValueError("Face descriptions must identify distinct characters.")
        names.add(name)
    if rows:
        positions = [row["position"] for row in rows]
        family = ["left", "center", "right"] if positions[0] in {"left", "center", "right"} else ORDINALS
        if any(position not in family for position in positions) or any(
                family.index(left) >= family.index(right) for left, right in zip(positions, positions[1:])):
            raise ValueError("Face positions must use one strictly increasing left-to-right order.")
    return rows


class SingleFaceDetector:
    def __init__(self, dimensions, segment):
        self.dimensions, self.segment = dimensions, segment

    def setAux(self, value):
        pass

    def detect(self, *args, **kwargs):
        return self.dimensions, [self.segment]


def ordered_faces(module, image, inputs, expected):
    detector = inputs["bbox_detector"]
    detector.setAux("face")
    try:
        segs = detector.detect(image, inputs["bbox_threshold"], inputs["bbox_dilation"],
                               inputs["bbox_crop_factor"], inputs["drop_size"], detailer_hook=None)
    finally:
        detector.setAux(None)
    if inputs.get("sam_model_opt") is not None:
        mask = module.core.make_sam_mask(inputs["sam_model_opt"], segs, image,
            *[inputs[key] for key in ("sam_detection_hint", "sam_dilation", "sam_threshold",
                                     "sam_bbox_expansion", "sam_mask_hint_threshold", "sam_mask_hint_use_negative")])
        segs = module.core.segs_bitwise_and_mask(segs, mask)
    elif inputs.get("segm_detector_opt") is not None:
        segmenter = inputs["segm_detector_opt"]
        masks = segmenter.detect(image, inputs["bbox_threshold"], inputs["bbox_dilation"],
                                 inputs["bbox_crop_factor"], inputs["drop_size"])
        segs = masks if getattr(segmenter, "override_bbox_by_segm", False) else module.core.segs_bitwise_and_mask(
            segs, module.core.segs_to_combined_mask(masks))
    faces = []
    for index, seg in enumerate(segs[1]):
        mask = seg.cropped_mask
        if mask is None or not (module.torch.count_nonzero(mask) if module.torch.is_tensor(mask) else module.np.count_nonzero(mask)):
            continue
        bbox = seg.bbox
        if (len(bbox) != 4 or not all(math.isfinite(float(value)) for value in bbox)
                or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
            raise ValueError("A detected face has invalid coordinates; character routing was not applied.")
        faces.append(((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2, index, seg))
    if len(faces) != expected:
        raise ValueError(f"Face routing mismatch: expected {expected} characters but detected {len(faces)} faces. No character prompts were applied.")
    return segs[0], [item[3] for item in sorted(faces, key=lambda item: item[:3])]


class DMCFaceDetailer:
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "DETAILER_PIPE", "IMAGE")
    RETURN_NAMES = ("image", "cropped_refined", "cropped_enhanced_alpha", "mask", "detailer_pipe", "cnet_images")
    OUTPUT_IS_LIST = (False, True, True, False, False, True)
    FUNCTION = "doit"
    CATEGORY = "donut/create"

    @staticmethod
    def upstream():
        import nodes
        return nodes.NODE_CLASS_MAPPINGS["DonutFaceDetailer"]

    @classmethod
    def INPUT_TYPES(cls):
        result = copy.deepcopy(cls.upstream().INPUT_TYPES())
        result.setdefault("optional", {})["face_prompts_json"] = ("STRING", {"default": "[]", "multiline": True, "dynamicPrompts": False})
        return result

    def doit(self, *args, face_prompts_json="[]", **kwargs):
        rows = face_prompts(face_prompts_json)
        upstream = self.upstream()
        if len(rows) <= 1:
            return upstream().doit(*args, **kwargs)
        bound = inspect.signature(upstream().doit).bind(*args, **kwargs)
        bound.apply_defaults()
        inputs = dict(bound.arguments)
        inputs.update(inputs.pop("nag_options", {}))
        if inputs.get("edit_mode") or any(inputs.get(key) is not None for key in ("face_reference", "face_reference_b")):
            raise ValueError("Multi-face identity editing and identity references are not supported. Use a single face description or disable identity editing.")
        if inputs.get("wildcard", "").strip() or inputs.get("detailer_hook") is not None:
            raise ValueError("Multi-face routing does not support detailer wildcards or custom hooks that can replace character conditioning.")
        if inputs["max_faces"] < len(rows):
            raise ValueError("Maximum faces is below the number of character descriptions. No character prompts were applied.")
        module = sys.modules[upstream.__module__]
        module._ensure_impact()
        module.offload_model_for_auxiliary_stage(inputs["model"], module.comfy.model_management)
        images = [image.unsqueeze(0) for image in inputs["image"]]
        # Validate every batch image before the first refinement pass. Never
        # truncate detections by size, which would shift positional identities.
        detections = [ordered_faces(module, image, inputs, len(rows)) for image in images]
        import nodes
        positives = []
        for row in rows:
            fresh = nodes.CLIPTextEncode().encode(inputs["clip"], row["face_prompt"])[0]
            fresh = module.reapply_edit_variance(fresh, inputs["positive"])
            positives.append(module.prepare_positive_conditioning_taps(inputs["model"], fresh))
        forwarded = {key: value for key, value in inputs.items() if key != "image"}
        for old, new in (("guide_size_for", "guide_size_for_bbox"), ("noise_mask", "noise_mask_enabled"),
                         ("wildcard", "wildcard_opt"), ("segm_detector_opt", "segm_detector")):
            forwarded[new] = forwarded.pop(old)
        forwarded.update(resolution=inputs["resolution"] ** 2, sam_model_opt=None, segm_detector=None, max_faces=1)
        outputs, masks, crops, alphas, controls = [], [], [], [], []
        for image_index, (image, (dimensions, segments)) in enumerate(zip(images, detections)):
            for face_index, (seg, row, positive) in enumerate(zip(segments, rows, positives)):
                seed = inputs["seed"] + image_index
                if inputs.get("vary_seed_per_face"):
                    seed = (seed + face_index) & 0xffffffffffffffff
                result = upstream.enhance_face(image, **{**forwarded, "positive": positive,
                    "edit_prompt": row["face_prompt"], "seed": seed, "vary_seed_per_face": False,
                    "bbox_detector": SingleFaceDetector(dimensions, seg)})
                image = result[0]
                crops.extend(result[1]); alphas.extend(result[2]); controls.extend(result[4])
            outputs.append(image)
            mask = module.core.segs_to_combined_mask((dimensions, segments))
            masks.append(mask.unsqueeze(0) if mask.ndim == 2 else mask)
        if not crops:
            crops = [module.impact_utils.empty_pil_tensor()]
        if not alphas:
            alphas = [module.impact_utils.empty_pil_tensor()]
        if not controls:
            controls = [module.impact_utils.empty_pil_tensor()]
        pipe = tuple(inputs.get(key) for key in ("model", "clip", "vae", "positive", "negative", "wildcard",
            "bbox_detector", "segm_detector_opt", "sam_model_opt", "detailer_hook")) + (None,) * 4
        return module.torch.cat(outputs, dim=0), crops, alphas, module.torch.cat(masks, dim=0), pipe, controls


NODE_CLASS_MAPPINGS = {"DMCFaceDetailer": DMCFaceDetailer}
NODE_DISPLAY_NAME_MAPPINGS = {"DMCFaceDetailer": "DMC Face Detailer · Ordered Characters"}
