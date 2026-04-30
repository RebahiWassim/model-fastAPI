"""
predict.py — FastAPI router with prediction endpoints for all cancer types.

  POST /bone/predict    → BoneCancerModel  (EfficientNet-B0, 2 classes)
  POST /colon/predict   → ColonCancerModel (ViT-Base,        2 colon classes)
"""

import io
import base64
import logging

import numpy as np
import torch
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from app.core.config import (
    BONE_MODEL_PATH,
    COLON_MODEL_PATH,
    BONE_CLASS_NAMES,
    COLON_CLASS_NAMES,
    DEVICE,
    IMG_SIZE,
)
from app.model.model import load_bone_model, load_colon_model
from app.model.preprocessing import (
    GradCAM,
    draw_bbox_on_image,
    heatmap_to_bbox,
    preprocess_for_bone,
    preprocess_for_colon,
    validate_image,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# ─────────────────────────────────────────────────────────────────────────────
#  Lazy-loaded model singletons
# ─────────────────────────────────────────────────────────────────────────────

_bone_model  = None
_colon_model = None

ALLOWED_TYPES   = {"image/png", "image/jpeg", "image/bmp", "image/jpg"}


def init_bone_model():
    global _bone_model
    _bone_model = load_bone_model(BONE_MODEL_PATH, DEVICE)
    return _bone_model


def init_colon_model():
    global _colon_model
    _colon_model = load_colon_model(COLON_MODEL_PATH, DEVICE)
    return _colon_model


# ─────────────────────────────────────────────────────────────────────────────
#  Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _encode_image_b64(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _encode_pil_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _uncertainty(probs: np.ndarray, subset_indices=None) -> float:
    """
    Entropy-based uncertainty score in [0, 1].
    Pass *subset_indices* to restrict to a subset of classes
    (e.g. only the two colon classes out of 5).
    """
    p = probs[sorted(subset_indices)] if subset_indices else probs
    p = p / p.sum()
    p = np.clip(p, 1e-9, 1.0)
    entropy = -np.sum(p * np.log(p))
    return round(float(entropy / np.log(len(p))), 4)


def _validate_upload(file: UploadFile):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported file type")


# ─────────────────────────────────────────────────────────────────────────────
#  Bone Cancer endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/bone/predict", tags=["bone-cancer"])
async def predict_bone(file: UploadFile = File(...)):
    """
    Classify a histopathology image as **Cancer** or **Normal** using
    EfficientNet-B0. Returns confidence, uncertainty, and a GradCAM-annotated
    image when cancer is detected.
    """
    _validate_upload(file)

    try:
        contents = await file.read()
        tensor, vis_arr = preprocess_for_bone(contents, IMG_SIZE)
        tensor = tensor.to(DEVICE)

        _bone_model.eval()
        with torch.no_grad():
            probs = torch.softmax(_bone_model(tensor), dim=1)[0].cpu().numpy()

        class_index = int(np.argmax(probs))
        pred_class  = BONE_CLASS_NAMES[class_index]
        confidence  = round(float(probs[class_index]), 4)
        uncertainty = _uncertainty(probs)
        is_high_risk = class_index == 0          # index 0 == "Cancer"

        image_b64 = _encode_image_b64(vis_arr)

        # GradCAM bounding box for positive predictions
        if pred_class == "Cancer":
            try:
                cam  = GradCAM(_bone_model).generate(tensor.clone(), class_idx=0)
                bbox = heatmap_to_bbox(cam, threshold=0.4, img_size=IMG_SIZE)
                if bbox is not None:
                    image_b64 = _encode_image_b64(draw_bbox_on_image(vis_arr.copy(), bbox))
            except Exception:
                pass

        return JSONResponse(content={
            "status":      "success",
            "type":        "1D" ,
            "prediction":  pred_class,
            "class_index": class_index,
            "confidence":  confidence,
            "diagnostics": {
                "uncertainty_score": uncertainty,
                "is_high_risk":      is_high_risk,
                "all_probabilities": {
                    cls: round(float(p), 4)
                    for cls, p in zip(BONE_CLASS_NAMES, probs)
                },
            },
            "original_image": image_b64,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Bone prediction error")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
#  Colon Cancer endpoint
# ─────────────────────────────────────────────────────────────────────────────

# Only the two colon subtypes — lung indices are intentionally excluded
# Adjust these indices to match your COLON_CLASS_NAMES order:
#   e.g. 0 = "Colon Adenocarcinoma"  ← high risk
#        1 = "Colon Benign Tissue"
_COLON_INDICES = {0, 1}   # colon classes only (no lung)
_HIGH_RISK_IDX = 0        # Colon Adenocarcinoma


@router.post("/colon/predict", tags=["colon-cancer"])
async def predict_colon(file: UploadFile = File(...)):
    """
    Classify a histopathology image into one of the 2 colon cancer subtypes
    using ViT-Base. Lung classes are excluded from the response.
    Uncertainty is computed over the two colon classes only.
    """
    _validate_upload(file)

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        if not validate_image(image):
            raise HTTPException(status_code=422, detail="Invalid or corrupt image")

        tensor = preprocess_for_colon(image)
        tensor = tensor.to(DEVICE)

        with torch.no_grad():
            logits = _colon_model(tensor)
            probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

        # Restrict prediction to colon classes only
        colon_indices = sorted(_COLON_INDICES)
        colon_probs   = probs[colon_indices]
        colon_probs   = colon_probs / colon_probs.sum()   # re-normalize to 100 %

        local_idx   = int(np.argmax(colon_probs))         # index within colon subset
        class_index = colon_indices[local_idx]            # original model index
        pred_class  = COLON_CLASS_NAMES[class_index]
        confidence  = round(float(colon_probs[local_idx]), 4)
        uncertainty = _uncertainty(probs, subset_indices=_COLON_INDICES)
        is_high_risk = class_index == _HIGH_RISK_IDX

        image_b64 = _encode_pil_b64(image)

        return JSONResponse(content={
            "status":      "success",
            "type":        "1D",
            "prediction":  pred_class,
            "class_index": class_index,
            "confidence":  confidence,
            "diagnostics": {
                "uncertainty_score": uncertainty,
                "is_high_risk":      is_high_risk,
                "all_probabilities": {
                    COLON_CLASS_NAMES[i]: round(float(colon_probs[j]), 4)
                    for j, i in enumerate(colon_indices)
                },
            },
            "original_image": image_b64,
        })

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Colon prediction error")
        raise HTTPException(status_code=500, detail=str(e))
