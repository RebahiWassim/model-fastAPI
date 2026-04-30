"""
preprocessing.py — Image preprocessing utilities for all cancer models.

  preprocess_for_bone()   : bytes  → (tensor, vis_arr)  for EfficientNet
  preprocess_for_colon()  : PIL    → tensor              for ViT
  GradCAM                 : gradient-weighted class activation maps
  heatmap_to_bbox()       : CAM → bounding-box tuple
  draw_bbox_on_image()    : draw red rectangle on a numpy array
  validate_image()        : sanity-check a PIL image
"""

import io
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms


# ─────────────────────────────────────────────────────────────────────────────
#  Shared ImageNet normalisation
# ─────────────────────────────────────────────────────────────────────────────

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_VIT_MEAN = [0.5, 0.5, 0.5]
_VIT_STD  = [0.5, 0.5, 0.5]


# ─────────────────────────────────────────────────────────────────────────────
#  Bone Cancer  —  EfficientNet-B0 preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _bone_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])


def preprocess_for_bone(
    image_bytes: bytes,
    img_size: int = 224,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Decode *image_bytes*, apply EfficientNet-style preprocessing.

    Returns:
        tensor  : float32 [1, 3, H, W] ready for BoneCancerModel.
        vis_arr : uint8   [H, W, 3]    for visualisation / GradCAM overlay.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    vis_arr = np.array(img.resize((img_size, img_size))).copy()
    tensor = _bone_transform(img_size)(img).unsqueeze(0)
    return tensor, vis_arr


# ─────────────────────────────────────────────────────────────────────────────
#  Colon / Lung Cancer  —  ViT-Base preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_for_colon(
    image: Union[Image.Image, np.ndarray],
    target_size: Tuple[int, int] = (224, 224),
    vit_normalisation: bool = False,
) -> torch.Tensor:
    """
    Resize and normalise *image* for ColonCancerModel (ViT-Base).

    Args:
        image            : PIL Image or numpy uint8 array.
        target_size      : (H, W) – default (224, 224).
        vit_normalisation: Use ViT-style [0.5, 0.5, 0.5] stats instead of
                           ImageNet stats (set True when re-training with those
                           stats).

    Returns:
        Float32 tensor [1, 3, H, W].
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(
            image if image.dtype == np.uint8 else (image * 255).astype(np.uint8)
        )

    mean, std = (_VIT_MEAN, _VIT_STD) if vit_normalisation else (_IMAGENET_MEAN, _IMAGENET_STD)

    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return transform(image).unsqueeze(0)


def validate_image(image: Image.Image) -> bool:
    """Return True when *image* is a valid RGB-compatible image ≥ 32 px."""
    try:
        if image.mode not in {"RGB", "L", "RGBA"}:
            return False
        if min(image.size) < 32:
            return False
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
#  GradCAM  (bone cancer)
# ─────────────────────────────────────────────────────────────────────────────

class GradCAM:
    """Gradient-weighted Class Activation Maps for EfficientNet-B0."""

    def __init__(self, model):
        self.model = model
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        target_layer = model.backbone.features[-1]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        """Compute the normalised CAM for *class_idx*."""
        tensor = tensor.to(next(self.model.parameters()).device)
        tensor.requires_grad_(True)

        output = self.model(tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1).squeeze(0)
        cam = F.relu(cam).cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        cam = (
            (cam - cam_min) / (cam_max - cam_min)
            if cam_max - cam_min > 1e-8
            else np.zeros_like(cam)
        )
        return cam


# ─────────────────────────────────────────────────────────────────────────────
#  Bounding-box helpers
# ─────────────────────────────────────────────────────────────────────────────

def heatmap_to_bbox(
    cam: np.ndarray,
    threshold: float = 0.4,
    img_size: int = 224,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Threshold a GradCAM heatmap and return the bounding box of the
    largest connected component as *(x0, y0, x1, y1)*, or None.
    """
    binary = (cam >= threshold).astype(np.uint8) * 255
    resized = cv2.resize(binary, (img_size, img_size))
    contours, _ = cv2.findContours(resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return (x, y, x + w, y + h)


def draw_bbox_on_image(
    vis_arr: np.ndarray,
    bbox: Tuple[int, int, int, int],
) -> np.ndarray:
    """Draw a 3-px red bounding rectangle on *vis_arr* (uint8 HxWx3)."""
    img_pil = Image.fromarray(vis_arr.astype(np.uint8))
    draw = ImageDraw.Draw(img_pil)
    x0, y0, x1, y1 = bbox
    for offset in range(3):
        draw.rectangle(
            [x0 - offset, y0 - offset, x1 + offset, y1 + offset],
            outline=(255, 0, 0),
        )
    return np.array(img_pil)
