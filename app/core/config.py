import os
import torch

# ─────────────────────────────────────────────
#  Device
# ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─────────────────────────────────────────────
#  Image
# ─────────────────────────────────────────────
IMG_SIZE = int(os.getenv("IMG_SIZE", 224))
IMAGE_SIZE = (IMG_SIZE, IMG_SIZE)

# ─────────────────────────────────────────────
#  Model paths
# ─────────────────────────────────────────────
BONE_MODEL_PATH  = os.getenv("BONE_MODEL_PATH",  "models/bone_cancer_model.pth")
COLON_MODEL_PATH = os.getenv("COLON_MODEL_PATH", "models/best_vit_lc25000.pth")

# ─────────────────────────────────────────────
#  Class names per cancer type
# ─────────────────────────────────────────────
BONE_CLASS_NAMES = ["Cancer", "Normal"]

COLON_CLASS_NAMES = [
    "Colon Adenocarcinoma",
    "Colon Benign",
    "Lung Adenocarcinoma",
    "Lung Benign",
    "Lung Squamous Cell",
]

# ─────────────────────────────────────────────
#  API metadata
# ─────────────────────────────────────────────
API_TITLE       = "Cancer Detection API"
API_VERSION     = "1.0.0"
API_DESCRIPTION = (
    "Unified API for bone-cancer detection (EfficientNet-B0) "
    "and colon/lung cancer detection (ViT-Base)."
)

# ─────────────────────────────────────────────
#  Settings object (backward-compat with bone repo)
# ─────────────────────────────────────────────
class Settings:
    MODEL_PATH  = BONE_MODEL_PATH
    IMG_SIZE    = IMG_SIZE
    MC_PASSES   = int(os.getenv("MC_PASSES", 10))
    DEVICE      = DEVICE
    CLASS_NAMES = BONE_CLASS_NAMES
    API_TITLE       = API_TITLE
    API_VERSION     = API_VERSION
    API_DESCRIPTION = API_DESCRIPTION

settings = Settings()
