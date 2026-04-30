# Cancer Detection API

Unified FastAPI service merging the **bone-cancer-api** and **colon-cancer-AI** repositories into a single codebase.

## Models

| Cancer type | Architecture | Classes |
|-------------|-------------|---------|
| Bone        | EfficientNet-B0 | Cancer, Normal |
| Colon / Lung | ViT-Base/16  | Colon Adenocarcinoma, Colon Benign, Lung Adenocarcinoma, Lung Benign, Lung Squamous Cell |

## Project structure

```
cancer-detection-api/
├── app/
│   ├── api/routes/
│   │   └── predict.py          # All POST /bone/predict & /colon/predict endpoints
│   ├── core/
│   │   └── config.py           # Unified settings (paths, class names, device)
│   ├── model/
│   │   ├── model.py            # BoneCancerModel + ColonCancerModel definitions & loaders
│   │   └── preprocessing.py    # preprocess_for_bone, preprocess_for_colon, GradCAM, helpers
│   └── main.py                 # FastAPI app, startup, router mount
├── models/
│   ├── bone_cancer_model.pth   # EfficientNet-B0 weights (place here)
│   └── best_vit_lc25000.pth   # ViT-Base weights       (place here)
├── tests/
│   └── test_api.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/`                    | Status + endpoint list |
| GET  | `/health`              | Health check |
| POST | `/api/v1/bone/predict` | Bone cancer classification |
| POST | `/api/v1/colon/predict`| Colon/lung cancer classification |

## Quick start

```bash
# 1. Place model weights in models/
# 2. Install dependencies
pip install -r requirements.txt

# 3. Run
uvicorn app.main:app --host 0.0.0.0 --port 8000

# 4. Or with Docker
docker build -t cancer-api .
docker run -p 8000:8000 cancer-api
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `BONE_MODEL_PATH`  | `models/bone_cancer_model.pth`  | Path to bone model weights |
| `COLON_MODEL_PATH` | `models/best_vit_lc25000.pth`  | Path to colon model weights |
| `IMG_SIZE`         | `224`                            | Input image size (px) |
