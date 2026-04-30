import logging

from fastapi import FastAPI

from app.api.routes import predict
from app.core.config import API_DESCRIPTION, API_TITLE, API_VERSION, DEVICE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
)

# ── Load both models at startup ──────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    predict.init_bone_model()
    predict.init_colon_model()
    logger.info("Both models loaded. Running on %s", DEVICE)


# ── Mount the shared router ──────────────────────────────────────────────────

app.include_router(predict.router, prefix="/api/v1")


# ── Health / root ────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "status":  "ok",
        "device":  str(DEVICE),
        "endpoints": [
            "/api/v1/bone/predict",
            "/api/v1/colon/predict",
        ],
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
