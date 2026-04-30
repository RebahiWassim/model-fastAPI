"""
tests/test_api.py — Integration tests for the unified Cancer Detection API.
Run with:  pytest tests/test_api.py -v
"""

import io
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

client = TestClient(app)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_png_bytes(width: int = 256, height: int = 256) -> bytes:
    """Create a small in-memory RGB PNG for testing."""
    img = Image.new("RGB", (width, height), color=(128, 64, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Root / health ─────────────────────────────────────────────────────────────

def test_root():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "/api/v1/bone/predict"  in data["endpoints"]
    assert "/api/v1/colon/predict" in data["endpoints"]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "healthy"}


# ── Bone Cancer ───────────────────────────────────────────────────────────────

def test_bone_predict_returns_valid_schema():
    png = _make_png_bytes()
    r = client.post(
        "/api/v1/bone/predict",
        files={"file": ("test.png", png, "image/png")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"]      == "success"
    assert data["cancer_type"] == "bone"
    assert data["prediction"]  in {"Cancer", "Normal"}
    assert 0.0 <= data["confidence"] <= 1.0
    assert "uncertainty_score" in data["diagnostics"]
    assert "original_image"    in data


def test_bone_predict_rejects_unsupported_type():
    r = client.post(
        "/api/v1/bone/predict",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 415


# ── Colon Cancer ──────────────────────────────────────────────────────────────

def test_colon_predict_returns_valid_schema():
    png = _make_png_bytes()
    r = client.post(
        "/api/v1/colon/predict",
        files={"file": ("test.png", png, "image/png")},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"]      == "success"
    assert data["cancer_type"] == "colon"
    assert data["prediction"]  in {
        "Colon Adenocarcinoma",
        "Colon Benign",
        "Lung Adenocarcinoma",
        "Lung Benign",
        "Lung Squamous Cell",
    }
    assert 0.0 <= data["confidence"] <= 1.0
    assert "colon_probabilities" in data["diagnostics"]
    assert "original_image"      in data


def test_colon_predict_rejects_unsupported_type():
    r = client.post(
        "/api/v1/colon/predict",
        files={"file": ("test.gif", b"GIF89a", "image/gif")},
    )
    assert r.status_code == 415
