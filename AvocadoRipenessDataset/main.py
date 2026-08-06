# AvocadoRipenessDataset/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator
from contextlib import asynccontextmanager
from pathlib import Path
import joblib
import uvicorn
import numpy as np

BASE_DIR = Path(__file__).parent

VALID_COLORS = {"dark green", "green", "purple"}
COLOR_CATEGORIES = ["dark green", "green", "purple"]

RIPENESS_LABELS = {
    "unripe": "Unripe",
    "ripe": "Ripe",
    "overripe": "Overripe"
}


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model  = joblib.load(BASE_DIR / "model_rf_AvocadoRipenessDataset.pkl")
    app.state.scaler = joblib.load(BASE_DIR / "scaler_AvocadoRipenessDataset.pkl")
    yield


avocado_app = FastAPI(title="Avocado Ripeness Classifier", lifespan=lifespan)


# ── Schema ─────────────────────────────────────────────────────────────────────
class AvocadoSchema(BaseModel):
    firmness:       float
    hue:            float
    saturation:     float
    brightness:     float
    color_category: str
    sound_db:       float
    weight_g:       float
    size_cm3:       float

    @field_validator("color_category")
    @classmethod
    def validate_color(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in VALID_COLORS:
            raise ValueError(f"color_category должен быть одним из: {VALID_COLORS}")
        return v


# ── Utils ──────────────────────────────────────────────────────────────────────
def build_features(avocado: AvocadoSchema) -> np.ndarray:
    numerical = [
        avocado.firmness,
        avocado.hue,
        avocado.saturation,
        avocado.brightness,
        avocado.sound_db,
        avocado.weight_g,
        avocado.size_cm3,
    ]
    ohe = [1 if avocado.color_category == c else 0 for c in COLOR_CATEGORIES]
    return np.array([numerical + ohe], dtype=float)


# ── Endpoint ───────────────────────────────────────────────────────────────────
@avocado_app.post("/predict")
def predict(avocado: AvocadoSchema):
    features = build_features(avocado)
    scaled   = avocado_app.state.scaler.transform(features)

    pred  = str(avocado_app.state.model.predict(scaled)[0])
    proba = avocado_app.state.model.predict_proba(scaled)[0].tolist()

    return {
        "status": RIPENESS_LABELS.get(pred, pred),
        "probabilities": {
            label: round(p, 4)
            for label, p in zip(avocado_app.state.model.classes_, proba)
        }
    }


if __name__ == "__main__":
    uvicorn.run("main:avocado_app", host="127.0.0.1", port=8000, reload=False)