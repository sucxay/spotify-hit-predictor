"""
main.py — Spotify Hit Predictor API
Run: uvicorn main:app --reload
"""

import os
import pickle
import time
import warnings
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import io

warnings.filterwarnings("ignore")

# ── Feature metadata ────────────────────────────────────────────────────────
NUMERIC_COLS = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence",
    "loudness", "tempo", "duration_ms"
]
CATEGORICAL_COLS = ["key", "mode", "time_signature"]
ALL_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS

MODEL_PATH = os.getenv("MODEL_PATH", "best_model.pkl")

# ── App state ────────────────────────────────────────────────────────────────
model_store = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file '{MODEL_PATH}' not found. "
            "Run `python train.py --data <csv>` first."
        )
    with open(MODEL_PATH, "rb") as f:
        model_store["pipeline"] = pickle.load(f)
    print(f"✅ Model loaded from {MODEL_PATH}")
    yield
    # shutdown
    model_store.clear()


app = FastAPI(
    title="Spotify Hit Predictor",
    description=(
        "Predicts whether a Spotify track will be a **hit** based on its audio features. "
        "A 'hit' is defined as a track that stayed on the chart for at least the median "
        "number of weeks. Powered by the best of Logistic Regression / Random Forest / XGBoost."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────────
class TrackFeatures(BaseModel):
    danceability:     float = Field(..., ge=0.0, le=1.0,  example=0.735,  description="0.0–1.0")
    energy:           float = Field(..., ge=0.0, le=1.0,  example=0.578,  description="0.0–1.0")
    speechiness:      float = Field(..., ge=0.0, le=1.0,  example=0.042,  description="0.0–1.0")
    acousticness:     float = Field(..., ge=0.0, le=1.0,  example=0.102,  description="0.0–1.0")
    instrumentalness: float = Field(..., ge=0.0, le=1.0,  example=0.0,    description="0.0–1.0")
    liveness:         float = Field(..., ge=0.0, le=1.0,  example=0.116,  description="0.0–1.0")
    valence:          float = Field(..., ge=0.0, le=1.0,  example=0.632,  description="0.0–1.0")
    loudness:         float = Field(..., ge=-60.0, le=5.0, example=-5.883, description="dB, typically −60 to 0")
    tempo:            float = Field(..., ge=0.0, le=300.0, example=122.4,  description="BPM")
    duration_ms:      int   = Field(..., ge=0,             example=210000, description="Track length in ms")
    key:              int   = Field(..., ge=0, le=11,      example=7,      description="0=C … 11=B")
    mode:             int   = Field(..., ge=0, le=1,       example=1,      description="0=minor, 1=major")
    time_signature:   int   = Field(..., ge=1, le=7,       example=4,      description="Beats per bar")

    @field_validator("duration_ms")
    @classmethod
    def duration_positive(cls, v):
        if v <= 0:
            raise ValueError("duration_ms must be > 0")
        return v


class PredictionResponse(BaseModel):
    prediction:   int   = Field(..., description="1 = Hit, 0 = Not a Hit")
    label:        str   = Field(..., description="'Hit' or 'Not a Hit'")
    probability:  float = Field(..., description="Confidence score (0–1) for the predicted class")
    hit_prob:     float = Field(..., description="Probability of being a Hit")
    not_hit_prob: float = Field(..., description="Probability of NOT being a Hit")
    latency_ms:   float = Field(..., description="Inference time in milliseconds")


class BatchPredictionResponse(BaseModel):
    count:       int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status:      str
    model_loaded: bool
    model_path:  str


# ── Helpers ───────────────────────────────────────────────────────────────────
def features_to_df(track: TrackFeatures) -> pd.DataFrame:
    df = pd.DataFrame([track.model_dump()])[ALL_FEATURES]
    # OneHotEncoder was fit on string categories from the CSV — cast to match
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype(str)
    return df


def run_inference(df: pd.DataFrame) -> list[dict]:
    pipeline = model_store["pipeline"]
    t0 = time.perf_counter()
    preds = pipeline.predict(df)
    probas = pipeline.predict_proba(df)
    latency = (time.perf_counter() - t0) * 1000

    results = []
    for i, (pred, proba) in enumerate(zip(preds, probas)):
        hit_p     = float(proba[1])
        not_hit_p = float(proba[0])
        results.append(PredictionResponse(
            prediction=int(pred),
            label="Hit" if pred == 1 else "Not a Hit",
            probability=round(max(hit_p, not_hit_p), 4),
            hit_prob=round(hit_p, 4),
            not_hit_prob=round(not_hit_p, 4),
            latency_ms=round(latency / len(df), 3),
        ))
    return results


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", tags=["General"])
def root():
    return {"message": "Spotify Hit Predictor API — visit /docs for interactive UI"}


@app.get("/health", response_model=HealthResponse, tags=["General"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded="pipeline" in model_store,
        model_path=MODEL_PATH,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(track: TrackFeatures):
    """
    Predict whether a **single track** is a hit based on its audio features.
    Returns prediction, confidence, and per-class probabilities.
    """
    if "pipeline" not in model_store:
        raise HTTPException(status_code=503, detail="Model not loaded")
    df = features_to_df(track)
    return run_inference(df)[0]


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(tracks: List[TrackFeatures]):
    """
    Predict hits for a **list of tracks** in one request (max 500).
    """
    if "pipeline" not in model_store:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if len(tracks) > 500:
        raise HTTPException(status_code=400, detail="Max 500 tracks per batch request")
    if len(tracks) == 0:
        raise HTTPException(status_code=400, detail="Send at least one track")

    rows = [t.model_dump() for t in tracks]
    df = pd.DataFrame(rows)[ALL_FEATURES]
    results = run_inference(df)

    return BatchPredictionResponse(count=len(results), predictions=results)


@app.post("/predict/csv", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_csv(file: UploadFile = File(...)):
    """
    Upload a **CSV file** with audio feature columns and get predictions back.
    The CSV must contain all required columns (see /features for the list).
    Extra columns (e.g. track name) are ignored.
    """
    if "pipeline" not in model_store:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files accepted")

    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")

    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing columns: {missing}")

    if len(df) > 500:
        raise HTTPException(status_code=400, detail="Max 500 rows per CSV upload")

    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype(str)
    results = run_inference(df[ALL_FEATURES])
    return BatchPredictionResponse(count=len(results), predictions=results)


@app.get("/features", tags=["General"])
def list_features():
    """Returns the list of required feature names and their expected types."""
    return {
        "numeric_features": {
            c: {"type": "float", "range": "0.0–1.0"} for c in NUMERIC_COLS
            if c not in ("loudness", "tempo", "duration_ms")
        } | {
            "loudness":    {"type": "float", "range": "-60 to 5 dB"},
            "tempo":       {"type": "float", "range": "0–300 BPM"},
            "duration_ms": {"type": "int",   "range": "> 0 ms"},
        },
        "categorical_features": {
            "key":            {"type": "int", "range": "0–11 (C=0 … B=11)"},
            "mode":           {"type": "int", "range": "0=minor, 1=major"},
            "time_signature": {"type": "int", "range": "1–7 (beats per bar)"},
        }
    }
