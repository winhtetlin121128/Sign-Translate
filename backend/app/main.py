"""FastAPI service for Myanmar sign hand/body landmark extraction and inference."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("MSL_DATA_DIR", ROOT / "data" / "myanmar-sign-language-recognition"))
LABEL_MAP_PATH = Path(os.getenv("MSL_LABEL_MAP", DATA_DIR / "label_map.json"))
MODEL_PATH = Path(os.getenv("MSL_MODEL_PATH", ROOT / "models" / "msl_classifier.keras"))
SEQUENCE_LENGTH = int(os.getenv("MSL_SEQUENCE_LENGTH", "200"))
FEATURE_COUNT = 75 * 3

origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if origin.strip()]
app = FastAPI(title="Myanmar Sign Recognition API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

holistic = mp.solutions.holistic.Holistic(
    static_image_mode=True,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
)
classifier: Any | None = None
labels: list[str] = []
if LABEL_MAP_PATH.exists():
    label_map = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    labels = [label for label, _ in sorted(label_map["label2idx"].items(), key=lambda item: item[1])]
if MODEL_PATH.exists():
    classifier = tf.keras.models.load_model(MODEL_PATH, compile=False)


class RecognitionRequest(BaseModel):
    # Feature order is 33 pose points, 21 left-hand points, 21 right-hand points;
    # each point is x/y/z, matching the included 75x3 dataset arrays.
    keypoints: list[list[list[float]]] = Field(min_length=1, max_length=SEQUENCE_LENGTH)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "myanmar-sign-recognition",
        "mediapipe": mp.__version__,
        "tensorflow": tf.__version__,
        "classifier_loaded": classifier is not None,
        "label_count": len(labels),
        "sequence_length": SEQUENCE_LENGTH,
    }


@app.post("/api/track/frame")
async def track_frame(image: UploadFile = File(...)) -> dict[str, Any]:
    """Detect pose and hands from one JPEG/PNG camera frame."""
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP camera frame.")
    encoded = np.frombuffer(await image.read(), dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded image.")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = holistic.process(rgb)
    points = np.zeros((75, 3), dtype=np.float32)
    if result.pose_landmarks:
        for index, point in enumerate(result.pose_landmarks.landmark[:33]):
            points[index] = (point.x, point.y, point.z)
    for hand, start in ((result.left_hand_landmarks, 33), (result.right_hand_landmarks, 54)):
        if hand:
            for index, point in enumerate(hand.landmark[:21]):
                points[start + index] = (point.x, point.y, point.z)

    return {
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "hands_detected": int(result.left_hand_landmarks is not None) + int(result.right_hand_landmarks is not None),
        "keypoints": points.tolist(),
    }


@app.post("/api/recognize")
def recognize(request: RecognitionRequest) -> dict[str, Any]:
    """Classify a sequence of extracted keypoints with a compatible Keras model."""
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Recognition weights are not installed. Set MSL_MODEL_PATH to a compatible .keras model.",
        )
    sequence = np.asarray(request.keypoints, dtype=np.float32)
    if sequence.ndim != 3 or sequence.shape[1:] != (75, 3):
        raise HTTPException(status_code=422, detail="Each frame must contain 75 landmarks with x/y/z coordinates.")

    # Dataset sequences are 75x3. Match the web demo's shoulder-relative
    # normalization and right-padding convention before invoking the model.
    sequence = sequence.reshape((-1, FEATURE_COUNT))
    if len(sequence) > SEQUENCE_LENGTH:
        start = (len(sequence) - SEQUENCE_LENGTH) // 2
        sequence = sequence[start : start + SEQUENCE_LENGTH]
    shoulder_mid = (sequence[:, 11 * 3 : 12 * 3] + sequence[:, 12 * 3 : 13 * 3]) / 2
    shoulder_width = np.linalg.norm(sequence[:, 12 * 3 : 12 * 3 + 3] - sequence[:, 11 * 3 : 11 * 3 + 3], axis=1)
    shoulder_width = np.where(shoulder_width > 1e-6, shoulder_width, 1)
    sequence = (sequence.reshape((-1, 75, 3)) - shoulder_mid[:, None, :]) / shoulder_width[:, None, None]
    sequence[:, :, 2] = 0
    length = len(sequence)
    padded = np.zeros((1, SEQUENCE_LENGTH, FEATURE_COUNT), dtype=np.float32)
    padded[0, :length] = sequence.reshape((length, FEATURE_COUNT))
    mask = np.ones((1, SEQUENCE_LENGTH), dtype=bool)
    mask[0, :length] = False

    try:
        output = classifier([padded, mask], training=False)
    except (TypeError, ValueError):
        output = classifier(padded, training=False)
    scores = np.asarray(output).reshape(-1)
    if not labels or scores.size != len(labels):
        raise HTTPException(status_code=500, detail="Model output size does not match the 541-class label map.")
    scores = np.exp(scores - np.max(scores))
    scores /= np.sum(scores) or 1
    top_indices = np.argsort(scores)[-3:][::-1]
    return {
        "predictions": [
            {"label": labels[int(index)], "confidence": float(scores[index])}
            for index in top_indices
        ]
    }

