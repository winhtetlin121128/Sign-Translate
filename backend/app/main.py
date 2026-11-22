"""FastAPI service for Myanmar sign hand/body landmark extraction and inference."""

from __future__ import annotations

import json
import os
from typing import Any
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("MSL_DATA_DIR", ROOT / "data" / "myanmar-sign-language-recognition"))
LABEL_MAP_PATH = Path(os.getenv("MSL_LABEL_MAP", DATA_DIR / "label_map.json"))
MODEL_PATH = Path(os.getenv("MSL_MODEL_PATH", ROOT / "models" / "msl_classifier.keras"))
SEQUENCE_LENGTH = int(os.getenv("MSL_SEQUENCE_LENGTH", "200"))
FEATURE_COUNT = 75 * 3
MODEL_CACHE = Path(os.getenv("MEDIAPIPE_MODEL_DIR", "/tmp/msl-models"))
HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"

origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if origin.strip()]
app = FastAPI(title="Myanmar Sign Recognition API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://sign-translate(?:-six|-git-hand-data(?:-[a-z0-9-]+)?)?\.vercel\.app",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

hand_landmarker: Any | None = None
pose_landmarker: Any | None = None
vision_error = ""
classifier: Any | None = None
labels: list[str] = []
if LABEL_MAP_PATH.exists():
    label_map = json.loads(LABEL_MAP_PATH.read_text(encoding="utf-8"))
    labels = [label for label, _ in sorted(label_map["label2idx"].items(), key=lambda item: item[1])]
if MODEL_PATH.exists():
    classifier = tf.keras.models.load_model(MODEL_PATH, compile=False)


def _model_asset(filename: str, url: str) -> Path:
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_CACHE / filename
    if not model_path.exists():
        urlretrieve(url, model_path)
    return model_path


@app.on_event("startup")
def initialize_mediapipe() -> None:
    global hand_landmarker, pose_landmarker, vision_error
    try:
        hand_options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_model_asset("hand_landmarker.task", HAND_MODEL_URL))),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.5,
        )
        pose_options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(_model_asset("pose_landmarker_full.task", POSE_MODEL_URL))),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        hand_landmarker = mp_vision.HandLandmarker.create_from_options(hand_options)
        pose_landmarker = mp_vision.PoseLandmarker.create_from_options(pose_options)
    except Exception as error:
        vision_error = str(error)


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
        "mediapipe_ready": hand_landmarker is not None and pose_landmarker is not None,
        "mediapipe_error": vision_error or None,
        "classifier_loaded": classifier is not None,
        "label_count": len(labels),
        "sequence_length": SEQUENCE_LENGTH,
    }


@app.post("/api/track/frame")
async def track_frame(image: UploadFile = File(...)) -> dict[str, Any]:
    """Detect pose and hands from one JPEG/PNG camera frame."""
    if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(status_code=415, detail="Upload a JPEG, PNG, or WebP camera frame.")
    if hand_landmarker is None or pose_landmarker is None:
        raise HTTPException(status_code=503, detail=f"MediaPipe Tasks failed to initialize: {vision_error or 'not ready'}")
    encoded = np.frombuffer(await image.read(), dtype=np.uint8)
    frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode the uploaded image.")

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    hand_result = hand_landmarker.detect(mp_image)
    pose_result = pose_landmarker.detect(mp_image)
    points = np.zeros((75, 3), dtype=np.float32)
    if pose_result.pose_landmarks:
        for index, point in enumerate(pose_result.pose_landmarks[0][:33]):
            points[index] = (point.x, point.y, point.z)
    for hand_index, hand in enumerate(hand_result.hand_landmarks):
        handedness = hand_result.handedness[hand_index][0].category_name if hand_index < len(hand_result.handedness) else "Left"
        start = 54 if handedness == "Right" else 33
        if hand:
            for index, point in enumerate(hand[:21]):
                points[start + index] = (point.x, point.y, point.z)

    return {
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "hands_detected": len(hand_result.hand_landmarks),
        "keypoints": points.tolist(),
    }


@app.on_event("shutdown")
def close_mediapipe() -> None:
    if hand_landmarker is not None:
        hand_landmarker.close()
    if pose_landmarker is not None:
        pose_landmarker.close()


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

