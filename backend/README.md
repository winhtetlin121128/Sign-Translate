# Python recognition API

This service follows the supplied architecture: Python + MediaPipe + OpenCV for landmark extraction, TensorFlow/Keras for classification, and FastAPI for the browser API. Firebase is optional and is not required for camera tracking or recognition.

## Start locally

Use Python 3.11 on Windows, macOS, or Linux:

```sh
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Check `http://localhost:8000/health`; interactive API documentation is at `http://localhost:8000/docs`.

## Endpoints

- `GET /health` reports the runtime versions and whether classifier weights are installed.
- `POST /api/track/frame` accepts one JPEG, PNG, or WebP frame as multipart field `image`, and returns pose/hand keypoints in the dataset's `(75, 3)` order: 33 pose, 21 left hand, 21 right hand.
- `POST /api/recognize` accepts a JSON body shaped as `{ "keypoints": [[[x, y, z], ...], ...] }` with up to 200 frames and returns the top three gloss labels.

## Classifier weights required

The repository contains annotations and extracted keypoint samples, but no trained `.keras` model. Set `MSL_MODEL_PATH` to a compatible TensorFlow/Keras model before recognition can return predictions. Its input must accept `(batch, 200, 225)` and optionally a boolean padding mask `(batch, 200)`, and its output must contain 541 scores in the order in `data/myanmar-sign-language-recognition/label_map.json`. The endpoint returns HTTP 503 until weights are supplied. This prevents sample labels from being presented as real inference.

The provided upstream demo weights are ONNX, not Keras. To use those weights, either retain the current in-browser ONNX inference or convert/retrain and validate a compatible Keras model before deploying this Python service.

## Hosting

Deploy this directory to a Python-capable host that supports long-running ASGI services, environment variables, HTTPS, and outbound access to the model file if it is stored remotely. Configure `CORS_ORIGINS` with the exact deployed frontend origin. For a small demo, Render, Railway, or a container service can run the `uvicorn` command above. GitHub Pages only hosts static frontend files; it cannot run FastAPI, MediaPipe, or TensorFlow.

To connect the React dashboard, set `VITE_API_BASE_URL` to the backend's public HTTPS origin in the frontend build environment. The dashboard checks `/health`; when `classifier_loaded` is true, it sends the current keypoint sequence to `/api/recognize`. When the Keras file is absent, the existing in-browser ONNX classifier keeps handling recognition.

