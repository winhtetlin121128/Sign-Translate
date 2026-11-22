# Signa · Myanmar Sign Language Recognition

A browser based Myanmar Sign Language demo. Start a tracking session to see hand landmarks and live BiLSTM word predictions, or browse the dataset vocabulary below the camera.

## Run locally

```sh
npm install
npm run dev
```

Allow camera access when prompted. The page uses MediaPipe Pose and Hand Landmarker in the browser, then passes 200-frame pose and hand sequences to the BiLSTM classifier. Camera frames are processed locally. The classifier weights and MediaPipe task models download from their respective upstream hosts the first time a session starts.

## Data and model sources

- Myanmar vocabulary, annotations, and extracted keypoints are from [lawun330/myanmar-sign-language-recognition](https://github.com/lawun330/myanmar-sign-language-recognition).
- The underlying [MSL4Emergency corpus](https://github.com/ye-kyaw-thu/MSL4Emergency) is licensed CC BY-NC-SA 4.0. Keep the attribution and license terms when sharing or adapting the dataset; commercial use is not permitted under that license.
- The BiLSTM ONNX model is loaded directly from the source repository at runtime. The model files are not redistributed in this project.

The local `data/myanmar-sign-language-recognition/` folder contains 558 annotations, 541 label classes, train/validation/test and five-fold splits, 558 pose-and-hand landmark sequences with visibility arrays, extraction statistics, and five sample inference videos. The homepage bundles the label map and annotations for the word dashboard, and shows a representative hand-landmark sketch on each word card.

Regenerate the word-card hand sketches after replacing keypoint files with:

```sh
npm run generate:hand-shapes
```

## Recognition notes

The dashboard shows the current top model match, a short list of stable session predictions, and a searchable vocabulary. Predictions can vary with camera framing and signing style; the model is an educational demonstration rather than a certified translation service.
