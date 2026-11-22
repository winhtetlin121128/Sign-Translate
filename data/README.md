# Myanmar Sign Recognition Data

Copied from [lawun330/myanmar-sign-language-recognition](https://github.com/lawun330/myanmar-sign-language-recognition/tree/main/data), whose README credits Sayar's annotations and the [MSL4Emergency dataset](https://github.com/ye-kyaw-thu/MSL4Emergency).

The upstream corpus is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Attribute the original source, share adaptations under the same license, and do not use the corpus commercially.

Contents:

- `myanmar-sign-language-recognition/annotations.txt`: Myanmar written text paired with sign transcription.
- `myanmar-sign-language-recognition/label_map.json`: 541 sign labels and their model indices.
- `myanmar-sign-language-recognition/keypoints/`: MediaPipe Pose + Hands sequences as float arrays `(frames, 75, 3)`, visibility scores `(frames, 33)`, and extraction statistics.
- `myanmar-sign-language-recognition/splits.json` and `kfold_splits.json`: fixed train/validation/test and five-fold partitions.
- `myanmar-sign-language-recognition/sample4infer/`: five sample clips for inference.

The original full training videos are not included in this folder.
