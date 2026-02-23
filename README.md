# CLIP Trained Model

Trained CLIP-based aesthetic scoring model and the code to run it.

## Contents

- **`best_model_complete.pth`** – Trained model weights (stored via Git LFS).
- **`aesthetic_scorer/`** – Everything needed to run the model: scoring code, dependencies, and usage guide.

## Quick start

1. Install dependencies (from repo root):
   ```bash
   pip install -r aesthetic_scorer/requirements.txt
   ```
2. Score an image (model path defaults to this repo’s `best_model_complete.pth`):
   ```bash
   python aesthetic_scorer/core_scorer/score_image.py your_image.jpg --features
   ```

See **`aesthetic_scorer/README.md`** for full setup and usage.
