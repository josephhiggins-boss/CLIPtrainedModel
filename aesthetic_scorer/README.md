# Aesthetic Scorer - User Guide

This system scores images for aesthetic quality and gives you detailed information about what makes them good or bad.

## What This Does

When you give it an image, it tells you:
- **Overall Score**: A number from 0 to 100 
- **Technical Features**: How bright it is, how sharp, how much noise, exposure, contrast, saturation, warmth
- **Diagnostic Information**: Where the sharpness is distributed, how much of the image is dark

## What You Need to Run This

1. **Python** (version 3.8 or newer)
   - Check if you have it: Open command prompt and type `python --version`
   - If you don't have it, download from python.org

2. **Python Packages** (install these):
   ```bash
   pip install torch transformers pillow numpy scipy
   ```
   
   These are the only ones you actually need:
   - `torch` - The AI framework (PyTorch)
   - `transformers` - For the CLIP model that understands images
   - `pillow` - For opening and reading image files
   - `numpy` - For doing math on images
   - `scipy` - For fast image processing

3. **Your Trained Model File**
   - This is a `.pth` file that contains the trained AI model
   - Put it somewhere easy to find, like `D:\best_model_complete.pth`
   - This file is big (587MB) 
## How to Install Everything

1. Open command prompt (PowerShell or Command Prompt)

2. Go to your project folder

3. Install the packages:
   ```bash
   pip install torch transformers pillow numpy scipy
   ```

4. Wait for it to finish (might take a few minutes)


## The Main System

**The main system with all features is in `core_scorer/` folder.**

This is what you should use - it has:
- Aesthetic scoring (0-100%)
- Technical features (brightness, sharpness, noise, exposure, contrast, saturation, warmth)
- Diagnostic information (detail distribution, darkness level)

## How to Use It

### Score an Image

```bash
python core_scorer/score_image.py your_image.jpg --features
```

Replace:
- `your_image.jpg` with your actual image file


### What You'll See

```
============================================================
  Aesthetic Score: 72.45%
============================================================

Technical Features (Calibrated Scores):
  - Brightness: 65.0% (raw: 0.6500)
  - Sharpness: 78.0% (raw: 0.0234)
  - Noise: 25.0% (raw: 0.0012)
  - Exposure: 82.0% (raw: 0.8200)
  - Contrast: 71.0% (raw: 0.1005)
  - Saturation: 68.0% (raw: 0.4192)

Diagnostic Information:
  - Centre-Edge Sharpness Ratio (raw): 1.45
  - Detail Distribution: mild_centre_dominant
  - Dark Fraction (raw): 0.15
  - Darkness Level: bright
```

### Using in Python Code

```python
from core_scorer import ImageScorer

# Load the model (do this once)
scorer = ImageScorer(
    model_path='D:\\best_model_complete.pth',
    extract_features=True
)

# Score an image
result = scorer.score_image('photo.jpg')

# Get the score
print(f"Score: {result['aesthetic_score']:.2f}%")

# Get technical features
tech = result['technical_features']
print(f"Brightness: {tech['brightness']*100:.1f}%")
print(f"Sharpness: {tech['sharpness']*100:.1f}%")
```

## Understanding the Scores

### Aesthetic Score (0-100%)

This is the overall quality rating:
- **80-100%**: Really good photos
- **60-79%**: Good photos
- **40-59%**: Average photos
- **20-39%**: Below average
- **0-19%**: Poor quality

### Technical Features

These are measured from the actual image pixels:

- **Brightness**: How bright or dark the image is (higher = brighter)
- **Exposure**: How well exposed it is - not too dark, not too bright, no clipping (higher = better)
- **Sharpness**: How in focus it is (higher = sharper)
- **Noise**: How much grain/noise there is (lower = less noise, which is better)
- **Contrast**: Difference between light and dark areas (higher = more contrast)
- **Saturation**: How colourful it is (higher = more colourful)
- **Warmth**: Warm colours (red/yellow) vs cool colours (blue) - positive = warm, negative = cool

### Diagnostic Information

These aren't quality scores, just descriptions:

- **Detail Distribution**: Where the sharpness is in the image
  - `edge_dominant_detail` = edges are sharper than centre
  - `uniform_detail` = same sharpness everywhere
  - `centre_dominant_detail` = centre is sharper (might be shallow depth of field)

- **Darkness Level**: How much of the image is dark
  - `very_bright`, `bright`, `balanced`, `dark`, `very_dark`, `low_key_or_silhouette`


## File Structure

The main files you need are in `core_scorer/`:
- `image_scorer.py` - The main class you use
- `score_image.py` - Command line script
- `aesthetic_model.py` - The AI model
- `features/` - Technical feature extraction

Everything else is optional or for advanced use.

## What Each Score Means

**Aesthetic Score**: The overall quality. This is what the AI model thinks after looking at everything.

**Technical Features**: Measured from pixels. These are facts about the image (brightness, sharpness, noise, exposure, contrast, saturation, warmth).

**Diagnostics**: Just descriptions, not quality judgments. Like "this has shallow depth of field" not "shallow depth of field is good/bad".


The system is pretty straightforward once you have everything installed.
