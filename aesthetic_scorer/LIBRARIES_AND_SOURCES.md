# Libraries and Open Source Codebases Used

This document lists all the open source libraries and codebases used in this project, with links to their source code repositories.

## Core Deep Learning Libraries

### 1. PyTorch
**What it does:** Deep learning framework - the foundation for all neural network operations

**Source Code:**
- GitHub: https://github.com/pytorch/pytorch
- Official Website: https://pytorch.org/
- License: BSD-style (open source)
- Documentation: https://pytorch.org/docs/stable/index.html

**Used for:**
- Neural network architecture (AestheticProbe model)
- Tensor operations
- Model loading and inference
- GPU/CPU computation

**In our code:**
- `core_scorer/aesthetic_model.py` - Uses `torch.nn` for model layers
- `core_scorer/model_loader.py` - Uses `torch.load()` to load checkpoints
- All model forward passes use PyTorch tensors

---

### 2. HuggingFace Transformers
**What it does:** Provides pre-trained models including CLIP (Contrastive Language-Image Pre-training)

**Source Code:**
- GitHub: https://github.com/huggingface/transformers
- Official Website: https://huggingface.co/docs/transformers
- License: Apache 2.0 (open source)
- Documentation: https://huggingface.co/docs/transformers/index

**Used for:**
- CLIP model (ViT-B/16 vision encoder)
- CLIP processor (image preprocessing)
- Pre-trained model weights

**In our code:**
- `core_scorer/aesthetic_model.py` - Imports `CLIPModel, CLIPProcessor`
- `core_scorer/model_loader.py` - Uses `CLIPProcessor.from_pretrained()`

---

### 3. OpenAI CLIP (via Transformers)
**What it does:** Vision-language model that understands images and text together

**Original CLIP Repository:**
- GitHub: https://github.com/openai/CLIP
- Paper: "Learning Transferable Visual Models From Natural Language Supervision"
- License: MIT (open source)

**Note:** We use CLIP through HuggingFace Transformers, which provides a PyTorch-compatible implementation. The original OpenAI CLIP is also open source.

**Used for:**
- Image feature extraction (frozen CLIP backbone)
- Understanding image content for aesthetic scoring

**In our code:**
- `core_scorer/aesthetic_model.py` - CLIP vision encoder as frozen backbone
- The model uses CLIP's image features as input to the aesthetic scoring head

---

## Image Processing Libraries

### 4. Pillow (PIL - Python Imaging Library)
**What it does:** Image loading, manipulation, and format conversion

**Source Code:**
- GitHub: https://github.com/python-pillow/Pillow
- Official Website: https://pillow.readthedocs.io/
- License: PIL License (open source, HPND)
- Documentation: https://pillow.readthedocs.io/en/stable/

**Used for:**
- Loading images from files (JPG, PNG, WEBP, etc.)
- Image format conversion
- Basic image operations

**In our code:**
- `core_scorer/image_scorer.py` - `Image.open()` to load images
- `core_scorer/features/technicals.py` - Image loading for CV metrics

---

### 5. NumPy
**What it does:** Numerical computing library for array operations

**Source Code:**
- GitHub: https://github.com/numpy/numpy
- Official Website: https://numpy.org/
- License: BSD (open source)
- Documentation: https://numpy.org/doc/stable/

**Used for:**
- Array operations on image pixels
- Mathematical computations for CV metrics
- Statistical operations (percentiles, mean, std, etc.)

**In the code:**
- `core_scorer/features/technicals.py` - All CV metric calculations use NumPy arrays
- `core_scorer/features/calibration.py` - Percentile calculations use NumPy

---

### 6. SciPy
**What it does:** Scientific computing library (includes signal processing)

**Source Code:**
- GitHub: https://github.com/scipy/scipy
- Official Website: https://scipy.org/
- License: BSD (open source)
- Documentation: https://docs.scipy.org/doc/scipy/

**Used for:**
- Fast 2D convolution (`scipy.signal.convolve2d`)
- Optimized image processing operations

**In our code:**
- `core_scorer/features/technicals.py` - Uses `scipy.signal.convolve2d` for Sobel filters and convolution operations

---

## Standard Library (Built-in Python)

These are part of Python itself, no installation needed:

- **`pathlib`** - File path handling (Python 3.4+)
- **`typing`** - Type hints (Python 3.5+)
- **`logging`** - Logging system
- **`json`** - JSON file reading/writing
- **`datetime`** - Date/time operations

---

## How to Access Source Code

### Viewing Source Code Online

1. **PyTorch:** https://github.com/pytorch/pytorch
   - Main repository with all PyTorch code
   - Look in `torch/nn/` for neural network layers

2. **Transformers:** https://github.com/huggingface/transformers
   - CLIP implementation: `src/transformers/models/clip/`
   - Model files: `src/transformers/models/clip/modeling_clip.py`

3. **CLIP (Original):** https://github.com/openai/CLIP
   - Original OpenAI implementation
   - Reference implementation

4. **Pillow:** https://github.com/python-pillow/Pillow
   - Image processing code

5. **NumPy:** https://github.com/numpy/numpy
   - Core numerical operations

6. **SciPy:** https://github.com/scipy/scipy
   - Signal processing: `scipy/signal/`


### Understanding CLIP
- **Transformers CLIP Model:** `transformers/src/transformers/models/clip/modeling_clip.py`
- **CLIP Vision Encoder:** Look for `CLIPVisionModel` class
- **Original CLIP:** `CLIP/clip/model.py`

### Understanding PyTorch
- **Neural Network Layers:** `pytorch/torch/nn/modules/`
- **Linear Layer:** `pytorch/torch/nn/modules/linear.py`
- **Model Loading:** `pytorch/torch/serialization.py`

### Understanding Image Processing
- **Pillow Image Class:** `Pillow/src/PIL/Image.py`
- **NumPy Arrays:** `numpy/numpy/core/src/multiarray/`

---

## License Information

All libraries used are open source:

- **PyTorch:** BSD License
- **Transformers:** Apache 2.0
- **CLIP:** MIT License
- **Pillow:** PIL License (HPND)
- **NumPy:** BSD License
- **SciPy:** BSD License
---

## Additional Resources

### Documentation
- PyTorch Tutorials: https://pytorch.org/tutorials/
- HuggingFace Course: https://huggingface.co/course
- CLIP Paper: https://arxiv.org/abs/2103.00020
- NumPy User Guide: https://numpy.org/doc/stable/user/

### Community
- PyTorch Forums: https://discuss.pytorch.org/
- HuggingFace Forums: https://discuss.huggingface.co/
- Stack Overflow: Tag questions with library names

---

## Summary

**Main Libraries:**
1. **PyTorch** - Deep learning framework
2. **HuggingFace Transformers** - CLIP model access
3. **OpenAI CLIP** - Vision-language model (via Transformers)
4. **Pillow** - Image processing
5. **NumPy** - Numerical computing
6. **SciPy** - Scientific computing


Compiled using ChatGPT
