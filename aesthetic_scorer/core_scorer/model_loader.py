"""
Model Loader - Load Trained Models from Files

Loads the trained aesthetic scoring model from saved checkpoint files (.pth files).

What this does:
- Loads model weights from checkpoint files
- Handles different checkpoint formats (nested dicts, direct state dicts, etc.)
- Sets up the model for scoring 
- Returns the model and image processor ready to use

Used by ImageScorer when you initialize it with a model path.
"""

import torch
from pathlib import Path
from typing import Tuple
import logging

from .aesthetic_model import AestheticProbe
from transformers import CLIPProcessor

logger = logging.getLogger(__name__)


def load_aesthetic_model(
    checkpoint_path: str,
    device: str = 'cpu'
) -> Tuple[AestheticProbe, CLIPProcessor]:
    """
    Load trained aesthetic model from checkpoint.
    
    Handles different checkpoint formats:
    - Direct state dict
    - Nested with 'model_state_dict' key
    - Nested with 'state_dict' key
    - Extra keys (good_emb, bad_emb) that aren't in the model
    
    Args:
        checkpoint_path: Path to saved model weights (.pth file)
        device: Device to load model on ('cuda' or 'cpu')
    
    Returns:
        Tuple of (model, processor):
        - model: Loaded model in eval mode, ready for inference
        - processor: CLIP processor for image preprocessing
    
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If model loading fails
    """
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}"
        )
    
    logger.info(f"Loading model from {checkpoint_path}")
    
    # Initialize model (with positional embeddings removed for universal sizes)
    model = AestheticProbe(remove_pos_embeddings=True)
    
    # Load saved weights
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load checkpoint from {checkpoint_path}: {e}"
        )
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            logger.debug("Found 'model_state_dict' in checkpoint")
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            logger.debug("Found 'state_dict' in checkpoint")
        else:
            # Assume the dict itself is the state dict
            state_dict = checkpoint
            logger.debug("Using checkpoint dict directly as state_dict")
    else:
        # Checkpoint is directly a state dict
        state_dict = checkpoint
        logger.debug("Checkpoint is a direct state dict")
    
    # Get model's expected state dict keys
    model_state_dict = model.state_dict()
    
    # Filter state dict to only include matching keys
    # This handles models with extra components (good_emb, bad_emb, etc.)
    filtered_state_dict = {}
    skipped_keys = []
    
    for key, value in state_dict.items():
        # Include CLIP layers and model layers that exist in our model
        if key in model_state_dict:
            filtered_state_dict[key] = value
        elif key.startswith('clip.'):
            # Include all CLIP layers even if not explicitly in model_state_dict
            # (they're part of the CLIP submodule)
            filtered_state_dict[key] = value
        else:
            skipped_keys.append(key)
    
    if skipped_keys:
        logger.debug(f"Skipped {len(skipped_keys)} keys not in model: {skipped_keys[:5]}...")
    
    # Load only the matching keys with strict=False to handle extra/missing keys
    missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys in checkpoint: {missing_keys[:5]}...")
    if unexpected_keys:
        logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys[:5]}...")
    
    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    # Get CLIP processor
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch16')
    
    logger.info(f"Model loaded successfully on {device}")
    
    return model, processor

