"""
Image Scorer - Main Interface for Scoring Images

The main class you use to score images with aesthetic assessment.

What this does:
- Scores images as a percentage (0-100 aesthetic quality)
- Extracts technical features (brightness, sharpness, noise, exposure, contrast, saturation, warmth)
- Provides diagnostic information (detail distribution, darkness level)

Technical features are CV-based (computed from image pixels) and calibrated to 0-100 scores.

This is the primary interface - import this class and use it to score images.
All CLI scripts and examples use this class internally.
"""

import torch
from PIL import Image
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging

from .model_loader import load_aesthetic_model
from .aesthetic_model import AestheticProbe
from transformers import CLIPProcessor
from .features.technicals import extract_raw_metrics, compute_detail_distribution_label, compute_dark_fraction_label
from .features.calibration import load_calibration, apply_calibration, NON_QUALITY_METRICS
from .features.guards import apply_guards
import numpy as np

logger = logging.getLogger(__name__)


class ImageScorer:
    """
    Score images using trained aesthetic assessment model
    
    Provides both aesthetic scoring feature
    extraction for detailed analysis.
    
    Usage:
        scorer = ImageScorer('best_model_complete.pth')
        result = scorer.score_image('photo.jpg')
        print(f"Score: {result['aesthetic_score']:.1f}%")
        print(f"Technical: {result['technical_features']}")
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = 'auto',
        extract_features: bool = True,
        calibration_path: Optional[Union[str, Path]] = None
    ):
        """
        Initialize the image scorer
        
        Args:
            model_path: Path to trained model checkpoint (.pth file)
            device: Device to run on ('auto', 'cpu', or 'cuda')
            extract_features: If True, extract features in addition to score
            calibration_path: Optional path to calibration JSON file. If None, uses default:
                             core_scorer/calibration/technical_calibration.json
        """
        self.model_path = Path(model_path)
        self.extract_features = extract_features
        
        # Auto-detect device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing ImageScorer on device: {self.device}")
        
        # Load model and processor
        self.model, self.processor = load_aesthetic_model(
            str(self.model_path),
            str(self.device)
        )
        
        # Load calibration if available
        if calibration_path is None:
            calibration_path = Path(__file__).parent / 'calibration' / 'technical_calibration.json'
        else:
            calibration_path = Path(calibration_path)
        
        self.calibration_params = None
        if calibration_path.exists():
            try:
                self.calibration_params = load_calibration(calibration_path)
                logger.info(f"Loaded calibration from {calibration_path}")
            except Exception as e:
                logger.warning(f"Failed to load calibration from {calibration_path}: {e}")
        else:
            logger.warning(f"Calibration file not found: {calibration_path}. Technical features will use raw metrics.")
        
        # Feature extractor no longer needed (removed abstract features and styles)
        # Keep extract_features flag for backward compatibility, but it only affects technical features now
        self.feature_extractor = None
    
    def _preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Load and preprocess image for CLIP.
        
        Preserves aspect ratio by using padding instead of cropping
        This is critical for panoramic images and unusual aspect ratios
        
        Args:
            image_path: Path to image file
        
        Returns:
            pixel_values: Preprocessed image tensor ready for model input
                Shape: (1, 3, 224, 224)
        
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be opened or processed
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        try:
            # Load image and convert to RGB (handles different formats)
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise ValueError(f"Failed to open image {image_path}: {e}")
        
        # Use CLIP processor for preprocessing
        # Note: Standard CLIP processor may crop; for universal sizes, we'd need
        # custom preprocessing, but for now we use standard processor
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(self.device)
        except Exception as e:
            raise ValueError(f"Failed to process image {image_path}: {e}")
        
        return pixel_values
    
    def _predict(self, pixel_values: torch.Tensor) -> float:
        """
        Run aesthetic model inference.
        
        Args:
            pixel_values: Preprocessed image tensor
        
        Returns:
            aesthetic_score: Predicted aesthetic quality score (0-100 percentage)
        """
        with torch.no_grad():
            scores = self.model(pixel_values)
            # Extract scalar from tensor
            score = scores.item() if isinstance(scores, torch.Tensor) else scores
        
        return float(score)
    
    def _extract_features(self, pixel_values: torch.Tensor, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract technical features from image using CV metrics.
        
        Computes raw CV metrics from image pixels, applies calibration if available,
        and computes diagnostic labels.
        
        Args:
            pixel_values: Preprocessed image tensor
            image_path: Path to original image (needed for CV metrics on full-resolution image)
        
        Returns:
            Dictionary with technical_features, raw_metrics, and diagnostic_info
        """
        if not self.extract_features:
            return {
                'technical_features': {},
                'raw_metrics': {}
            }
        
        # Load full-resolution PIL image for CV metrics
        image_path = Path(image_path)
        try:
            pil_image = Image.open(image_path).convert('RGB')
        except Exception as e:
            logger.error(f"Failed to load image for CV metrics: {e}")
            pil_image = None
        
        # Extract CV-based technical features
        if pil_image is not None:
            # Compute raw CV metrics
            raw_metrics = extract_raw_metrics(pil_image)
            
            # Separate quality metrics from diagnostic metrics
            quality_metrics = {k: v for k, v in raw_metrics.items() 
                             if k not in NON_QUALITY_METRICS}
            diagnostic_metrics = {k: v for k, v in raw_metrics.items() 
                                 if k in NON_QUALITY_METRICS}
            
            # Apply calibration if available (only to quality metrics)
            if self.calibration_params:
                try:
                    technical_features = apply_calibration(quality_metrics, self.calibration_params)
                    # Convert to 0-1 range for backward compatibility (divide by 100)
                    technical_features_normalized = {k: v / 100.0 for k, v in technical_features.items()}
                    # But keep 0-100 as primary output
                    technical_features = technical_features_normalized
                except Exception as e:
                    logger.warning(f"Failed to apply calibration: {e}. Using raw metrics.")
                    # Fallback: use raw metrics (normalized to 0-1 where needed)
                    # Map only expected technical feature keys
                    expected_keys = {'brightness', 'sharpness', 'noise', 'exposure', 'contrast', 'saturation'}
                    technical_features = {}
                    for k, v in quality_metrics.items():
                        key = k.replace('_raw', '')
                        if key in expected_keys:
                            # Normalize to 0-1 (most raw metrics are already 0-1, but clamp to be safe)
                            technical_features[key] = float(np.clip(v, 0.0, 1.0))
            else:
                # No calibration: use raw metrics (normalized to 0-1 where needed)
                logger.warning("No calibration available. Using raw metrics.")
                # Map only expected technical feature keys
                expected_keys = {'brightness', 'sharpness', 'noise', 'exposure', 'contrast', 'saturation'}
                technical_features = {}
                for k, v in quality_metrics.items():
                    key = k.replace('_raw', '')
                    if key in expected_keys:
                        # Normalize to 0-1 (most raw metrics are already 0-1, but clamp to be safe)
                        technical_features[key] = float(np.clip(v, 0.0, 1.0))
        else:
            raw_metrics = {}
            technical_features = {}
            diagnostic_metrics = {}
        
        # Apply consistency guards (only on technical features now)
        guard_messages = []
        if raw_metrics and technical_features:
            # Convert technical_features to 0-100 for guards
            technical_scores_100 = {k: v * 100.0 for k, v in technical_features.items()}
            
            adjusted_technical, guard_messages = apply_guards(
                raw_metrics=raw_metrics,
                technical_scores=technical_scores_100
            )
            
            # Log guard messages
            for msg in guard_messages:
                logger.debug(f"Guard: {msg}")
            
            # Convert back to 0-1 for output (backward compatibility)
            technical_features = {k: v / 100.0 for k, v in adjusted_technical.items()}
        
        # Compute diagnostic labels for diagnostic metrics
        diagnostic_info = {}
        if 'center_edge_sharpness_ratio_raw' in raw_metrics:
            ratio = raw_metrics['center_edge_sharpness_ratio_raw']
            diagnostic_info['center_edge_sharpness_ratio_raw'] = ratio
            diagnostic_info['detail_distribution'] = compute_detail_distribution_label(ratio)
        
        # Include dark_fraction_raw with categorical label
        if 'dark_fraction_raw' in raw_metrics:
            dark_frac = raw_metrics['dark_fraction_raw']
            diagnostic_info['dark_fraction_raw'] = dark_frac
            diagnostic_info['darkness_level'] = compute_dark_fraction_label(dark_frac)
        
        return {
            'technical_features': technical_features,
            'raw_metrics': raw_metrics,
            'diagnostic_info': diagnostic_info
        }
    
    def score_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Score a single image and extract features.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Dictionary with:
            - aesthetic_score: float (0-100 percentage)
            - technical_features: dict with brightness, sharpness, noise, etc. (0-1 range)
            - abstract_features: dict with mood, atmosphere, etc. (0-1 range)
            - photographic_styles: dict with detected styles (binary flags)
        
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be processed
        """
        try:
            # Preprocess image
            pixel_values = self._preprocess_image(image_path)
            
            # Get aesthetic score
            aesthetic_score = self._predict(pixel_values)
            
            # Extract features if enabled
            if self.extract_features:
                features = self._extract_features(pixel_values, image_path)
            else:
                features = {
                    'technical_features': {},
                    'raw_metrics': {}
                }
            
            # Combine results
            result = {
                'aesthetic_score': aesthetic_score,
                **features
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error scoring image {image_path}: {e}")
            raise
    
    def score_batch(self, image_paths: List[Union[str, Path]]) -> List[Dict[str, Any]]:
        """
        Score multiple images
        
        Args:
            image_paths: List of paths to image files
        
        Returns:
            List of result dictionaries (one per image)
            Each dictionary has the same structure as score_image() output
        
        Note:
            Continues processing even if individual images fail.
            Failed images will raise exceptions - caller should handle them.
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.score_image(image_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to score {image_path}: {e}")
                # Re-raise to let caller decide how to handle
                raise
        
        return results

