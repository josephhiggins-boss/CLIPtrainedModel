"""
Technicals - Raw CV Metric Extraction

Extract deterministic computer vision metrics directly from PIL images.
These metrics measure actual image properties, not predicted features.
"""

import numpy as np
from PIL import Image
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def compute_detail_distribution_label(center_edge_ratio: float) -> str:
    """
    Compute diagnostic label for centre-edge sharpness ratio.
    
    This is NOT a quality score - it's a categorical descriptor of detail distribution.
    
    Args:
        center_edge_ratio: Ratio of centre sharpness to edge sharpness
        
    Returns:
        Diagnostic label string:
        - "edge_dominant_detail" if ratio < 0.85
        - "uniform_detail" if 0.85 <= ratio <= 1.15
        - "mild_centre_dominant" if 1.15 < ratio <= 1.30
        - "centre_dominant_detail (possible shallow DoF)" if ratio > 1.30
    """
    if center_edge_ratio < 0.85:
        return "edge_dominant_detail"
    elif center_edge_ratio <= 1.15:
        return "uniform_detail"
    elif center_edge_ratio <= 1.30:
        return "mild_centre_dominant"
    else:
        return "centre_dominant_detail (possible shallow DoF)"


def compute_dark_fraction_label(dark_fraction: float) -> str:
    """
    Compute diagnostic label for dark fraction.
    
    This is NOT a quality score - it's a categorical descriptor of image darkness.
    Used to support advice like "not a silhouette" or "not low-key".
    
    Args:
        dark_fraction: Fraction of pixels with Y < 0.15 (0.0 to 1.0)
        
    Returns:
        Diagnostic label string:
        - "very_bright" if dark_fraction < 0.10
        - "bright" if 0.10 <= dark_fraction < 0.20
        - "balanced" if 0.20 <= dark_fraction < 0.40
        - "dark" if 0.40 <= dark_fraction < 0.60
        - "very_dark" if 0.60 <= dark_fraction < 0.80
        - "low_key_or_silhouette" if dark_fraction >= 0.80
    """
    if dark_fraction < 0.10:
        return "very_bright"
    elif dark_fraction < 0.20:
        return "bright"
    elif dark_fraction < 0.40:
        return "balanced"
    elif dark_fraction < 0.60:
        return "dark"
    elif dark_fraction < 0.80:
        return "very_dark"
    else:
        return "low_key_or_silhouette"


def extract_raw_metrics(image: Image.Image) -> Dict[str, float]:
    """
    Extract raw computer vision metrics from a PIL image.
    
    Computes deterministic metrics directly from image pixels:
    - Brightness, clipping, dynamic range, exposure
    - Contrast, saturation, warmth
    - Sharpness, noise
    
    Args:
        image: PIL Image in RGB mode
        
    Returns:
        Dictionary with raw metric values (keys end with '_raw')
        Includes: brightness_raw, highlight_clip_raw, shadow_clip_raw,
                  dynamic_range_raw, exposure_raw, contrast_raw,
                  saturation_raw, warmth_raw, sharpness_raw, noise_raw
    """
    # Convert to numpy array (H, W, 3) with values in [0, 255]
    img_array = np.array(image, dtype=np.float32)
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    
    # Extract RGB channels
    R = img_array[:, :, 0]
    G = img_array[:, :, 1]
    B = img_array[:, :, 2]
    
    # Compute luminance using ITU-R BT.709 weights
    Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
    
    metrics = {}
    
    # 1. Brightness: median luminance
    metrics['brightness_raw'] = float(np.median(Y))
    
    # 2. Clipping: fraction of pixels above/below thresholds
    metrics['highlight_clip_raw'] = float(np.mean(Y > 0.98))
    metrics['shadow_clip_raw'] = float(np.mean(Y < 0.02))
    
    # Dark fraction: fraction of pixels with Y < 0.15 (for silhouette detection)
    metrics['dark_fraction_raw'] = float(np.mean(Y < 0.15))
    
    # 3. Dynamic range (scene tonal spread proxy)
    # Note: This measures usable tonal spread in the scene, NOT camera sensor dynamic range
    p5 = np.percentile(Y, 5)
    p95 = np.percentile(Y, 95)
    metrics['dynamic_range_raw'] = float(p95 - p5)
    
    # 4. Exposure: combined penalty from clipping and brightness deviation
    clip_penalty = min(1.0, metrics['highlight_clip_raw'] + metrics['shadow_clip_raw'])
    mid_penalty = abs(metrics['brightness_raw'] - 0.5) * 2.0
    exposure_raw = 1.0 - (0.6 * clip_penalty + 0.4 * mid_penalty)
    metrics['exposure_raw'] = float(np.clip(exposure_raw, 0.0, 1.0))
    
    # 5. Contrast: combination of global std and local tile stds
    global_std = np.std(Y)
    
    # Compute 8x8 tile stds (handle small images)
    h, w = Y.shape
    tile_size = 8
    tile_stds = []
    
    for i in range(0, h, tile_size):
        for j in range(0, w, tile_size):
            tile = Y[i:min(i+tile_size, h), j:min(j+tile_size, w)]
            if tile.size > 0:
                tile_stds.append(np.std(tile))
    
    local_std = np.mean(tile_stds) if tile_stds else global_std
    metrics['contrast_raw'] = float(0.5 * global_std + 0.5 * local_std)
    
    # 6. Saturation: mean saturation from RGB→HSV, excluding very dark/bright pixels
    # Convert RGB to HSV
    max_rgb = np.maximum(np.maximum(R, G), B)
    min_rgb = np.minimum(np.minimum(R, G), B)
    delta = max_rgb - min_rgb
    
    # Saturation = delta / max_rgb (avoid division by zero)
    # Use np.divide with where to suppress warnings and handle edge cases
    eps = 1e-10
    saturation = np.divide(delta, max_rgb + eps, out=np.zeros_like(delta), where=(max_rgb > eps))
    
    # Exclude very dark (Y < 0.05) or very bright (Y > 0.95) pixels
    valid_mask = (Y >= 0.05) & (Y <= 0.95)
    
    if np.sum(valid_mask) == 0:
        # Grayscale image edge case
        logger.debug("No valid pixels for saturation calculation (grayscale image)")
        metrics['saturation_raw'] = 0.0
    else:
        metrics['saturation_raw'] = float(np.mean(saturation[valid_mask]))
    
    # 7. Warmth: mean of (R-B)/(R+B+eps) over midtones
    eps = 1e-6
    warmth_ratio = (R - B) / (R + B + eps)
    
    # Only use midtones (0.1 < Y < 0.9)
    midtone_mask = (Y > 0.1) & (Y < 0.9)
    
    if np.sum(midtone_mask) > 0:
        metrics['warmth_raw'] = float(np.mean(warmth_ratio[midtone_mask]))
    else:
        metrics['warmth_raw'] = 0.0
    
    # 8. Sharpness: combine Laplacian variance, Tenengrad, and edge density
    sharpness_score = _compute_sharpness(Y)
    metrics['sharpness_raw'] = float(sharpness_score)
    
    # 9. Centre-edge sharpness ratio: for DoF/bokeh detection
    center_edge_ratio = _compute_center_edge_sharpness_ratio(Y)
    metrics['center_edge_sharpness_ratio_raw'] = float(center_edge_ratio)
    
    # 10. Noise: MAD of high-pass residual in flat regions
    noise_score = _compute_noise(Y)
    metrics['noise_raw'] = float(noise_score)
    
    return metrics


def _compute_sharpness(Y: np.ndarray) -> float:
    """
    Compute sharpness metric combining multiple methods.
    
    Combines:
    - Laplacian variance
    - Tenengrad (Sobel gradient magnitude)
    - Edge density
    
    Args:
        Y: Luminance channel (normalized 0-1)
        
    Returns:
        Combined sharpness score (higher = sharper)
    """
    h, w = Y.shape
    
    # Laplacian kernel
    laplacian_kernel = np.array([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=np.float32)
    
    # Apply Laplacian via convolution
    laplacian = _convolve2d(Y, laplacian_kernel)
    laplacian_variance = np.var(laplacian)
    
    # Tenengrad: Sobel gradients
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)
    
    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float32)
    
    grad_x = _convolve2d(Y, sobel_x)
    grad_y = _convolve2d(Y, sobel_y)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    tenengrad = np.mean(grad_mag**2)
    
    # Edge density: fraction of pixels above threshold
    threshold = np.percentile(grad_mag, 90)
    edge_density = np.mean(grad_mag > threshold)
    
    # Normalize and combine (all components contribute equally)
    # Normalize each to [0, 1] range approximately
    laplacian_norm = np.clip(laplacian_variance * 100, 0, 1)
    tenengrad_norm = np.clip(tenengrad * 10, 0, 1)
    
    combined = (laplacian_norm + tenengrad_norm + edge_density) / 3.0
    
    return float(combined)


def _compute_noise(Y: np.ndarray) -> float:
    """
    Estimate noise level in flat regions.
    
    Algorithm:
    1. Compute gradient magnitude
    2. Select lowest 10% gradient pixels as "flat"
    3. Fallback: If flat pixels < 1% of image, use lowest 20%
    4. Compute high-pass residual (img - blur) on flat pixels
    5. Noise = MAD(residual) scaled to sigma estimate
    
    Args:
        Y: Luminance channel (normalized 0-1)
        
    Returns:
        Noise estimate (lower is better)
    """
    h, w = Y.shape
    total_pixels = h * w
    
    # Compute gradient magnitude
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)
    
    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float32)
    
    grad_x = _convolve2d(Y, sobel_x)
    grad_y = _convolve2d(Y, sobel_y)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    # Select flat regions: lowest 10% of gradient pixels
    flat_threshold_10 = np.percentile(grad_mag, 10)
    flat_mask_10 = grad_mag <= flat_threshold_10
    
    # Check if we have enough flat pixels (at least 1% of image)
    min_flat_pixels = max(1, int(total_pixels * 0.01))
    
    if np.sum(flat_mask_10) < min_flat_pixels:
        # Fallback: use lowest 20%
        flat_threshold_20 = np.percentile(grad_mag, 20)
        flat_mask = grad_mag <= flat_threshold_20
        logger.debug(f"Using 20% flat region threshold (had {np.sum(flat_mask_10)} pixels, need {min_flat_pixels})")
    else:
        flat_mask = flat_mask_10
    
    # Compute high-pass residual: image - blur
    # Simple box blur approximation (3x3)
    blur_kernel = np.ones((3, 3), dtype=np.float32) / 9.0
    blurred = _convolve2d(Y, blur_kernel)
    residual = Y - blurred
    
    # Extract residual only from flat regions
    flat_residual = residual[flat_mask]
    
    if len(flat_residual) == 0:
        # Fallback: use global estimate
        flat_residual = residual.flatten()
    
    # MAD (Median Absolute Deviation) scaled to sigma estimate
    # MAD ≈ 0.6745 * sigma for normal distribution
    mad = np.median(np.abs(flat_residual - np.median(flat_residual)))
    noise_estimate = mad / 0.6745 if mad > 0 else 0.0
    
    return float(noise_estimate)


def _compute_center_edge_sharpness_ratio(Y: np.ndarray) -> float:
    """
    Ratio of sharpness in a central region vs outer ring using Tenengrad energy.
    >1 suggests centre sharper than edges (possible shallow DoF).
    ~1 suggests uniform sharpness (deep DoF).
    <1 suggests edges sharper than centre (often composition, not DoF).
    """
    h, w = Y.shape
    if h < 32 or w < 32:
        return 1.0

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)

    gx = _convolve2d(Y, sobel_x)
    gy = _convolve2d(Y, sobel_y)
    grad2 = gx * gx + gy * gy  # Tenengrad energy per pixel

    # Central box: middle 40% of image
    ch0 = int(h * 0.30); ch1 = int(h * 0.70)
    cw0 = int(w * 0.30); cw1 = int(w * 0.70)

    center = grad2[ch0:ch1, cw0:cw1]
    center_energy = float(np.mean(center))

    # Outer ring = everything outside centre box
    ring_mask = np.ones((h, w), dtype=bool)
    ring_mask[ch0:ch1, cw0:cw1] = False
    ring = grad2[ring_mask]
    if ring.size < 10:
        return 1.0
    ring_energy = float(np.mean(ring))

    eps = 1e-12
    ratio = center_energy / (ring_energy + eps)

    # Optional: clamp to keep logs sane
    return float(np.clip(ratio, 0.0, 5.0))


def _convolve2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Fast 2D convolution using scipy (much faster than manual loops).
    
    Args:
        image: 2D array
        kernel: 2D kernel (odd dimensions)
        
    Returns:
        Convolved image (same size as input, zero-padded)
    """
    # Use scipy.signal.convolve2d (optimized C implementation)
    # This is 100-1000x faster than nested Python loops
    try:
        from scipy import signal
        return signal.convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)
    except ImportError:
        # If scipy not available, raise helpful error
        raise ImportError(
            "scipy is required for fast convolution. Install with: pip install scipy\n"
            "The convolution operations are too slow without scipy."
        )

