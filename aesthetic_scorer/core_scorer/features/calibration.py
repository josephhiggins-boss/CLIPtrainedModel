"""
Calibration - Percentile-Based Calibration

Map raw CV metrics to 0-100 scores using percentile-based calibration.
Calibration is learned from a corpus of images and stored as JSON.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Metrics where lower values are better (will be inverted during calibration)
LOWER_IS_BETTER_METRICS = {
    'noise_raw',
    'highlight_clip_raw',
    'shadow_clip_raw'
}

# Metrics that are NOT quality scores - these are diagnostic/descriptive only
# These should NOT be calibrated as 0-100 quality scores
NON_QUALITY_METRICS = {
    'center_edge_sharpness_ratio_raw',  # Diagnostic: describes detail distribution, not quality (centre-edge)
    'dark_fraction_raw'  # Descriptive: fraction of dark pixels, not a quality metric
}

# All other metrics are higher-is-better (default)


def fit_calibration(metrics_list: List[Dict[str, float]]) -> Dict[str, Any]:
    """
    Build percentile-based calibration from a list of raw metric dictionaries.
    
    For each metric, computes 101 percentile values (0th through 100th percentile)
    which are used to map raw values to 0-100 scores.
    
    Args:
        metrics_list: List of dictionaries, each containing raw metrics from one image
                      Keys should be raw metric names (e.g., 'brightness_raw')
        
    Returns:
        Calibration parameters dictionary with:
        - metadata: date, num_images, version
        - metrics: dict mapping metric names to percentile arrays and lower_is_better flag
    """
    if not metrics_list:
        raise ValueError("metrics_list cannot be empty")
    
    # Collect all metric names
    all_metric_names = set()
    for metrics in metrics_list:
        all_metric_names.update(metrics.keys())
    
    # Build calibration for each metric
    calibration_metrics = {}
    
    for metric_name in sorted(all_metric_names):
        # Skip non-quality metrics (diagnostic/descriptive only)
        if metric_name in NON_QUALITY_METRICS:
            logger.debug(f"Skipping non-quality metric {metric_name} from calibration")
            continue
        
        # Collect all values for this metric
        values = []
        for metrics in metrics_list:
            if metric_name in metrics:
                val = metrics[metric_name]
                if not np.isnan(val) and not np.isinf(val):
                    values.append(val)
        
        if not values:
            logger.warning(f"No valid values found for metric {metric_name}, skipping")
            continue
        
        values = np.array(values)
        
        # Compute 101 percentiles (0, 1, 2, ..., 100)
        percentiles = []
        for p in range(101):
            percentile_value = np.percentile(values, p)
            percentiles.append(float(percentile_value))
        
        # Determine if lower is better
        lower_is_better = metric_name in LOWER_IS_BETTER_METRICS
        
        calibration_metrics[metric_name] = {
            'percentiles': percentiles,
            'lower_is_better': lower_is_better
        }
    
    # Build calibration dict
    calibration = {
        'metadata': {
            'date': datetime.now().isoformat(),
            'num_images': len(metrics_list),
            'version': '1.0'
        },
        'metrics': calibration_metrics
    }
    
    return calibration


def apply_calibration(raw_metrics: Dict[str, float], params: Dict[str, Any]) -> Dict[str, float]:
    """
    Map raw metrics to calibrated 0-100 scores using percentile interpolation.
    
    Args:
        raw_metrics: Dictionary of raw metric values (keys end with '_raw')
        params: Calibration parameters from fit_calibration() or load_calibration()
        
    Returns:
        Dictionary mapping metric names (without '_raw' suffix) to 0-100 scores
        Example: {'brightness': 78.5, 'exposure': 65.2, ...}
    """
    calibrated = {}
    metrics_calib = params.get('metrics', {})
    
    for raw_key, raw_value in raw_metrics.items():
        if raw_key not in metrics_calib:
            logger.debug(f"No calibration found for {raw_key}, skipping")
            continue
        
        calib_data = metrics_calib[raw_key]
        percentiles = calib_data['percentiles']
        lower_is_better = calib_data.get('lower_is_better', False)
        
        # Robust handling: clamp to [v0, v100]
        v0 = percentiles[0]
        v100 = percentiles[-1]
        
        # Handle edge case: all calibration values identical
        if v0 == v100:
            logger.debug(f"All calibration values identical for {raw_key}, returning neutral score 50")
            # Remove '_raw' suffix for output key
            output_key = raw_key.replace('_raw', '')
            calibrated[output_key] = 50.0
            continue
        
        # Clamp raw value to calibration range
        clamped_value = np.clip(raw_value, v0, v100)
        
        # Find percentile using linear interpolation
        # Find which two percentiles bracket the value
        score = _interpolate_percentile(clamped_value, percentiles)
        
        # Invert if lower is better
        if lower_is_better:
            score = 100.0 - score
        
        # Remove '_raw' suffix for output key
        output_key = raw_key.replace('_raw', '')
        calibrated[output_key] = float(score)
    
    return calibrated


def _interpolate_percentile(value: float, percentiles: List[float]) -> float:
    """
    Interpolate which percentile (0-100) a value corresponds to.
    
    Uses linear interpolation between percentile knots.
    Handles ties gracefully.
    
    Args:
        value: Raw metric value
        percentiles: List of 101 percentile values (0th through 100th)
        
    Returns:
        Percentile score (0-100)
    """
    # Find the two percentiles that bracket the value
    for i in range(len(percentiles) - 1):
        if percentiles[i] <= value <= percentiles[i + 1]:
            # Linear interpolation
            if percentiles[i + 1] == percentiles[i]:
                # Handle ties: return the lower percentile
                return float(i)
            
            # Interpolate between percentile i and i+1
            t = (value - percentiles[i]) / (percentiles[i + 1] - percentiles[i])
            return float(i + t)
    
    # Value is outside range (shouldn't happen after clamping, but handle it)
    if value <= percentiles[0]:
        return 0.0
    else:
        return 100.0


def load_calibration(json_path: Path) -> Dict[str, Any]:
    """
    Load calibration parameters from JSON file.
    
    Args:
        json_path: Path to calibration JSON file
        
    Returns:
        Calibration parameters dictionary
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {json_path}")
    
    with open(json_path, 'r') as f:
        calibration = json.load(f)
    
    logger.info(f"Loaded calibration from {json_path}")
    return calibration


def save_calibration(params: Dict[str, Any], json_path: Path) -> None:
    """
    Save calibration parameters to JSON file.
    
    Args:
        params: Calibration parameters from fit_calibration()
        json_path: Path to save JSON file (will create parent directories if needed)
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'w') as f:
        json.dump(params, f, indent=2)
    
    logger.info(f"Saved calibration to {json_path}")
