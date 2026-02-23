"""
Features Package - CV-Based Technical Features

Provides deterministic computer vision metrics, calibration, and consistency guards.
"""

from .technicals import extract_raw_metrics, compute_detail_distribution_label, compute_dark_fraction_label
from .calibration import (
    fit_calibration,
    apply_calibration,
    load_calibration,
    save_calibration,
    NON_QUALITY_METRICS
)
from .guards import apply_guards

__all__ = [
    'extract_raw_metrics',
    'compute_detail_distribution_label',
    'compute_dark_fraction_label',
    'fit_calibration',
    'apply_calibration',
    'load_calibration',
    'save_calibration',
    'NON_QUALITY_METRICS',
    'apply_guards'
]

