"""
Guards - Consistency Guards

Apply rule-based consistency guards to prevent contradictory outputs.
Guards operate on calibrated technical scores and raw metrics.
"""

from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


def apply_guards(
    raw_metrics: Dict[str, float],
    technical_scores: Dict[str, float]
) -> Tuple[Dict[str, float], List[str]]:
    """
    Apply consistency guards to prevent contradictory outputs.
    
    Guards check for logical inconsistencies and adjust scores accordingly.
    
    Args:
        raw_metrics: Dictionary of raw metric values (keys end with '_raw')
        technical_scores: Dictionary of calibrated technical scores (0-100)
        
    Returns:
        Tuple of:
        - Adjusted technical_scores
        - List of guard messages explaining what was adjusted
    """
    adjusted_technical = technical_scores.copy()
    guard_messages = []
    
    # Guard 1: Sharpness/Noise contradiction
    # If sharpness > 75 and noise > 70, this is contradictory - trust sharpness more
    if 'sharpness' in adjusted_technical and 'noise' in adjusted_technical:
        sharpness_score = adjusted_technical['sharpness']
        noise_score = adjusted_technical['noise']
        
        if sharpness_score > 75 and noise_score > 70:
            old_noise = noise_score
            adjusted_technical['noise'] = min(noise_score, 60.0)
            guard_messages.append(
                f"Adjusted noise from {old_noise:.1f} to {adjusted_technical['noise']:.1f} "
                f"(contradictory with high sharpness {sharpness_score:.1f})"
            )
    
    # Guard 2: Exposure validation
    # Low clipping alone doesn't guarantee good exposure - check clipping AND midtones
    highlight_clip = raw_metrics.get('highlight_clip_raw', 1.0)
    shadow_clip = raw_metrics.get('shadow_clip_raw', 1.0)
    brightness_raw = raw_metrics.get('brightness_raw', 0.5)
    
    if 'exposure' in adjusted_technical:
        exposure_score = adjusted_technical['exposure']
        
        # Check: low clipping AND midtones
        if (highlight_clip < 0.01 and shadow_clip < 0.01 and 
            abs(brightness_raw - 0.5) < 0.15):
            # Low clipping + midtone placement suggests good exposure
            if exposure_score < 80:
                old_exposure = exposure_score
                adjusted_technical['exposure'] = max(exposure_score, 80.0)
                guard_messages.append(
                    f"Adjusted exposure from {old_exposure:.1f} to {adjusted_technical['exposure']:.1f} "
                    f"(low clipping + midtones indicate good exposure)"
                )
    
    return adjusted_technical, guard_messages

