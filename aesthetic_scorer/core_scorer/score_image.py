"""
Score Image CLI - Score a Single Image

Command-line script to score an image and display the result.

Usage:
    python core_scorer/score_image.py photo.jpg
    python core_scorer/score_image.py photo.jpg --model best_model.pth
    python core_scorer/score_image.py photo.jpg --model best_model.pth --features
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from core_scorer import ImageScorer


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description='Score image aesthetic quality using trained CLIP model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python core_scorer/score_image.py photo.jpg
  python core_scorer/score_image.py photo.jpg --model best_model.pth
  python core_scorer/score_image.py photo.jpg --model best_model.pth --features
  python core_scorer/score_image.py photo.jpg --device cuda
        """
    )
    
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to image file to score'
    )
    
    repo_root = Path(__file__).resolve().parent.parent.parent
    default_model = repo_root / 'best_model_complete.pth'
    parser.add_argument(
        '--model',
        type=str,
        default=str(default_model),
        help='Path to trained model checkpoint (default: repo root best_model_complete.pth)'
    )
    
    parser.add_argument(
        '--features',
        action='store_true',
        help='Show detailed feature breakdown (technical features and diagnostic info)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='Device to run on (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # validate paths
    image_path = Path(args.image_path)
    model_path = Path(args.model)
    
    if not image_path.exists():
        print(f"[ERROR] Image not found at {image_path}", file=sys.stderr)
        sys.exit(1)
    
    if not model_path.exists():
        print(f"[ERROR] Model not found at {model_path}", file=sys.stderr)
        sys.exit(1)
    
    # load model and score image
    try:
        print(f"Loading model from {model_path}...")
        scorer = ImageScorer(
            model_path=model_path,
            device=args.device,
            extract_features=args.features
        )
        print("[OK] Model loaded successfully\n")
        
        print(f"Scoring image: {image_path.name}")
        result = scorer.score_image(image_path)
        
        # display result
        print("=" * 60)
        print(f"  Aesthetic Score: {result['aesthetic_score']:.2f}%")
        print("=" * 60)
        
        if args.features:
            # raw metrics (for explainability)
            raw_metrics = result.get('raw_metrics', {})
            if raw_metrics:
                print("\nRaw Metrics (for reference):")
                for name, value in sorted(raw_metrics.items()):
                    print(f"  - {name}: {value:.4f}")
            
            # technical features (calibrated scores)
            technical = result.get('technical_features', {})
            if technical:
                print("\nTechnical Features (Calibrated Scores):")
                for name, value in sorted(technical.items()):
                    percentage = value * 100
                    # Show raw value if available
                    raw_key = f"{name}_raw"
                    if raw_key in raw_metrics:
                        raw_val = raw_metrics[raw_key]
                        print(f"  - {name.replace('_', ' ').title()}: {percentage:.1f}% (raw: {raw_val:.4f})")
                    else:
                        print(f"  - {name.replace('_', ' ').title()}: {percentage:.1f}%")
            
            # diagnostic information (not quality scores)
            diagnostic_info = result.get('diagnostic_info', {})
            if diagnostic_info:
                print("\nDiagnostic Information:")
                if 'center_edge_sharpness_ratio_raw' in diagnostic_info:
                    ratio = diagnostic_info['center_edge_sharpness_ratio_raw']
                    label = diagnostic_info.get('detail_distribution', 'unknown')
                    print(f"  - Centre-Edge Sharpness Ratio (raw): {ratio:.3f}")
                    print(f"  - Detail Distribution: {label}")
                if 'dark_fraction_raw' in diagnostic_info:
                    dark_frac = diagnostic_info['dark_fraction_raw']
                    darkness_label = diagnostic_info.get('darkness_level', 'unknown')
                    print(f"  - Dark Fraction (raw): {dark_frac:.3f}")
                    print(f"  - Darkness Level: {darkness_label}")
        
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] Error processing image: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()




