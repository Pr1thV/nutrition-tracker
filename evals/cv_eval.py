"""
Offline Computer Vision Model Benchmark Runner.
Calculates Top-1 Accuracy, Top-3 Accuracy, and Macro-F1 across Indian Food Classes.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict

from vision.config import VisionConfig
from vision.evaluate import evaluate_classifier

logger = logging.getLogger(__name__)


def run_cv_benchmark(model_path: str = None, synthetic: bool = True) -> Dict[str, float]:
    """Runs the offline CV classification benchmark."""
    logger.info("Starting Offline Computer Vision Evaluation Benchmark...")
    p = Path(model_path) if model_path else None
    results = evaluate_classifier(model_path=p, use_synthetic=synthetic)

    logger.info(f"CV Benchmark Completed. Top-1 Accuracy: {results['top_1_accuracy']}, Macro-F1: {results['macro_f1_score']}")
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Run CV Benchmark")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true", default=True)
    args = parser.parse_args()

    run_cv_benchmark(model_path=args.model_path, synthetic=args.synthetic)
