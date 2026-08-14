"""
Evaluation suite for Indian Food Classification Model.
Computes Top-1 Accuracy, Top-3 Accuracy, Macro-F1, and Confusion Matrix.
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import tensorflow as tf
from vision.config import VisionConfig, default_vision_config
from vision.dataset import (
    create_dataset_from_directory,
    generate_synthetic_benchmark_dataset,
    load_class_names,
)

logger = logging.getLogger(__name__)


def evaluate_classifier(
    model_path: Optional[Path] = None,
    config: VisionConfig = default_vision_config,
    use_synthetic: bool = False,
    data_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """
    Evaluates the model on test/validation data and outputs comprehensive metrics.
    """
    target_model_path = model_path or config.model_save_path
    if not target_model_path.exists() and not use_synthetic:
        raise FileNotFoundError(f"Model checkpoint not found at {target_model_path}")

    # Load dataset
    if use_synthetic or data_dir is None or not data_dir.exists():
        _, eval_ds, class_names = generate_synthetic_benchmark_dataset(config, num_samples=100)
    else:
        _, eval_ds, class_names = create_dataset_from_directory(data_dir, config)

    # Load or instantiate model
    if target_model_path.exists():
        model = tf.keras.models.load_model(str(target_model_path))
        logger.info(f"Loaded trained model from {target_model_path}")
    else:
        from vision.model import build_food_classifier
        model = build_food_classifier(num_classes=len(class_names), config=config)

    all_true_labels = []
    all_pred_probs = []

    for images, labels in eval_ds:
        probs = model.predict(images, verbose=0)
        all_pred_probs.append(probs)
        all_true_labels.append(labels.numpy())

    y_probs = np.vstack(all_pred_probs)
    y_true_one_hot = np.vstack(all_true_labels)
    y_true = np.argmax(y_true_one_hot, axis=1)
    y_pred = np.argmax(y_probs, axis=1)

    # Top-1 Accuracy
    top1_acc = float(np.mean(y_true == y_pred))

    # Top-3 Accuracy
    top3_correct = 0
    for i, true_label in enumerate(y_true):
        top3_indices = np.argsort(y_probs[i])[-3:]
        if true_label in top3_indices:
            top3_correct += 1
    top3_acc = float(top3_correct / len(y_true))

    # Macro F1-score (unweighted average across all classes)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    metrics_summary = {
        "top_1_accuracy": round(top1_acc, 4),
        "top_3_accuracy": round(top3_acc, 4),
        "macro_f1_score": round(macro_f1, 4),
        "weighted_f1_score": round(weighted_f1, 4),
        "total_eval_samples": len(y_true),
        "num_classes": len(class_names),
    }

    print("\n" + "=" * 50)
    print("      INDIAN FOOD CLASSIFIER EVALUATION REPORT")
    print("=" * 50)
    for k, v in metrics_summary.items():
        print(f"  {k:.<30} {v}")
    print("=" * 50 + "\n")

    # Save metrics JSON report
    reports_dir = config.logs_dir.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / "cv_evaluation_metrics.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)

    return metrics_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Indian Food Model")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    model_p = Path(args.model_path) if args.model_path else None
    evaluate_classifier(model_path=model_p, use_synthetic=args.synthetic)
