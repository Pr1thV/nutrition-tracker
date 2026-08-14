"""
Two-Stage Training Pipeline for Indian Food Classification.

Stage 1: Feature Extraction (Frozen Backbone, Fast Convergence)
Stage 2: Fine-Tuning (Unfrozen Top Layers, Low Learning Rate)
"""
import argparse
import logging
from pathlib import Path
from typing import Optional

import tensorflow as tf
from vision.config import VisionConfig, default_vision_config
from vision.dataset import (
    create_dataset_from_directory,
    generate_synthetic_benchmark_dataset,
    load_class_names,
)
from vision.model import build_food_classifier, freeze_backbone, unfreeze_top_layers

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def train_model(
    config: VisionConfig = default_vision_config,
    use_synthetic: bool = False,
    data_dir: Optional[Path] = None,
) -> tf.keras.Model:
    """
    Executes the two-stage progressive transfer learning workflow.
    """
    # 1. Dataset Loading
    if use_synthetic or data_dir is None or not data_dir.exists():
        logger.info("Using synthetic benchmark dataset for training verification.")
        train_ds, val_ds, class_names = generate_synthetic_benchmark_dataset(config)
    else:
        logger.info(f"Loading image data from {data_dir}")
        train_ds, val_ds, class_names = create_dataset_from_directory(data_dir, config)

    num_classes = len(class_names)
    logger.info(f"Target classification classes: {num_classes}")

    # 2. Build Model Architecture
    model = build_food_classifier(
        num_classes=num_classes,
        config=config,
        include_augmentation=True,
        pretrained_weights="imagenet" if not use_synthetic else None,
    )

    # 3. Stage 1: Feature Extraction (Backbone Frozen)
    logger.info("=== STAGE 1: Feature Extraction (Frozen Backbone) ===")
    freeze_backbone(model)

    stage1_optimizer = tf.keras.optimizers.Adam(learning_rate=config.stage1_learning_rate)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=config.label_smoothing)

    model.compile(
        optimizer=stage1_optimizer,
        loss=loss_fn,
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
        ],
    )

    callbacks_stage1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.reduce_lr_factor,
            patience=config.reduce_lr_patience,
            verbose=1,
        ),
    ]
    try:
        callbacks_stage1.append(tf.keras.callbacks.TensorBoard(log_dir=str(config.logs_dir / "stage1")))
    except Exception:
        logger.info("TensorBoard logging disabled.")

    history_stage1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.stage1_epochs,
        callbacks=callbacks_stage1,
    )

    # 4. Stage 2: Fine-Tuning (Unfreeze Top Layers)
    logger.info("=== STAGE 2: Fine-Tuning Top Convolutional Blocks ===")
    unfreeze_top_layers(model, unfreeze_count=config.unfreeze_layers_count)

    stage2_optimizer = tf.keras.optimizers.Adam(learning_rate=config.stage2_learning_rate)
    model.compile(
        optimizer=stage2_optimizer,
        loss=loss_fn,
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
        ],
    )

    callbacks_stage2 = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(config.model_save_path),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
    ]
    try:
        callbacks_stage2.append(tf.keras.callbacks.TensorBoard(log_dir=str(config.logs_dir / "stage2")))
    except Exception:
        pass

    history_stage2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.stage2_epochs,
        callbacks=callbacks_stage2,
    )

    # 5. Final Model Artifact & Class Names Save
    config.model_save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(config.model_save_path))

    classes_save_file = config.model_save_path.parent / "trained_classes.json"
    with open(classes_save_file, "w", encoding="utf-8") as f:
        json.dump({"classes": class_names, "num_classes": len(class_names)}, f, indent=2)

    logger.info(f"Model successfully saved to: {config.model_save_path}")
    logger.info(f"Trained classes mapping saved to: {classes_save_file}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Indian Food Classifier")
    parser.add_argument("--synthetic", action="store_true", help="Run with synthetic data benchmark")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to Indian food images dataset")
    parser.add_argument("--epochs-s1", type=int, default=5, help="Stage 1 Epochs")
    parser.add_argument("--epochs-s2", type=int, default=5, help="Stage 2 Epochs")
    args = parser.parse_args()

    custom_config = VisionConfig(
        stage1_epochs=args.epochs_s1,
        stage2_epochs=args.epochs_s2,
    )
    dataset_path = Path(args.data_dir) if args.data_dir else custom_config.dataset_dir
    train_model(config=custom_config, use_synthetic=args.synthetic, data_dir=dataset_path)
