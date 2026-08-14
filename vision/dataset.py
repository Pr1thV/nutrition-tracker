"""
Dataset loading, preprocessing, and augmentation pipelines for Indian Food Classification.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
from vision.config import VisionConfig, default_vision_config

logger = logging.getLogger(__name__)


def load_class_names(config: VisionConfig = default_vision_config) -> List[str]:
    """Load class names list from configuration or JSON metadata."""
    if config.classes_file.exists():
        with open(config.classes_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("classes", [])
    return [f"class_{i}" for i in range(config.num_classes)]


def build_augmentation_layer(config: VisionConfig = default_vision_config) -> tf.keras.Sequential:
    """
    Constructs a Keras data augmentation pipeline.
    Augmentation improves generalization against varied kitchen lighting, camera angles, and plate orientations.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(config.random_flip),
            tf.keras.layers.RandomRotation(config.random_rotation),
            tf.keras.layers.RandomZoom(config.random_zoom[0], config.random_zoom[1]),
            tf.keras.layers.RandomBrightness(config.random_brightness),
        ],
        name="data_augmentation",
    )


def create_dataset_from_directory(
    data_dir: Path,
    config: VisionConfig = default_vision_config,
    validation_split: float = 0.2,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Loads train and validation datasets from an image directory structure.
    Expected directory format:
    data_dir/
      ├── biryani_chicken/
      │   ├── 001.jpg
      ├── dal_tadka/
      ...
    """
    logger.info(f"Loading image datasets from: {data_dir}")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(data_dir),
        labels="inferred",
        label_mode="categorical",
        image_size=(config.img_height, config.img_width),
        batch_size=config.batch_size,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        shuffle=True,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(data_dir),
        labels="inferred",
        label_mode="categorical",
        image_size=(config.img_height, config.img_width),
        batch_size=config.batch_size,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        shuffle=False,
    )

    class_names = train_ds.class_names
    autotune = tf.data.AUTOTUNE

    # Optimize data pipeline for training throughput
    train_ds = train_ds.cache().prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)

    return train_ds, val_ds, class_names


def generate_synthetic_benchmark_dataset(
    config: VisionConfig = default_vision_config,
    num_samples: int = 128,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, List[str]]:
    """
    Generates synthetic dummy images and one-hot labels for CI/CD test verification
    and rapid local prototyping when full image datasets are pending download.
    """
    class_names = load_class_names(config)[: config.num_classes]
    num_classes = len(class_names)

    # Generate synthetic image tensor batches (RGB 224x224)
    images = np.random.uniform(
        low=0.0, high=255.0, size=(num_samples, config.img_height, config.img_width, 3)
    ).astype(np.float32)

    # Random one-hot label distribution
    random_indices = np.random.randint(0, num_classes, size=(num_samples,))
    labels = tf.keras.utils.to_categorical(random_indices, num_classes=num_classes)

    split_idx = int(num_samples * 0.75)

    train_ds = (
        tf.data.Dataset.from_tensor_slices((images[:split_idx], labels[:split_idx]))
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices((images[split_idx:], labels[split_idx:]))
        .batch(config.batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, class_names
