"""
EfficientNet-B0 Transfer Learning Architecture for Indian Food Classification.
"""
import logging
from typing import Optional, Tuple

import tensorflow as tf
from vision.config import VisionConfig, default_vision_config
from vision.dataset import build_augmentation_layer

logger = logging.getLogger(__name__)


def build_food_classifier(
    num_classes: int,
    config: VisionConfig = default_vision_config,
    include_augmentation: bool = True,
    pretrained_weights: str = "imagenet",
) -> tf.keras.Model:
    """
    Builds an EfficientNet-B0 transfer learning classification model.

    Architecture highlights:
    1. Input layer (224x224x3)
    2. Data Augmentation layer (active during training only)
    3. Pretrained EfficientNetB0 Feature Extractor (ImageNet weights)
    4. Global Average Pooling (spatial dimension reduction)
    5. Batch Normalization & Dropout (regularization against overfitting)
    6. Dense 256-unit Projection with L2 penalty
    7. Softmax output layer for multi-class probability distribution
    """
    inputs = tf.keras.Input(shape=config.input_shape, name="image_input")

    x = inputs
    if include_augmentation:
        augmentation_layer = build_augmentation_layer(config)
        x = augmentation_layer(x)

    # EfficientNet has built-in normalization, but scaling is handled inside the application layer
    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights=pretrained_weights,
        input_tensor=x,
        pooling=None,
    )
    base_model.trainable = False  # Start frozen for Stage 1

    features = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(features)
    x = tf.keras.layers.BatchNormalization(name="batch_norm_head")(x)
    x = tf.keras.layers.Dense(
        config.dense_units,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(config.weight_decay),
        name="dense_projection",
    )(x)
    x = tf.keras.layers.Dropout(config.dropout_rate, name="dropout_head")(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        dtype="float32",
        name="food_class_probabilities",
    )(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="IndianFoodEfficientNet")
    logger.info(f"Initialized {model.name} with {num_classes} output classes.")
    return model


def freeze_backbone(model: tf.keras.Model) -> None:
    """Freezes all layers belonging to the base feature extractor."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "efficientnet" in layer.name.lower():
            layer.trainable = False
    logger.info("Base backbone frozen for feature extraction (Stage 1).")


def unfreeze_top_layers(model: tf.keras.Model, unfreeze_count: int = 30) -> None:
    """
    Unfreezes the top `unfreeze_count` layers of the backbone for fine-tuning.
    Keeps BatchNormalization layers frozen to stabilize running mean and variance stats.
    """
    # Locate base model inside the functional graph
    base_model = None
    for layer in model.layers:
        if "efficientnet" in layer.name.lower() or isinstance(layer, tf.keras.Model):
            base_model = layer
            break

    if base_model is None:
        logger.warning("Could not identify distinct base model layer. Unfreezing top model layers directly.")
        for layer in model.layers[-unfreeze_count:]:
            if not isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = True
        return

    base_model.trainable = True
    # Freeze earlier layers, unfreeze only top layers
    for layer in base_model.layers[:-unfreeze_count]:
        layer.trainable = False

    # Keep BatchNormalization in inference mode even when fine-tuning
    for layer in base_model.layers[-unfreeze_count:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    logger.info(f"Unfroze top {unfreeze_count} layers of base model for fine-tuning (Stage 2).")
