"""
Unit Tests for Computer Vision Pipeline.
"""
import numpy as np
from PIL import Image
import pytest
from vision.config import VisionConfig
from vision.dataset import generate_synthetic_benchmark_dataset, load_class_names
from vision.model import build_food_classifier, freeze_backbone, unfreeze_top_layers
from vision.predict import FoodClassifierInference


def test_load_class_names():
    classes = load_class_names()
    assert isinstance(classes, list)
    assert len(classes) >= 50
    assert "dal_tadka" in classes
    assert "chapati_roti" in classes
    assert "biryani_chicken" in classes


def test_synthetic_dataset_generation():
    config = VisionConfig(batch_size=8, num_classes=10)
    train_ds, val_ds, class_names = generate_synthetic_benchmark_dataset(config, num_samples=16)

    for images, labels in train_ds.take(1):
        assert images.shape == (8, 224, 224, 3)
        assert labels.shape == (8, 10)


def test_model_architecture_and_fine_tuning_controls():
    config = VisionConfig(num_classes=10)
    model = build_food_classifier(
        num_classes=10,
        config=config,
        include_augmentation=False,
        pretrained_weights=None,
    )

    assert model.input_shape == (None, 224, 224, 3)
    assert model.output_shape == (None, 10)

    # Test backbone freezing
    freeze_backbone(model)
    # Test top-layer unfreezing
    unfreeze_top_layers(model, unfreeze_count=10)


def test_food_classifier_inference():
    config = VisionConfig(num_classes=50)
    classifier = FoodClassifierInference(config=config)

    # Create dummy RGB image in memory
    dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    predictions = classifier.predict(dummy_img, top_k=3)

    assert len(predictions) == 3
    assert "class_name" in predictions[0]
    assert "confidence" in predictions[0]
    assert "display_name" in predictions[0]
    assert 0.0 <= predictions[0]["confidence"] <= 1.0
