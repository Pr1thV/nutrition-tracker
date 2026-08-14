"""
Vision model and training hyperparameters configuration.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"


@dataclass
class VisionConfig:
    # Image Input Specs
    img_height: int = 224
    img_width: int = 224
    num_channels: int = 3
    input_shape: Tuple[int, int, int] = (224, 224, 3)

    # Dataset Paths & Split Ratios
    dataset_dir: Path = (
        DATA_DIR / "Foodies_Challenge_Dataset"
        if (DATA_DIR / "Foodies_Challenge_Dataset").exists()
        else DATA_DIR / "indian_food_dataset"
    )
    classes_file: Path = DATA_DIR / "indian_food_classes.json"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Architecture & Base Model
    base_model_name: str = "EfficientNetB0"
    dense_units: int = 256
    dropout_rate: float = 0.3
    num_classes: int = 50

    # Training - Stage 1 (Frozen Backbone Feature Extraction)
    stage1_epochs: int = 10
    stage1_learning_rate: float = 1e-3
    batch_size: int = 32

    # Training - Stage 2 (Fine-tuning Unfrozen Top Layers)
    stage2_epochs: int = 15
    stage2_learning_rate: float = 1e-5
    unfreeze_layers_count: int = 30

    # Augmentation Settings
    random_rotation: float = 0.15  # +/- 15% (54 deg)
    random_zoom: Tuple[float, float] = (-0.15, 0.15)
    random_brightness: float = 0.15
    random_flip: str = "horizontal"

    # Optimization & Regularization
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    early_stopping_patience: int = 5
    reduce_lr_patience: int = 2
    reduce_lr_factor: float = 0.2

    # Export & Checkpoints
    model_save_path: Path = MODELS_DIR / "indian_food_efficientnet.keras"
    best_weights_path: Path = MODELS_DIR / "best_model.weights.h5"
    tflite_save_path: Path = MODELS_DIR / "indian_food_model.tflite"
    logs_dir: Path = BASE_DIR / "logs" / "fit"

    def __post_init__(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


default_vision_config = VisionConfig()
