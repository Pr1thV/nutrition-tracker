"""
Inference engine for Indian Food Classification.
Accepts image paths or raw byte streams and returns Top-K predictions with confidence scores.
"""
import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
import tensorflow as tf
from vision.config import VisionConfig, default_vision_config
from vision.dataset import load_class_names

logger = logging.getLogger(__name__)


class FoodClassifierInference:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        config: VisionConfig = default_vision_config,
    ):
        self.config = config
        self.model_path = model_path or config.model_save_path
        self.model: Optional[tf.keras.Model] = None
        self.class_names = load_class_names(config)
        self.display_names: Dict[str, str] = {}
        self.default_portions: Dict[str, int] = {}

        self._load_metadata()
        self._load_model()

    def _load_metadata(self) -> None:
        """Loads human-readable display names, default portion weights, and trained classes."""
        # 1. Load trained classes mapping if available
        trained_classes_file = self.model_path.parent / "trained_classes.json"
        if trained_classes_file.exists():
            try:
                with open(trained_classes_file, "r", encoding="utf-8") as f:
                    t_data = json.load(f)
                    self.class_names = t_data.get("classes", self.class_names)
            except Exception as e:
                logger.warning(f"Could not load trained_classes.json: {e}")

        # 2. Load display names and portions
        if self.config.classes_file.exists():
            with open(self.config.classes_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.display_names = data.get("class_to_display_name", {})
                self.default_portions = data.get("default_portion_grams", {})

    def _load_model(self) -> None:
        """Loads lightweight TFLite model (<5MB, <30MB RAM) or falls back to Keras graph."""
        tflite_path = self.model_path.parent / "indian_food_efficientnet.tflite"
        if tflite_path.exists():
            logger.info(f"Loading lightweight TFLite model from {tflite_path}")
            try:
                self.interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                self.is_tflite = True
                return
            except Exception as e:
                logger.warning(f"Failed to initialize TFLite interpreter: {e}")

        self.is_tflite = False
        if self.model_path.exists():
            logger.info(f"Loading trained weights from {self.model_path}")
            self.model = tf.keras.models.load_model(str(self.model_path))
        else:
            logger.warning(
                f"Model file {self.model_path} not found. Initializing feature extraction baseline."
            )
            from vision.model import build_food_classifier
            self.model = build_food_classifier(
                num_classes=len(self.class_names),
                config=self.config,
                include_augmentation=False,
            )

    def preprocess_image(self, image_source: Union[str, Path, bytes, Image.Image]) -> np.ndarray:
        """Loads and normalizes an input image to (1, 224, 224, 3)."""
        if isinstance(image_source, (str, Path)):
            img = Image.open(image_source).convert("RGB")
        elif isinstance(image_source, bytes):
            img = Image.open(io.BytesIO(image_source)).convert("RGB")
        elif isinstance(image_source, Image.Image):
            img = image_source.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_source)}")

        img = img.resize((self.config.img_width, self.config.img_height), Image.Resampling.BILINEAR)
        img_array = np.array(img, dtype=np.float32)
        # Expand batch dimension: (1, 224, 224, 3)
        img_batch = np.expand_dims(img_array, axis=0)
        return img_batch

    def predict(
        self,
        image_source: Union[str, Path, bytes, Image.Image],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Runs model inference on the given image.
        Returns a ranked list of predictions with class ID, display name, confidence, and default portion.
        """
        input_batch = self.preprocess_image(image_source)

        if getattr(self, "is_tflite", False):
            self.interpreter.set_tensor(self.input_details[0]["index"], input_batch)
            self.interpreter.invoke()
            probabilities = self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        else:
            if self.model is None:
                raise RuntimeError("Model is not initialized.")
            probabilities = self.model.predict(input_batch, verbose=0)[0]

        top_indices = np.argsort(probabilities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            class_key = self.class_names[idx] if idx < len(self.class_names) else f"class_{idx}"
            confidence = float(probabilities[idx])
            display_name = self.display_names.get(class_key, class_key.replace("_", " ").title())
            portion_g = self.default_portions.get(class_key, 150)

            results.append(
                {
                    "class_name": class_key,
                    "display_name": display_name,
                    "confidence": round(confidence, 4),
                    "confidence_percentage": f"{round(confidence * 100, 1)}%",
                    "default_portion_grams": portion_g,
                }
            )

        return results


# Global singleton instance for easy import in agent tools
_classifier_instance: Optional[FoodClassifierInference] = None


def get_food_classifier() -> FoodClassifierInference:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = FoodClassifierInference()
    return _classifier_instance


def classify_food_image(
    image_source: Union[str, Path, bytes, Image.Image],
    top_k: int = 3,
) -> Dict[str, Any]:
    """Convenience helper to classify a single image and return structured prediction metadata."""
    classifier = get_food_classifier()
    predictions = classifier.predict(image_source, top_k=top_k)
    top_pred = predictions[0] if predictions else {}
    return {
        "top_prediction": top_pred.get("display_name", "Unknown Indian Dish"),
        "top_class": top_pred.get("class_name", "unknown"),
        "confidence": top_pred.get("confidence", 0.0),
        "confidence_percentage": top_pred.get("confidence_percentage", "0%"),
        "default_portion_grams": top_pred.get("default_portion_grams", 150),
        "top_k": predictions,
    }
