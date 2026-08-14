"""
Tool: Local Computer Vision Classifier Tool
Executes inference with fine-tuned EfficientNet-B0 on Indian food images.
"""
import logging
from typing import Any, Dict, List, Optional, Union
from PIL import Image
from vision.predict import get_food_classifier

logger = logging.getLogger(__name__)


def classify_food_image(
    image_input: Union[str, bytes, Image.Image],
    top_k: int = 3,
) -> Dict[str, Any]:
    """
    Runs the fine-tuned Indian Food EfficientNet-B0 classifier on an image.
    Returns the top-k predictions with confidence scores and standard portion weights.
    """
    try:
        classifier = get_food_classifier()
        predictions = classifier.predict(image_input, top_k=top_k)

        top_pred = predictions[0] if predictions else None
        return {
            "success": True,
            "top_prediction": top_pred["display_name"] if top_pred else "Unknown",
            "top_class_name": top_pred["class_name"] if top_pred else "unknown",
            "confidence": top_pred["confidence"] if top_pred else 0.0,
            "default_portion_grams": top_pred["default_portion_grams"] if top_pred else 150,
            "top_k_predictions": predictions,
        }
    except Exception as e:
        logger.error(f"Error in classify_food_image: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "top_prediction": "Indian Food",
            "confidence": 0.5,
            "default_portion_grams": 150,
            "top_k_predictions": [],
        }
