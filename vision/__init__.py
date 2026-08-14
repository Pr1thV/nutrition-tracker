"""
Vision Module for Indian Food Classification
"""
from vision.config import VisionConfig, default_vision_config
from vision.predict import FoodClassifierInference, get_food_classifier

__all__ = ["VisionConfig", "default_vision_config", "FoodClassifierInference", "get_food_classifier"]
