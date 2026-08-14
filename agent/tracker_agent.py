"""
Food Tracker & Vision Agent.
Orchestrates Multimodal Scene Decomposition, Local CV Verification, IFCT Grounding, and Logging.
"""
import io
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image
from sqlalchemy.ext.asyncio import AsyncSession
from agent.tools.classify_food_tool import classify_food_image
from agent.tools.lookup_nutrition_tool import lookup_nutrition_profile
from agent.tools.meal_logger_tool import log_meal_record

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

MULTIMODAL_PROMPT = """
You are an expert Clinical Nutritionist & Food Vision AI with deep knowledge of global and Indian cuisines.
Analyze this food photo. It may contain single or multi-item plates, Indian dishes, continental foods, home-cooked meals, snacks, fruits, or beverages.

For EACH distinct food item visible, extract:
1. "name": Specific, clear food name (e.g. "2 Boiled Eggs", "Poha with Peanuts", "Greek Yogurt with Berries", "Paneer Butter Masala", "Multigrain Bread Toast", "Black Coffee", "Chicken Biryani", "Mixed Fruit Bowl", "Dal Tadka")
2. "estimated_grams": Realistic portion weight in grams for this serving size
3. "calories": Total estimated calories (kcal) for this portion
4. "protein_g": Protein in grams
5. "carbs_g": Total carbohydrates in grams
6. "fat_g": Total fat in grams
7. "fiber_g": Dietary fiber in grams
8. "iron_mg": Estimated iron in mg
9. "calcium_mg": Estimated calcium in mg
10. "vitamin_c_mg": Estimated Vitamin C in mg
11. "description": Visual details (e.g., "2 whole eggs boiled with yolk", "1 medium bowl of yellow poha with curry leaves and peanuts")

Return strictly a JSON object with this structure:
{
  "meal_type": "breakfast", // breakfast, lunch, dinner, or snack
  "overall_summary": "Summary of meal components",
  "items": [
    {
      "name": "Poha with Peanuts",
      "estimated_grams": 160.0,
      "calories": 240.0,
      "protein_g": 4.5,
      "carbs_g": 38.0,
      "fat_g": 8.0,
      "fiber_g": 3.0,
      "iron_mg": 2.1,
      "calcium_mg": 25.0,
      "vitamin_c_mg": 6.5,
      "description": "Standard portion of flattened rice with mustard seeds and roasted peanuts"
    }
  ]
}
"""


def safe_parse_json(text: str) -> Optional[Dict[str, Any]]:
    """Resiliently parses JSON from LLM outputs with markdown fence stripping and regex fallback."""
    if not text:
        return None
    raw_text = text.strip()
    if "```json" in raw_text:
        raw_text = raw_text.split("```json")[1].split("```")[0]
    elif "```" in raw_text:
        raw_text = raw_text.split("```")[1].split("```")[0]
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except Exception:
        pass

    import re
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    import ast
    try:
        res = ast.literal_eval(raw_text)
        if isinstance(res, dict):
            return res
    except Exception:
        pass
    return None


class FoodTrackerAgent:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initializes Google GenAI client if API key is provided."""
        self.api_key = self.api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
                logger.info("Google GenAI client initialized successfully.")
            except Exception as e:
                logger.warning(f"Could not initialize GenAI client: {e}. Running in local CV fallback mode.")
                self._client = None
        else:
            logger.info("No GEMINI_API_KEY detected. Agent running in autonomous local CV mode.")

    async def _decompose_plate_multimodal(
        self, image: Image.Image
    ) -> Optional[Dict[str, Any]]:
        """Invokes Gemini Multimodal with image pre-scaling and token limits for lowest latency."""
        if self._client is None:
            return None

        # Candidate models to try in order
        candidate_models = [
            self.model_name,
            "gemini-flash-latest",
            "gemini-3.1-flash-lite-preview",
            "gemini-3-flash-preview",
        ]
        seen = set()
        models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]

        # High-performance image pre-scaling (reduces payload by 85-95%, speeds up upload & vision tokens)
        img_for_api = image.copy()
        img_for_api.thumbnail((768, 768), Image.Resampling.LANCZOS)
        img_byte_arr = io.BytesIO()
        img_for_api.save(img_byte_arr, format="JPEG", quality=82, optimize=True)
        img_bytes = img_byte_arr.getvalue()

        from google.genai import types

        for model_id in models_to_try:
            try:
                response = self._client.models.generate_content(
                    model=model_id,
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                        MULTIMODAL_PROMPT,
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                        max_output_tokens=1024,
                    ),
                )
                if response.text:
                    data = safe_parse_json(response.text)
                    if data and "items" in data:
                        logger.info(f"Multimodal decomposition succeeded with model '{model_id}'")
                        return data
            except Exception as e:
                logger.warning(f"Multimodal call to model '{model_id}' failed: {e}. Trying next candidate...")

        logger.error("All multimodal models failed. Falling back to local classifier.")
        return None

    async def analyze_and_log_food_photo(
        self,
        image_bytes: bytes,
        telegram_id: int,
        session: AsyncSession,
    ) -> Dict[str, Any]:
        """
        Complete Hybrid Vision + Agent pipeline optimized for sub-second latency:
        1. Runs Local EfficientNet-B0 and Gemini Multimodal in parallel.
        2. Grounds dishes via In-Memory IFCT database cache.
        3. Persists Meal Record.
        """
        import asyncio
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # 1. Run Local EfficientNet-B0 and Gemini Multimodal concurrently in parallel!
        loop = asyncio.get_running_loop()
        local_cv_task = loop.run_in_executor(None, classify_food_image, image, 3)
        multimodal_task = self._decompose_plate_multimodal(image)

        local_cv_result, decomposed_data = await asyncio.gather(local_cv_task, multimodal_task)

        items_to_ground: List[Dict[str, Any]] = []
        meal_type = "meal"

        if decomposed_data and "items" in decomposed_data and len(decomposed_data["items"]) > 0:
            meal_type = decomposed_data.get("meal_type", "meal")
            for item in decomposed_data["items"]:
                items_to_ground.append(
                    {
                        "name": item.get("name", "Unknown Dish"),
                        "portion_grams": float(item.get("estimated_grams", 150.0)),
                        "calories": float(item.get("calories", 0.0)),
                        "protein": float(item.get("protein_g", 0.0)),
                        "carbs": float(item.get("carbs_g", 0.0)),
                        "fat": float(item.get("fat_g", 0.0)),
                        "fiber": float(item.get("fiber_g", 0.0)),
                        "iron_mg": float(item.get("iron_mg", 0.0)),
                        "calcium_mg": float(item.get("calcium_mg", 0.0)),
                        "vitamin_c_mg": float(item.get("vitamin_c_mg", 0.0)),
                        "description": item.get("description", ""),
                    }
                )
        else:
            # Fallback to local CV top-1 prediction
            top_dish = local_cv_result["top_prediction"]
            default_portion = float(local_cv_result["default_portion_grams"])
            items_to_ground.append(
                {
                    "name": top_dish,
                    "portion_grams": default_portion,
                    "description": f"Classified via local EfficientNet (Confidence: {local_cv_result['confidence']*100:.1f}%)",
                }
            )

        # 3. Ground each item in the IFCT Database (or use Gemini Open-World Estimate)
        grounded_items = []
        for itm in items_to_ground:
            nutrients = await lookup_nutrition_profile(
                food_name=itm["name"],
                portion_grams=itm["portion_grams"],
                session=session,
            )

            # If not found in IFCT, but Gemini provided estimates, use Gemini's open-world numbers
            if not nutrients.get("is_grounded_in_ifct") and itm.get("calories", 0) > 0:
                nutrients["calories"] = round(itm["calories"], 1)
                nutrients["protein"] = round(itm["protein"], 1)
                nutrients["carbs"] = round(itm["carbs"], 1)
                nutrients["fat"] = round(itm["fat"], 1)
                nutrients["fiber"] = round(itm.get("fiber", 0.0), 1)
                nutrients["iron_mg"] = round(itm.get("iron_mg", 0.0), 2)
                nutrients["calcium_mg"] = round(itm.get("calcium_mg", 0.0), 1)
                nutrients["vitamin_c_mg"] = round(itm.get("vitamin_c_mg", 0.0), 1)
                nutrients["source"] = "Gemini Multimodal Open-World Nutrition"

            nutrients["description"] = itm.get("description", "")
            nutrients["confidence"] = local_cv_result.get("confidence", 0.9)
            grounded_items.append(nutrients)

        # 4. Commit meal record to database
        raw_analysis_str = json.dumps(
            {
                "multimodal_decomposition": decomposed_data,
                "local_cv_verification": local_cv_result,
            }
        )

        log_result = await log_meal_record(
            telegram_id=telegram_id,
            items_breakdown=grounded_items,
            meal_type=meal_type,
            raw_ai_analysis=raw_analysis_str,
            session=session,
        )

        return {
            "success": True,
            "meal_id": log_result["meal_id"],
            "meal_type": meal_type,
            "items": grounded_items,
            "total_calories": log_result["total_calories"],
            "total_protein": log_result["total_protein"],
            "total_carbs": log_result["total_carbs"],
            "total_fat": log_result["total_fat"],
            "total_fiber": log_result["total_fiber"],
            "local_cv_verification": local_cv_result,
        }
