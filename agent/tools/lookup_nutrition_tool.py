"""
Tool: IFCT Semantic Nutrition Lookup Tool
Queries the Indian Food Composition database with semantic/fuzzy resolution.
"""
import logging
from typing import Any, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from db.semantic_matcher import SemanticNutritionMatcher

logger = logging.getLogger(__name__)


async def lookup_nutrition_profile(
    food_name: str,
    portion_grams: float,
    session: AsyncSession,
) -> Dict[str, Any]:
    """
    Finds the IFCT nutritional profile for a given food name,
    scales calories and micronutrients to the exact portion_grams.
    """
    matcher = SemanticNutritionMatcher(session)
    profile = await matcher.find_profile(food_name)

    if profile is not None:
        nutrients = matcher.calculate_nutrients_for_portion(profile, portion_grams)
        nutrients["is_grounded_in_ifct"] = True
        nutrients["source"] = "ICMR-NIN IFCT 2017"
    else:
        nutrients = matcher.generate_fallback_estimate(food_name, portion_grams)
        nutrients["is_grounded_in_ifct"] = False
        nutrients["source"] = "Nutritional AI Fallback Estimation"

    return nutrients
