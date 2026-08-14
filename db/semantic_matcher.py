"""
Semantic and Fuzzy Matcher for Indian Dishes.
Solves the vocabulary mismatch between colloquial user descriptions, CV classifications,
and formal ICMR-NIN IFCT biochemical profiles.
"""
import difflib
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from db.models import DishNutritionProfile

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    """Normalizes string for comparison by lowercasing and removing punctuation."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


# In-memory global cache of IFCT dish profiles
_PROFILES_CACHE: Optional[List[DishNutritionProfile]] = None


def invalidate_ifct_cache():
    global _PROFILES_CACHE
    _PROFILES_CACHE = None


class SemanticNutritionMatcher:
    """
    Finds the best matching nutritional profile for a colloquial dish name,
    scales nutrients to the requested portion in grams, and returns structured breakdown.
    """

    def __init__(self, session: AsyncSession):
        self.session = session

    async def find_profile(self, food_query: str) -> Optional[DishNutritionProfile]:
        """
        Multi-stage resolution pipeline:
        1. Exact dish_key / display_name match
        2. Alias keyword containment match
        3. Fuzzy string similarity ranking
        """
        global _PROFILES_CACHE
        query_norm = clean_text(food_query)
        if not query_norm:
            return None

        # Fetch all profiles from memory cache or db
        if _PROFILES_CACHE is None or len(_PROFILES_CACHE) == 0:
            result = await self.session.execute(select(DishNutritionProfile))
            _PROFILES_CACHE = list(result.scalars().all())

        profiles = _PROFILES_CACHE

        if not profiles:
            return None

        # Stage 1: Exact or Direct Key Match
        for p in profiles:
            if p.dish_key == query_norm or clean_text(p.display_name) == query_norm:
                return p

        # Stage 2: Alias Keyword Containment
        for p in profiles:
            aliases_list = [clean_text(a) for a in p.aliases.split(",") if a.strip()]
            for alias in aliases_list:
                if alias == query_norm or query_norm in alias or alias in query_norm:
                    return p

        # Stage 3: Fuzzy Ratio Similarity
        best_match: Optional[DishNutritionProfile] = None
        highest_score = 0.0

        for p in profiles:
            candidates = [p.dish_key, p.display_name] + [a.strip() for a in p.aliases.split(",") if a.strip()]
            for cand in candidates:
                ratio = difflib.SequenceMatcher(None, query_norm, clean_text(cand)).ratio()
                if ratio > highest_score:
                    highest_score = ratio
                    best_match = p

        if highest_score >= 0.55:
            logger.info(f"Fuzzy matched '{food_query}' -> '{best_match.display_name}' (score: {highest_score:.2f})")
            return best_match

        logger.warning(f"No high-confidence match for '{food_query}'. Highest score was {highest_score:.2f}")
        return None

    def calculate_nutrients_for_portion(
        self, profile: DishNutritionProfile, portion_grams: float
    ) -> Dict[str, Any]:
        """
        Calculates exact macros and micronutrients scaled proportionally for portion_grams.
        Formula: nutrient = (nutrient_per_100g * portion_grams) / 100
        """
        factor = portion_grams / 100.0

        return {
            "food_name": profile.display_name,
            "dish_key": profile.dish_key,
            "portion_grams": round(portion_grams, 1),
            "calories": round(profile.calories_per_100g * factor, 1),
            "protein": round(profile.protein_per_100g * factor, 1),
            "carbs": round(profile.carbs_per_100g * factor, 1),
            "fat": round(profile.fat_per_100g * factor, 1),
            "fiber": round(profile.fiber_per_100g * factor, 1),
            # Key Micronutrients
            "iron_mg": round(profile.iron_mg_per_100g * factor, 2),
            "calcium_mg": round(profile.calcium_mg_per_100g * factor, 1),
            "vitamin_c_mg": round(profile.vitamin_c_mg_per_100g * factor, 1),
            "vitamin_a_ug": round(profile.vitamin_a_ug_per_100g * factor, 1),
            "sodium_mg": round(profile.sodium_mg_per_100g * factor, 1),
        }

    def generate_fallback_estimate(
        self, food_name: str, portion_grams: float = 150.0
    ) -> Dict[str, Any]:
        """
        Provides a conservative, balanced nutritional estimate when a food item is ungrounded.
        """
        factor = portion_grams / 100.0
        return {
            "food_name": food_name.title(),
            "dish_key": clean_text(food_name).replace(" ", "_"),
            "portion_grams": round(portion_grams, 1),
            "calories": round(140.0 * factor, 1),
            "protein": round(4.5 * factor, 1),
            "carbs": round(18.0 * factor, 1),
            "fat": round(5.5 * factor, 1),
            "fiber": round(2.0 * factor, 1),
            "iron_mg": round(1.2 * factor, 2),
            "calcium_mg": round(30.0 * factor, 1),
            "vitamin_c_mg": round(2.0 * factor, 1),
            "vitamin_a_ug": round(15.0 * factor, 1),
            "sodium_mg": round(200.0 * factor, 1),
        }
