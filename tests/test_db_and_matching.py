"""
Unit Tests for Database Models and Semantic IFCT Matching.
"""
import pytest
from db.connection import AsyncSessionLocal, init_db
from db.models import DishNutritionProfile, User
from db.seed_ifct import seed_ifct_database
from db.semantic_matcher import SemanticNutritionMatcher


@pytest.mark.asyncio
async def test_database_initialization_and_seeding():
    await init_db()
    inserted = await seed_ifct_database()
    assert inserted >= 0  # 0 if already seeded, >0 if fresh


@pytest.mark.asyncio
async def test_semantic_fuzzy_matching_exact_and_colloquial():
    await init_db()
    await seed_ifct_database()

    async with AsyncSessionLocal() as session:
        matcher = SemanticNutritionMatcher(session)

        # Test 1: Exact Key Match
        p1 = await matcher.find_profile("dal_tadka")
        assert p1 is not None
        assert "Dal Tadka" in p1.display_name

        # Test 2: Colloquial Hindi / English alias matches
        p2 = await matcher.find_profile("yellow dal fry")
        assert p2 is not None
        assert p2.dish_key == "dal_tadka"

        # Test 3: Roti alias match
        p3 = await matcher.find_profile("2 phulka rotis")
        assert p3 is not None
        assert p3.dish_key == "chapati_roti"

        # Test 4: Paneer variation match
        p4 = await matcher.find_profile("paneer makhani curry")
        assert p4 is not None
        assert p4.dish_key == "paneer_butter_masala"


@pytest.mark.asyncio
async def test_proportional_nutrient_scaling():
    await init_db()
    await seed_ifct_database()

    async with AsyncSessionLocal() as session:
        matcher = SemanticNutritionMatcher(session)
        p = await matcher.find_profile("chapati_roti")
        assert p is not None

        # 1 Roti (40g)
        scaled_40g = matcher.calculate_nutrients_for_portion(p, 40.0)
        assert scaled_40g["calories"] == round(p.calories_per_100g * 0.4, 1)
        assert scaled_40g["protein"] == round(p.protein_per_100g * 0.4, 1)

        # 2 Rotis (80g)
        scaled_80g = matcher.calculate_nutrients_for_portion(p, 80.0)
        assert scaled_80g["calories"] == round(p.calories_per_100g * 0.8, 1)
