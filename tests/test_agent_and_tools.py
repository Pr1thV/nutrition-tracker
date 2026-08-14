"""
Unit Tests for Agent System and Specialized Tools.
"""
import io
import numpy as np
from PIL import Image
import pytest
from agent.coach_agent import WellnessCoachAgent
from agent.root_dispatcher import NutritionAgentSystem
from agent.tools.classify_food_tool import classify_food_image
from agent.tools.daily_summary_tool import get_daily_nutrition_summary
from agent.tools.lookup_nutrition_tool import lookup_nutrition_profile
from agent.tools.meal_logger_tool import log_meal_record
from agent.tracker_agent import FoodTrackerAgent
from db.connection import AsyncSessionLocal, init_db
from db.seed_ifct import seed_ifct_database


@pytest.mark.asyncio
async def test_agent_tools_pipeline():
    await init_db()
    await seed_ifct_database()

    import random
    test_telegram_id = random.randint(100000000, 999999999)

    async with AsyncSessionLocal() as session:
        # 1. Test lookup tool
        nutr = await lookup_nutrition_profile("dal tadka", 150.0, session)
        assert nutr["is_grounded_in_ifct"] is True
        assert nutr["calories"] > 100.0

        # 2. Test meal logger tool
        items = [nutr]
        log_res = await log_meal_record(
            telegram_id=test_telegram_id,
            items_breakdown=items,
            meal_type="lunch",
            session=session,
        )
        assert log_res["success"] is True
        assert log_res["total_calories"] == nutr["calories"]

        # 3. Test daily summary tool
        summary = await get_daily_nutrition_summary(test_telegram_id, session)
        assert summary["has_meals_today"] is True
        assert summary["consumed"]["calories"] == nutr["calories"]
        assert summary["consumed"]["protein_g"] == nutr["protein"]


@pytest.mark.asyncio
async def test_tracker_agent_with_image_bytes():
    await init_db()
    await seed_ifct_database()

    import random
    test_telegram_id = random.randint(100000000, 999999999)

    # Create dummy image in memory
    dummy_img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    img_byte_arr = io.BytesIO()
    dummy_img.save(img_byte_arr, format="JPEG")
    img_bytes = img_byte_arr.getvalue()

    agent = FoodTrackerAgent()
    async with AsyncSessionLocal() as session:
        result = await agent.analyze_and_log_food_photo(
            image_bytes=img_bytes,
            telegram_id=test_telegram_id,
            session=session,
        )

        assert result["success"] is True
        assert "meal_id" in result
        assert result["total_calories"] > 0
        assert len(result["items"]) > 0


@pytest.mark.asyncio
async def test_coach_agent_query():
    await init_db()
    await seed_ifct_database()

    coach = WellnessCoachAgent()
    async with AsyncSessionLocal() as session:
        ans = await coach.answer_nutrition_query(
            user_message="How can I get more protein as a vegetarian?",
            telegram_id=12345678,
            session=session,
        )
        assert len(ans) > 20
        assert "protein" in ans.lower() or "paneer" in ans.lower() or "soya" in ans.lower()
