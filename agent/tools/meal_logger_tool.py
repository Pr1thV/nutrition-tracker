"""
Tool: Database Meal Logging Tool
Persists meal records and individual item nutrition breakdowns into the database.
"""
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from db.models import Meal, MealItem, User

logger = logging.getLogger(__name__)


async def log_meal_record(
    telegram_id: int,
    items_breakdown: List[Dict[str, Any]],
    meal_type: str = "meal",
    image_path: Optional[str] = None,
    raw_ai_analysis: Optional[str] = None,
    session: Optional[AsyncSession] = None,
) -> Dict[str, Any]:
    """
    Creates a new Meal record and associated MealItem entries for the specified user.
    """
    if session is None:
        raise ValueError("Active database session required for logging.")

    # 1. Fetch or create user
    user_query = select(User).where(User.telegram_id == telegram_id)
    result = await session.execute(user_query)
    user = result.scalar_one_or_none()

    if user is None:
        user = User(telegram_id=telegram_id)
        session.add(user)
        await session.flush()

    # 2. Aggregate totals
    total_cal = sum(item.get("calories", 0.0) for item in items_breakdown)
    total_protein = sum(item.get("protein", 0.0) for item in items_breakdown)
    total_carbs = sum(item.get("carbs", 0.0) for item in items_breakdown)
    total_fat = sum(item.get("fat", 0.0) for item in items_breakdown)
    total_fiber = sum(item.get("fiber", 0.0) for item in items_breakdown)

    # 3. Create Meal (use timezone-naive UTC for cross-DB PostgreSQL & SQLite compatibility)
    now_utc_naive = datetime.now(timezone.utc).replace(tzinfo=None)
    meal = Meal(
        user_id=user.id,
        meal_type=meal_type,
        logged_at=now_utc_naive,
        total_calories=round(total_cal, 1),
        total_protein=round(total_protein, 1),
        total_carbs=round(total_carbs, 1),
        total_fat=round(total_fat, 1),
        total_fiber=round(total_fiber, 1),
        image_path=image_path,
        raw_ai_analysis=raw_ai_analysis,
    )
    session.add(meal)
    await session.flush()

    # 4. Create MealItems
    meal_items = []
    for item in items_breakdown:
        m_item = MealItem(
            meal_id=meal.id,
            food_name=item.get("food_name", "Unknown Food"),
            portion_grams=item.get("portion_grams", 100.0),
            calories=item.get("calories", 0.0),
            protein=item.get("protein", 0.0),
            carbs=item.get("carbs", 0.0),
            fat=item.get("fat", 0.0),
            fiber=item.get("fiber", 0.0),
            iron_mg=item.get("iron_mg", 0.0),
            calcium_mg=item.get("calcium_mg", 0.0),
            vitamin_c_mg=item.get("vitamin_c_mg", 0.0),
            vitamin_a_ug=item.get("vitamin_a_ug", 0.0),
            sodium_mg=item.get("sodium_mg", 0.0),
            confidence_score=item.get("confidence", 1.0),
        )
        session.add(m_item)
        meal_items.append(m_item)

    await session.commit()
    logger.info(f"Logged Meal {meal.id} for user {telegram_id} ({total_cal} kcal)")

    return {
        "success": True,
        "meal_id": meal.id,
        "total_calories": round(total_cal, 1),
        "total_protein": round(total_protein, 1),
        "total_carbs": round(total_carbs, 1),
        "total_fat": round(total_fat, 1),
        "total_fiber": round(total_fiber, 1),
        "item_count": len(meal_items),
    }
