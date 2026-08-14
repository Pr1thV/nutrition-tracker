"""
Tool: Daily Nutrition Summary & Goal Evaluation Tool
Aggregates today's meals and compares consumed nutrients against user targets.
"""
from datetime import datetime, time, timezone
import logging
from typing import Any, Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from db.models import Meal, MealItem, User

logger = logging.getLogger(__name__)


async def get_daily_nutrition_summary(
    telegram_id: int,
    session: AsyncSession,
) -> Dict[str, Any]:
    """
    Computes total calories, macros, and key micronutrients consumed today (from 00:00 to now),
    and compares them to daily targets.
    """
    # 1. Fetch user
    user_q = select(User).where(User.telegram_id == telegram_id)
    user_res = await session.execute(user_q)
    user = user_res.scalar_one_or_none()

    if user is None:
        return {
            "error": "User profile not found. Send /start to begin!",
            "has_meals_today": False,
        }

    # 2. Today's timestamp boundaries
    now_utc = datetime.now(timezone.utc)
    today_start = datetime.combine(now_utc.date(), time.min).replace(tzinfo=timezone.utc)
    today_end = datetime.combine(now_utc.date(), time.max).replace(tzinfo=timezone.utc)

    # 3. Query today's meals with child items
    meals_q = (
        select(Meal)
        .where(Meal.user_id == user.id, Meal.logged_at >= today_start, Meal.logged_at <= today_end)
        .options(selectinload(Meal.items))
    )
    meals_res = await session.execute(meals_q)
    meals = list(meals_res.scalars().all())

    # 4. Aggregate totals
    consumed_cal = sum(m.total_calories for m in meals)
    consumed_protein = sum(m.total_protein for m in meals)
    consumed_carbs = sum(m.total_carbs for m in meals)
    consumed_fat = sum(m.total_fat for m in meals)
    consumed_fiber = sum(m.total_fiber for m in meals)

    # Micronutrients aggregation
    total_iron = 0.0
    total_calcium = 0.0
    total_vit_c = 0.0
    total_vit_a = 0.0
    total_sodium = 0.0

    all_meal_items = []
    for m in meals:
        for itm in m.items:
            total_iron += itm.iron_mg
            total_calcium += itm.calcium_mg
            total_vit_c += itm.vitamin_c_mg
            total_vit_a += itm.vitamin_a_ug
            total_sodium += itm.sodium_mg
            all_meal_items.append(
                {
                    "name": itm.food_name,
                    "portion_g": itm.portion_grams,
                    "calories": itm.calories,
                    "protein": itm.protein,
                }
            )

    cal_goal = user.daily_calorie_goal
    protein_goal = user.daily_protein_goal
    carbs_goal = user.daily_carbs_goal
    fat_goal = user.daily_fat_goal

    remaining_cal = max(0.0, cal_goal - consumed_cal)
    cal_percentage = min(100.0, round((consumed_cal / cal_goal) * 100, 1)) if cal_goal > 0 else 0.0

    return {
        "has_meals_today": len(meals) > 0,
        "meals_logged_count": len(meals),
        "consumed": {
            "calories": round(consumed_cal, 1),
            "protein_g": round(consumed_protein, 1),
            "carbs_g": round(consumed_carbs, 1),
            "fat_g": round(consumed_fat, 1),
            "fiber_g": round(consumed_fiber, 1),
            "iron_mg": round(total_iron, 2),
            "calcium_mg": round(total_calcium, 1),
            "vitamin_c_mg": round(total_vit_c, 1),
            "vitamin_a_ug": round(total_vit_a, 1),
            "sodium_mg": round(total_sodium, 1),
        },
        "targets": {
            "calories": round(cal_goal, 1),
            "protein_g": round(protein_goal, 1),
            "carbs_g": round(carbs_goal, 1),
            "fat_g": round(fat_goal, 1),
        },
        "remaining_calories": round(remaining_cal, 1),
        "calorie_progress_pct": cal_percentage,
        "items_today": all_meal_items,
    }
