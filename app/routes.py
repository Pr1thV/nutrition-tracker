"""
FastAPI Route Handlers for Food Vision, Nutrition Tracking, Coach Chat, and Feedback.
"""
import logging
from typing import Any, Dict, List
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agent.root_dispatcher import agent_system
from agent.tools.daily_summary_tool import get_daily_nutrition_summary
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DailySummaryResponse,
    FeedbackRequest,
    FeedbackResponse,
    FoodAnalysisResponse,
    PortionAdjustRequest,
)
from db.connection import get_db_session
from db.models import DishNutritionProfile, Feedback, Meal

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["Nutrition & Vision"])


@router.post("/analyze-food", response_model=FoodAnalysisResponse)
async def analyze_food_image(
    file: UploadFile = File(...),
    user_id: int = Form(default=1),
    session: AsyncSession = Depends(get_db_session),
) -> Dict[str, Any]:
    """
    Upload a food image file to detect Indian dishes, decompose the plate,
    ground in ICMR-NIN IFCT tables, and return full macronutrients & micronutrients.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must be a valid image (JPEG/PNG/WebP).",
        )

    try:
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty image file.")

        result = await agent_system.handle_photo_upload(
            image_bytes=image_bytes,
            telegram_id=user_id,
            session=session,
        )
        return result
    except Exception as e:
        logger.error(f"Food analysis endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze food image: {str(e)}",
        )


@router.get("/daily-summary/{user_id}", response_model=DailySummaryResponse)
async def get_user_daily_summary(
    user_id: int,
    session: AsyncSession = Depends(get_db_session),
) -> Dict[str, Any]:
    """
    Retrieves today's aggregated nutritional intake vs daily goals for a given user.
    """
    # Auto-ensure user profile exists
    user_query = select(Meal).limit(1)  # lightweight check
    from db.models import User
    u = await session.get(User, user_id)
    if not u:
        # Check by telegram_id
        q = select(User).where(User.telegram_id == user_id)
        res = await session.execute(q)
        existing = res.scalar_one_or_none()
        if not existing:
            new_u = User(telegram_id=user_id, first_name="AppUser", daily_calorie_goal=2000.0)
            session.add(new_u)
            await session.commit()

    summary = await get_daily_nutrition_summary(telegram_id=user_id, session=session)
    return summary


@router.post("/chat", response_model=ChatResponse)
async def chat_with_wellness_coach(
    request: ChatRequest,
    session: AsyncSession = Depends(get_db_session),
) -> Dict[str, Any]:
    """
    Conversational health & nutrition coaching endpoint powered by Google ADK / Gemini.
    """
    try:
        reply = await agent_system.handle_text_message(
            text=request.message,
            telegram_id=request.user_id,
            session=session,
        )
        return {"response": reply, "user_id": request.user_id}
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Coach agent error.",
        )


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    session: AsyncSession = Depends(get_db_session),
) -> Dict[str, Any]:
    """
    Online evaluation feedback loop (Thumbs Up / Down) for accuracy tracking and drift detection.
    """
    feedback_entry = Feedback(
        user_id=request.user_id,
        meal_id=request.meal_id,
        is_accurate=request.is_accurate,
        corrected_food_name=request.corrected_food_name,
        user_comment=request.user_comment or "API feedback submission",
    )
    session.add(feedback_entry)
    await session.commit()
    return {"status": "Feedback recorded successfully", "feedback_id": feedback_entry.id}


@router.post("/adjust-portion")
async def adjust_portion_scale(
    request: PortionAdjustRequest,
    session: AsyncSession = Depends(get_db_session),
) -> Dict[str, Any]:
    """
    Recalculates a logged meal's macros when the user adjusts portion size.
    """
    query = select(Meal).where(Meal.id == request.meal_id)
    result = await session.execute(query)
    meal = result.scalar_one_or_none()

    if not meal:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Meal record not found.")

    meal.total_calories = round(meal.total_calories * request.scale_factor, 1)
    meal.total_protein = round(meal.total_protein * request.scale_factor, 1)
    meal.total_carbs = round(meal.total_carbs * request.scale_factor, 1)
    meal.total_fat = round(meal.total_fat * request.scale_factor, 1)
    await session.commit()

    return {
        "status": "success",
        "meal_id": meal.id,
        "new_calories": meal.total_calories,
        "new_protein": meal.total_protein,
        "new_carbs": meal.total_carbs,
        "new_fat": meal.total_fat,
    }


@router.get("/dishes")
async def list_ifct_dishes(
    session: AsyncSession = Depends(get_db_session),
) -> List[Dict[str, Any]]:
    """
    Lists all available grounded Indian dishes from the ICMR-NIN IFCT 2017 database.
    """
    result = await session.execute(select(DishNutritionProfile))
    dishes = result.scalars().all()
    return [
        {
            "dish_key": d.dish_key,
            "display_name": d.display_name,
            "category": d.category,
            "default_serving_grams": d.default_serving_grams,
            "calories_per_100g": d.calories_per_100g,
            "protein_per_100g": d.protein_per_100g,
            "carbs_per_100g": d.carbs_per_100g,
            "fat_per_100g": d.fat_per_100g,
            "fiber_per_100g": d.fiber_per_100g,
            "iron_mg_per_100g": d.iron_mg_per_100g,
            "calcium_mg_per_100g": d.calcium_mg_per_100g,
        }
        for d in dishes
    ]
