"""
Pydantic Schemas for NutritionTrackerAI REST API.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    user_id: int = Field(default=1, description="Unique user or session ID")
    message: str = Field(..., min_length=1, description="User nutrition or health question")


class ChatResponse(BaseModel):
    response: str
    user_id: int


class FeedbackRequest(BaseModel):
    user_id: int = Field(default=1)
    meal_id: Optional[int] = None
    is_accurate: bool = Field(..., description="True for positive feedback, False for negative")
    corrected_food_name: Optional[str] = None
    user_comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: int


class PortionAdjustRequest(BaseModel):
    meal_id: int
    scale_factor: float = Field(..., description="e.g. 0.75 for -25%, 1.50 for +50%, or custom ratio")


class NutrientItemSchema(BaseModel):
    food_name: str
    dish_key: str
    portion_grams: float
    calories: float
    protein: float
    carbs: float
    fat: float
    fiber: float
    iron_mg: float
    calcium_mg: float
    vitamin_c_mg: float
    vitamin_a_ug: float
    sodium_mg: float
    confidence: Optional[float] = 1.0
    is_grounded_in_ifct: Optional[bool] = True
    description: Optional[str] = None


class FoodAnalysisResponse(BaseModel):
    success: bool
    meal_id: int
    meal_type: str
    items: List[NutrientItemSchema]
    total_calories: float
    total_protein: float
    total_carbs: float
    total_fat: float
    total_fiber: float
    local_cv_verification: Optional[Dict[str, Any]] = None


class DailySummaryResponse(BaseModel):
    has_meals_today: bool = False
    meals_logged_count: int = 0
    consumed: Dict[str, float] = Field(default_factory=dict)
    targets: Dict[str, float] = Field(default_factory=dict)
    remaining_calories: float = 2000.0
    calorie_progress_pct: float = 0.0
    items_today: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
