"""
SQLAlchemy ORM Models for NutritionTrackerAI.
Defines schemas for Users, Meals, MealItems, IFCT Nutritional Grounding Data, and Feedback.
"""
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def get_utc_now() -> datetime:
    """Returns timezone-naive UTC datetime for cross-DB compatibility."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    telegram_id: Mapped[int] = mapped_column(BigInteger, unique=True, index=True, nullable=False)
    username: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    first_name: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Daily Nutrition Targets
    daily_calorie_goal: Mapped[float] = mapped_column(Float, default=2000.0)
    daily_protein_goal: Mapped[float] = mapped_column(Float, default=75.0)
    daily_carbs_goal: Mapped[float] = mapped_column(Float, default=250.0)
    daily_fat_goal: Mapped[float] = mapped_column(Float, default=65.0)
    dietary_preference: Mapped[str] = mapped_column(String(50), default="omnivore")  # veg, non-veg, vegan

    created_at: Mapped[datetime] = mapped_column(DateTime, default=get_utc_now)

    # Relationships
    meals: Mapped[List["Meal"]] = relationship("Meal", back_populates="user", cascade="all, delete-orphan")
    feedback_entries: Mapped[List["Feedback"]] = relationship("Feedback", back_populates="user")


class Meal(Base):
    __tablename__ = "meals"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    meal_type: Mapped[str] = mapped_column(String(50), default="meal")  # breakfast, lunch, dinner, snack
    logged_at: Mapped[datetime] = mapped_column(DateTime, default=get_utc_now, index=True)

    # Aggregated Macro Totals for the Meal
    total_calories: Mapped[float] = mapped_column(Float, default=0.0)
    total_protein: Mapped[float] = mapped_column(Float, default=0.0)
    total_carbs: Mapped[float] = mapped_column(Float, default=0.0)
    total_fat: Mapped[float] = mapped_column(Float, default=0.0)
    total_fiber: Mapped[float] = mapped_column(Float, default=0.0)

    # Image metadata and model inference trace
    image_path: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    raw_ai_analysis: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="meals")
    items: Mapped[List["MealItem"]] = relationship("MealItem", back_populates="meal", cascade="all, delete-orphan")


class MealItem(Base):
    __tablename__ = "meal_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    meal_id: Mapped[int] = mapped_column(Integer, ForeignKey("meals.id"), nullable=False, index=True)
    food_name: Mapped[str] = mapped_column(String(150), nullable=False)
    portion_grams: Mapped[float] = mapped_column(Float, default=100.0)

    # Macros
    calories: Mapped[float] = mapped_column(Float, default=0.0)
    protein: Mapped[float] = mapped_column(Float, default=0.0)
    carbs: Mapped[float] = mapped_column(Float, default=0.0)
    fat: Mapped[float] = mapped_column(Float, default=0.0)
    fiber: Mapped[float] = mapped_column(Float, default=0.0)

    # Key Micronutrients (Indian Diet Focus)
    iron_mg: Mapped[float] = mapped_column(Float, default=0.0)
    calcium_mg: Mapped[float] = mapped_column(Float, default=0.0)
    vitamin_c_mg: Mapped[float] = mapped_column(Float, default=0.0)
    vitamin_a_ug: Mapped[float] = mapped_column(Float, default=0.0)
    sodium_mg: Mapped[float] = mapped_column(Float, default=0.0)

    confidence_score: Mapped[float] = mapped_column(Float, default=1.0)
    is_user_adjusted: Mapped[bool] = mapped_column(Boolean, default=False)

    # Relationships
    meal: Mapped["Meal"] = relationship("Meal", back_populates="items")


class DishNutritionProfile(Base):
    """
    Standard Indian recipes grounded in ICMR-NIN Indian Food Composition Tables (IFCT 2017).
    Nutrient densities stored per 100g of cooked dish.
    """
    __tablename__ = "dish_nutrition_profiles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dish_key: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(150), nullable=False)
    category: Mapped[str] = mapped_column(String(50))  # Dal, Curry, Bread, Rice, Snack, Sweet

    # Typical Serving Size
    default_serving_grams: Mapped[float] = mapped_column(Float, default=150.0)

    # Nutritional values per 100g
    calories_per_100g: Mapped[float] = mapped_column(Float, nullable=False)
    protein_per_100g: Mapped[float] = mapped_column(Float, nullable=False)
    carbs_per_100g: Mapped[float] = mapped_column(Float, nullable=False)
    fat_per_100g: Mapped[float] = mapped_column(Float, nullable=False)
    fiber_per_100g: Mapped[float] = mapped_column(Float, default=0.0)

    # Micronutrients per 100g
    iron_mg_per_100g: Mapped[float] = mapped_column(Float, default=0.0)
    calcium_mg_per_100g: Mapped[float] = mapped_column(Float, default=0.0)
    vitamin_c_mg_per_100g: Mapped[float] = mapped_column(Float, default=0.0)
    vitamin_a_ug_per_100g: Mapped[float] = mapped_column(Float, default=0.0)
    sodium_mg_per_100g: Mapped[float] = mapped_column(Float, default=0.0)

    # Comma-separated list of colloquial aliases for semantic search
    aliases: Mapped[str] = mapped_column(Text, default="")


class Feedback(Base):
    """Online evaluation feedback loop (Thumbs Up / Down) for model and estimate accuracy."""
    __tablename__ = "user_feedback"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)
    meal_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("meals.id"), nullable=True)
    is_accurate: Mapped[bool] = mapped_column(Boolean, nullable=False)  # True = Thumbs Up, False = Thumbs Down
    corrected_food_name: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)
    user_comment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="feedback_entries")
