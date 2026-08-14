"""
Google ADK / GenAI Agent Root Dispatcher.
Routes incoming Telegram requests to specialized agents (Tracker vs Coach).
"""
import logging
from typing import Any, Dict, Optional, Union
from sqlalchemy.ext.asyncio import AsyncSession
from agent.coach_agent import WellnessCoachAgent
from agent.tracker_agent import FoodTrackerAgent

logger = logging.getLogger(__name__)


class NutritionAgentSystem:
    """
    Unified entry point for the Multi-Agent nutrition tracking system.
    """

    def __init__(self):
        self.tracker_agent = FoodTrackerAgent()
        self.coach_agent = WellnessCoachAgent()

    async def handle_photo_upload(
        self,
        image_bytes: bytes,
        telegram_id: int,
        session: AsyncSession,
    ) -> Dict[str, Any]:
        """Dispatches food photos to the FoodTrackerAgent."""
        logger.info(f"Dispatching photo analysis for Telegram User: {telegram_id}")
        return await self.tracker_agent.analyze_and_log_food_photo(
            image_bytes=image_bytes,
            telegram_id=telegram_id,
            session=session,
        )

    async def handle_text_message(
        self,
        text: str,
        telegram_id: int,
        session: AsyncSession,
    ) -> str:
        """Dispatches text messages and queries to the WellnessCoachAgent."""
        logger.info(f"Dispatching conversational text query from User {telegram_id}: '{text}'")
        return await self.coach_agent.answer_nutrition_query(
            user_message=text,
            telegram_id=telegram_id,
            session=session,
        )


# Global agent system singleton
agent_system = NutritionAgentSystem()
