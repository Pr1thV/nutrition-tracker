"""
Wellness & Nutrition Coach Agent.
Provides personalized Indian diet advice, macro balance coaching, and nutritional Q&A.
"""
import json
import logging
import os
from typing import Any, Dict, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from agent.tools.daily_summary_tool import get_daily_nutrition_summary

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

COACH_SYSTEM_PROMPT = """
You are an empathetic, science-backed Indian Clinical Dietitian and Wellness Coach.
Your goal is to guide users to achieve their health, muscle-building, and fat-loss goals
while enjoying authentic Indian home-cooked meals (Thalis, Dals, Sabzis, Rotis, Rice, Paneer, Sattu, etc.).

Guidelines:
1. Ground advice in Indian dietary patterns (vegetarian sources like Paneer, Soya Chunks, Moong Sprouts, Chana, Dals, Greek Yogurt/Curd, Sattu).
2. Give actionable, portion-controlled suggestions.
3. Keep responses concise, warm, formatted with clean bullet points and emojis.
4. When reviewing daily summaries, highlight protein sufficiency and micronutrient diversity (Iron, Calcium, Fiber).
"""


class WellnessCoachAgent:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
        self._client = None
        self._init_client()

    def _init_client(self):
        self.api_key = self.api_key or os.getenv("GEMINI_API_KEY")
        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning(f"Coach agent running in rule-based fallback mode: {e}")
                self._client = None

    async def answer_nutrition_query(
        self,
        user_message: str,
        telegram_id: int,
        session: AsyncSession,
    ) -> str:
        """Answers general nutrition questions or provides personalized diet advice."""
        # 1. Fetch user's current day nutrition context
        summary = await get_daily_nutrition_summary(telegram_id, session)

        context_str = ""
        if summary.get("has_meals_today"):
            c = summary["consumed"]
            t = summary["targets"]
            context_str = (
                f"\nUser's Today Progress: Consumed {c['calories']} / {t['calories']} kcal, "
                f"Protein: {c['protein_g']} / {t['protein_g']}g, Carbs: {c['carbs_g']} / {t['carbs_g']}g, Fat: {c['fat_g']} / {t['fat_g']}g."
            )

        # 2. Invoke Gemini if available
        if self._client:
            try:
                prompt = (
                    f"{COACH_SYSTEM_PROMPT}\n\n"
                    f"User Context:{context_str}\n\n"
                    f"User Query: {user_message}\n\n"
                    "Response:"
                )
                response = self._client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                )
                if response.text:
                    coach_reply = response.text.strip()
                    from observability.setup import get_langfuse
                    lf = get_langfuse()
                    if lf:
                        try:
                            trace = lf.trace(
                                name="coach_chat_query",
                                user_id=str(telegram_id),
                                input=user_message,
                                output=coach_reply,
                            )
                            trace.generation(
                                name="gemini_coach_response",
                                model=self.model_name,
                                input=prompt,
                                output=coach_reply,
                            )
                        except Exception:
                            pass
                    return coach_reply
            except Exception as e:
                logger.error(f"Error calling Coach LLM: {e}")

        # 3. Rule-based expert fallback if offline / key missing
        msg_lower = user_message.lower()
        if "protein" in msg_lower:
            return (
                "💪 **Top High-Protein Vegetarian Indian Foods:**\n"
                "• **Soya Chunks:** ~52g protein per 100g (dry)\n"
                "• **Paneer / Cottage Cheese:** ~18g protein per 100g\n"
                "• **Sattu (Roasted Gram Flour):** ~20g protein per 100g\n"
                "• **Moong Dal / Sprouts:** ~24g protein per 100g (raw)\n"
                "• **Low-fat Dahi / Greek Yogurt:** ~10g protein per 100g\n\n"
                "💡 *Tip: Combine Dal + Rice or Roti to form a complete amino acid profile!*"
            )
        elif "calorie" in msg_lower or "weight loss" in msg_lower or "fat loss" in msg_lower:
            return (
                "🥗 **Smart Indian Calorie Management Tips:**\n"
                "• Measure cooking oil/ghee (1 tbsp = ~120 kcal)\n"
                "• Fill half your plate with salad (cucumber, tomato) and sabzi before rice/roti\n"
                "• Swap deep-fried snacks for roasted makhana, chana, or sprouted chaat\n"
                "• Prioritize protein and fiber in each meal for satiety!"
            )
        else:
            return (
                "🌿 **Nutrition Coach:** I'm here to help you balance your macros and hit your goals!\n"
                "You can log meals anytime by sending a photo 📸 or ask questions about Indian nutrition, recipes, and protein sources."
            )
