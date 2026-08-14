"""
Telegram Bot Update Handlers and Message Formatters for NutritionTrackerAI.
Seamlessly handles /start, /daily, /help, food photo uploads, and portion adjustments.
"""
import io
import logging
import os
from typing import Any, Dict, List, Optional
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from agent.root_dispatcher import agent_system
from agent.tools.daily_summary_tool import get_daily_nutrition_summary
from db.connection import AsyncSessionLocal
from db.models import Feedback, Meal

logger = logging.getLogger(__name__)


def build_macro_badge(label: str, value: float, unit: str = "g") -> str:
    """Formats macronutrient values cleanly."""
    return f"*{label}:* `{value:.1f}{unit}`"


def build_progress_bar(current: float, target: float, length: int = 10) -> str:
    """Generates a visual progress bar (e.g. 🟩🟩🟩🟩⬜⬜⬜⬜⬜⬜)."""
    if target <= 0:
        return "⬜" * length
    ratio = min(max(current / target, 0.0), 1.0)
    filled = int(round(ratio * length))
    empty = length - filled
    return "🟩" * filled + "⬜" * empty


def format_meal_breakdown_message_html(
    meal_data: Dict[str, Any],
    daily_consumed: Optional[Dict[str, float]] = None,
    daily_target: float = 2000.0,
) -> str:
    """Formats the food vision estimation card using clean HTML."""
    items: List[Dict[str, Any]] = meal_data.get("items", [])
    total_cal = meal_data.get("total_calories", 0.0)
    total_pro = meal_data.get("total_protein", 0.0)
    total_carb = meal_data.get("total_carbs", 0.0)
    total_fat = meal_data.get("total_fat", 0.0)
    total_fib = meal_data.get("total_fiber", 0.0)

    lines = [
        "🥗 <b>FOOD NUTRITION ESTIMATE</b>",
        "━━━━━━━━━━━━━━━━━━━━━",
    ]

    # Plate Item Decomposition
    for itm in items:
        name = itm.get("food_name", "Dish")
        grams = itm.get("portion_grams", 100.0)
        cal = itm.get("calories", 0.0)
        pro = itm.get("protein", 0.0)
        conf = itm.get("confidence", 1.0)
        conf_badge = f" <code>({int(conf*100)}%)</code>" if conf and conf < 1.0 else ""
        lines.append(f"• <b>{name}</b> (~<code>{int(grams)}g</code>){conf_badge}")
        lines.append(f"   ↳ <code>{int(cal)} kcal</code> | <code>{pro:.1f}g Protein</code>")

    lines.append("━━━━━━━━━━━━━━━━━━━━━")
    lines.append("📊 <b>TOTAL MEAL NUTRITION:</b>")
    lines.append(f"🔥 <b>Calories:</b> <code>{int(total_cal)} kcal</code>")
    lines.append(
        f"🥩 <b>Protein:</b> <code>{total_pro:.1f}g</code> | 🍞 <b>Carbs:</b> <code>{total_carb:.1f}g</code> | 🥑 <b>Fat:</b> <code>{total_fat:.1f}g</code> | 🌾 <b>Fiber:</b> <code>{total_fib:.1f}g</code>"
    )

    # Key Micronutrients
    if items:
        first = items[0]
        fe = first.get("iron_mg", 0.0)
        ca = first.get("calcium_mg", 0.0)
        vit_c = first.get("vitamin_c_mg", 0.0)
        if any([fe, ca, vit_c]):
            lines.append("━━━━━━━━━━━━━━━━━━━━━")
            lines.append("🔬 <b>Key Micronutrients (IFCT 2017):</b>")
            lines.append(f"• Iron: <code>{fe:.1f}mg</code> | Calcium: <code>{ca:.0f}mg</code> | Vit C: <code>{vit_c:.1f}mg</code>")

    # Daily Goal Progress Bar
    if daily_consumed is not None:
        today_cal = daily_consumed.get("calories", total_cal)
        pct = (today_cal / daily_target) * 100 if daily_target > 0 else 0
        pbar = build_progress_bar(today_cal, daily_target)
        lines.append("━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"📅 <b>Daily Budget:</b> <code>{int(today_cal)}</code> / <code>{int(daily_target)} kcal</code> ({int(pct)}%)")
        lines.append(f"<code>{pbar}</code>")

    lines.append("\n<i>Adjust serving size or confirm below:</i>")
    return "\n".join(lines)


def build_portion_adjustment_keyboard(meal_id: int) -> InlineKeyboardMarkup:
    """Builds interactive inline buttons for portion scaling and feedback."""
    keyboard = [
        [
            InlineKeyboardButton("🔹 Small (-25%)", callback_data=f"portion:{meal_id}:0.75"),
            InlineKeyboardButton("🔸 Large (+50%)", callback_data=f"portion:{meal_id}:1.50"),
            InlineKeyboardButton("2x Double", callback_data=f"portion:{meal_id}:2.00"),
        ],
        [
            InlineKeyboardButton("👍 Accurate", callback_data=f"fb:{meal_id}:up"),
            InlineKeyboardButton("👎 Incorrect", callback_data=f"fb:{meal_id}:down"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


# Command Handlers
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /start command."""
    user = update.effective_user
    text = (
        f"Namaste <b>{user.first_name}</b>! 🙏 Welcome to <b>NutritionTrackerAI</b>.\n\n"
        "📸 <b>How to use:</b>\n"
        "1. <b>Send a photo</b> of any Indian or international meal (Thali, Biryani, Roti, Dosa, Dal, Eggs, Salad, Oats).\n"
        "2. Our <b>EfficientNet-B0 + Gemini Vision</b> decomposes each dish on your plate.\n"
        "3. Every item is biochemically grounded with <b>ICMR-NIN IFCT 2017</b> nutritional data.\n"
        "4. Ask health and diet questions to your <b>AI Wellness Coach</b> anytime!\n\n"
        "⚡ <b>Commands:</b>\n"
        "• <code>/daily</code> — View today's calories, macros, & micronutrients\n"
        "• <code>/help</code> — Tips for best photo recognition"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def daily_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /daily command."""
    user = update.effective_user
    async with AsyncSessionLocal() as session:
        summary = await get_daily_nutrition_summary(telegram_id=user.id, session=session)

    if not summary.get("has_meals_today"):
        await update.message.reply_text(
            "📅 <b>Daily Summary</b>\n\nNo meals logged yet today! Snap and send a photo of your meal to begin.",
            parse_mode=ParseMode.HTML,
        )
        return

    c = summary["consumed"]
    t = summary["targets"]
    pct = summary.get("calorie_progress_pct", 0)
    pbar = build_progress_bar(c["calories"], t["calories"])

    text = (
        "📅 <b>TODAY'S NUTRITION SUMMARY</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        f"🔥 <b>Calories:</b> <code>{int(c['calories'])}</code> / <code>{int(t['calories'])} kcal</code> ({int(pct)}%)\n"
        f"<code>{pbar}</code>\n\n"
        "📊 <b>Macronutrient Breakdown:</b>\n"
        f"• 🥩 <b>Protein:</b> <code>{c['protein']:.1f}g</code> / <code>{t['protein']:.0f}g</code>\n"
        f"• 🍞 <b>Carbs:</b> <code>{c['carbs']:.1f}g</code> / <code>{t['carbs']:.0f}g</code>\n"
        f"• 🥑 <b>Fat:</b> <code>{c['fat']:.1f}g</code> / <code>{t['fat']:.0f}g</code>\n"
        f"• 🌾 <b>Fiber:</b> <code>{c['fiber']:.1f}g</code>\n\n"
        "🔬 <b>Micronutrients (ICMR-NIN Grounded):</b>\n"
        f"• Iron: <code>{c['iron_mg']:.1f}mg</code> | Calcium: <code>{c['calcium_mg']:.0f}mg</code>\n"
        f"• Vitamin C: <code>{c['vitamin_c_mg']:.1f}mg</code> | Sodium: <code>{c['sodium_mg']:.0f}mg</code>\n"
        "━━━━━━━━━━━━━━━━━━━━━\n"
        f"🍽️ <b>Meals logged today:</b> <code>{summary['meals_logged_count']}</code>"
    )
    await update.message.reply_text(text, parse_mode=ParseMode.HTML)


async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles incoming food photos."""
    user = update.effective_user
    status_msg = None
    try:
        photo_file = await update.message.photo[-1].get_file()

        status_msg = await update.message.reply_text(
            "🔍 <b>Analyzing your meal with AI...</b>\nDecomposing plate & calculating IFCT nutrients...",
            parse_mode=ParseMode.HTML,
        )

        image_byte_array = await photo_file.download_as_bytearray()
        image_bytes = bytes(image_byte_array)

        async with AsyncSessionLocal() as session:
            result = await agent_system.handle_photo_upload(
                image_bytes=image_bytes,
                telegram_id=user.id,
                session=session,
            )
            daily_summary = await get_daily_nutrition_summary(telegram_id=user.id, session=session)

        meal_id = result.get("meal_id", 0)
        reply_text = format_meal_breakdown_message_html(
            meal_data=result,
            daily_consumed=daily_summary.get("consumed"),
            daily_target=daily_summary.get("targets", {}).get("calories", 2000.0),
        )
        reply_markup = build_portion_adjustment_keyboard(meal_id=meal_id)

        # Edit status message in place
        if status_msg:
            try:
                await status_msg.edit_text(
                    reply_text,
                    reply_markup=reply_markup,
                    parse_mode=ParseMode.HTML,
                )
                return
            except Exception:
                await status_msg.delete()

        await update.message.reply_text(
            reply_text,
            reply_markup=reply_markup,
            parse_mode=ParseMode.HTML,
        )
    except Exception as e:
        logger.error(f"Error processing Telegram food photo: {e}", exc_info=True)
        if status_msg:
            try:
                await status_msg.edit_text(
                    f"⚠️ <b>Analysis Error:</b> Could not process photo: {str(e)}",
                    parse_mode=ParseMode.HTML,
                )
            except Exception:
                pass


async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles conversational nutrition advice messages."""
    user = update.effective_user
    text = update.message.text

    try:
        async with AsyncSessionLocal() as session:
            reply = await agent_system.handle_text_message(
                text=text,
                telegram_id=user.id,
                session=session,
            )
        await update.message.reply_text(reply)
    except Exception as e:
        logger.error(f"Error handling Telegram text message: {e}", exc_info=True)
        await update.message.reply_text("Namaste! Please ask your nutrition question or send a food photo.")


async def callback_query_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles portion scaling and feedback button clicks."""
    query = update.callback_query
    await query.answer()
    data = query.data

    if data.startswith("portion:"):
        parts = data.split(":")
        meal_id = int(parts[1])
        scale_factor = float(parts[2])

        async with AsyncSessionLocal() as session:
            from sqlalchemy import select
            q = select(Meal).where(Meal.id == meal_id)
            res = await session.execute(q)
            meal = res.scalar_one_or_none()
            if meal:
                meal.total_calories = round(meal.total_calories * scale_factor, 1)
                meal.total_protein = round(meal.total_protein * scale_factor, 1)
                meal.total_carbs = round(meal.total_carbs * scale_factor, 1)
                meal.total_fat = round(meal.total_fat * scale_factor, 1)
                await session.commit()
                await query.edit_message_text(
                    f"✅ <b>Portion adjusted ({scale_factor}x)!</b>\n\n"
                    f"🔥 <b>New Calories:</b> <code>{int(meal.total_calories)} kcal</code>\n"
                    f"🥩 <b>Protein:</b> <code>{meal.total_protein:.1f}g</code> | 🍞 <b>Carbs:</b> <code>{meal.total_carbs:.1f}g</code> | 🥑 <b>Fat:</b> <code>{meal.total_fat:.1f}g</code>",
                    parse_mode=ParseMode.HTML,
                )

    elif data.startswith("fb:"):
        parts = data.split(":")
        meal_id = int(parts[1])
        is_up = parts[2] == "up"

        async with AsyncSessionLocal() as session:
            fb = Feedback(
                user_id=query.from_user.id,
                meal_id=meal_id,
                is_accurate=is_up,
                user_comment="Telegram button feedback",
            )
            session.add(fb)
            await session.commit()

        msg = "🙏 Thank you! Your feedback helps fine-tune our Indian food recognition models." if is_up else "📝 Noted! We have logged this dispute to improve future model accuracy."
        await query.edit_message_text(msg)


def setup_telegram_bot_app() -> Optional[Application]:
    """Initializes python-telegram-bot Application instance if token is configured."""
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token or token == "your_telegram_bot_token_here":
        logger.info("TELEGRAM_BOT_TOKEN not configured. Running in Web-only mode.")
        return None

    try:
        app = Application.builder().token(token).build()
        app.add_handler(CommandHandler("start", start_command))
        app.add_handler(CommandHandler("daily", daily_command))
        app.add_handler(CommandHandler("help", start_command))
        app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))
        app.add_handler(CallbackQueryHandler(callback_query_handler))
        return app
    except Exception as e:
        logger.error(f"Failed to initialize Telegram application: {e}")
        return None
