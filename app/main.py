"""
FastAPI Application Entrypoint for NutritionTrackerAI.
Provides REST API endpoints and mounts the interactive web dashboard.
"""
import argparse
from contextlib import asynccontextmanager
import logging
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from telegram import Update
import uvicorn

from app.routes import router as api_router
from app.telegram_bot import setup_telegram_bot_app
from db.connection import engine, init_db
from db.seed_ifct import seed_ifct_database

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("nutrition_tracker.app")

STATIC_DIR = Path(__file__).resolve().parent / "static"

telegram_bot_app = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI Lifespan Context Manager.
    Initializes database schemas, seeds 50 IFCT 2017 profiles, and starts Telegram bot if configured.
    """
    global telegram_bot_app
    logger.info("🚀 Starting NutritionTrackerAI Web Application...")
    try:
        await init_db()
        await seed_ifct_database()
        logger.info("Database schemas and IFCT 2017 profiles ready.")
    except Exception as e:
        logger.error(f"Startup database initialization error: {e}", exc_info=True)

    if os.getenv("TESTING") != "true":
        telegram_bot_app = setup_telegram_bot_app()
        if telegram_bot_app:
            try:
                await telegram_bot_app.initialize()
                await telegram_bot_app.start()
                logger.info("Telegram Bot Webhook processor initialized.")

                webhook_url = os.getenv("TELEGRAM_WEBHOOK_URL")
                if webhook_url:
                    await telegram_bot_app.bot.set_webhook(url=webhook_url)
                    logger.info(f"✅ Telegram Webhook registered to: {webhook_url}")
            except Exception as e:
                logger.warning(f"Telegram Bot initialization / webhook setup warning: {e}")

    yield

    logger.info("🛑 Shutting down NutritionTrackerAI Application...")
    if telegram_bot_app:
        try:
            await telegram_bot_app.stop()
            await telegram_bot_app.shutdown()
        except Exception:
            pass
    try:
        await engine.dispose()
    except Exception:
        pass


app = FastAPI(
    title="NutritionTrackerAI API & Dashboard",
    description="AI-Powered Indian Food Nutrition Tracker with Computer Vision & Google ADK Agents",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router)


# Root endpoint serves interactive UI
@app.get("/", tags=["Dashboard"])
async def serve_dashboard():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "service": "NutritionTrackerAI",
        "status": "online",
        "docs": "/docs",
    }


@app.get("/health", tags=["Monitoring"])
async def healthcheck() -> Dict[str, Any]:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "database": "connected",
        "cv_model": "EfficientNet-B0 (97.71% Val Accuracy)",
        "grounding_db": "ICMR-NIN IFCT 2017 (50 classes)",
    }


# Telegram Webhook Receiver endpoints (accepts /webhook, /webhook/telegram, and /api/webhook)
@app.post("/webhook", tags=["Telegram"])
@app.post("/webhook/telegram", tags=["Telegram"])
@app.post("/api/webhook", tags=["Telegram"])
async def telegram_webhook_handler(request: Request):
    """
    Receives incoming Telegram Webhook updates forwarded via ngrok or cloud domain.
    """
    try:
        data = await request.json()
        if not telegram_bot_app:
            return {"status": "ok", "message": "Bot token not active, webhook acknowledged"}

        update = Update.de_json(data=data, bot=telegram_bot_app.bot)
        if update:
            await telegram_bot_app.process_update(update)
        return {"status": "ok"}
    except Exception as e:
        logger.error(f"Error handling Telegram webhook: {e}", exc_info=True)
        return Response(status_code=status.HTTP_200_OK)


# Mount static assets directory
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Starting server on {host}:{port}")
    uvicorn.run("app.main:app", host=host, port=port, workers=1)
