"""
Online Evaluation & Model Drift Monitoring.
Analyzes user feedback (Thumbs Up / Down) collected in real-time from Telegram interactions.
"""
import asyncio
import json
import logging
from typing import Any, Dict
from sqlalchemy import func, select
from db.connection import AsyncSessionLocal
from db.models import Feedback

logger = logging.getLogger(__name__)


async def compute_online_feedback_metrics() -> Dict[str, Any]:
    """
    Computes real-time online satisfaction and accuracy rates from live user feedback.
    """
    async with AsyncSessionLocal() as session:
        # Total feedback count
        total_q = select(func.count(Feedback.id))
        total_res = await session.execute(total_q)
        total_count = total_res.scalar() or 0

        if total_count == 0:
            return {
                "total_feedback_events": 0,
                "accuracy_rate_pct": 100.0,
                "drift_detected": False,
                "status": "No online feedback recorded yet.",
            }

        # Positive feedback count
        positive_q = select(func.count(Feedback.id)).where(Feedback.is_accurate == True)
        pos_res = await session.execute(positive_q)
        positive_count = pos_res.scalar() or 0

        accuracy_rate = (positive_count / total_count) * 100.0
        drift_detected = accuracy_rate < 75.0  # Alert if inaccuracy > 25%

        metrics = {
            "total_feedback_events": total_count,
            "positive_feedback_count": positive_count,
            "negative_feedback_count": total_count - positive_count,
            "accuracy_rate_pct": round(accuracy_rate, 2),
            "drift_detected": drift_detected,
            "drift_alert": "⚠️ Model Drift Detected: High user dispute rate." if drift_detected else "✅ Performance Stable",
        }

        print("\n" + "=" * 50)
        print("        ONLINE EVALUATION & DRIFT REPORT")
        print("=" * 50)
        for k, v in metrics.items():
            print(f"  {k:.<30} {v}")
        print("=" * 50 + "\n")

        return metrics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(compute_online_feedback_metrics())
