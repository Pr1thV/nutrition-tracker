"""
Unit and Integration Tests for FastAPI REST Application Endpoints.
"""
import os
os.environ["TESTING"] = "true"

import pytest
from fastapi.testclient import TestClient
from app.main import app


def test_root_endpoint():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200


def test_healthcheck_endpoint():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["database"] == "connected"


def test_dishes_list_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/dishes")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 20


def test_coach_chat_endpoint():
    with TestClient(app) as client:
        response = client.post(
            "/api/chat",
            json={"user_id": 9999, "message": "Give me a high protein vegetarian Indian diet tip."},
        )
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert len(data["response"]) > 10


def test_feedback_endpoint():
    with TestClient(app) as client:
        response = client.post(
            "/api/feedback",
            json={"user_id": 9999, "is_accurate": True, "user_comment": "Great estimation!"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "Feedback recorded successfully"


def test_daily_summary_endpoint():
    with TestClient(app) as client:
        response = client.get("/api/daily-summary/9999")
        assert response.status_code == 200
        data = response.json()
        assert "consumed" in data
        assert "targets" in data
