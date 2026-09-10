"""
Thin wrapper around the backend API.
Keeps app.py free of raw requests calls and makes the backend URL
configuration a single point of change.
"""
import os

import requests
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


class ApiError(Exception):
    pass


def check_health() -> bool:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except requests.RequestException:
        return False


def ask_question(question: str, history: list[dict] | None = None) -> dict:
    """
    Send a question to the RAG backend.

    Args:
        question: The current user question.
        history:  Optional list of previous turns for follow-up context.
                  Each entry: {"role": "user"|"assistant", "content": "..."}
    """
    try:
        payload: dict = {"question": question}
        if history:
            payload["history"] = history
        response = requests.post(
            f"{API_BASE_URL}/query",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise ApiError(f"Could not reach the backend at {API_BASE_URL}: {exc}") from exc
