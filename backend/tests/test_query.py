"""
Pytest suite testing health, valid queries, greetings, and out-of-scope handling.
"""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_query_happy_path():
    response = client.post("/query", json={"question": "What is this document about?"})
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "sources" in body
    assert isinstance(body["sources"], list)


def test_query_greeting():
    response = client.post("/query", json={"question": "السلام عليكم"})
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "وعليكم السلام" in body["answer"] or "أهلاً" in body["answer"]
    assert body["sources"] == []


def test_query_out_of_scope():
    response = client.post("/query", json={"question": "بتحب الرنجه"})
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert "خارج نطاق" in body["answer"] or "غير مذكورة" in body["answer"]
    # Sources must be empty for out of scope queries
    assert body["sources"] == []


def test_query_invalid_input():
    # empty string violates min_length=1 on QueryRequest.question
    response = client.post("/query", json={"question": ""})
    assert response.status_code == 422
