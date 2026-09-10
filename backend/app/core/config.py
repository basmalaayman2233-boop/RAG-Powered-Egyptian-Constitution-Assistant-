"""
Application settings, loaded from environment variables / .env file.
Nothing here is domain-specific — you shouldn't need to touch this file
unless you add a new configurable value.
"""
import os
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(_ENV_PATH, ".env", "backend/.env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # --- App ---
    app_name: str = "RAG Document Assistant"
    api_v1_prefix: str = "/api"

    # --- CORS ---
    frontend_origin: str = "http://localhost:8501"

    # --- Vector store ---
    vector_store_dir: str = "data/vector_store"
    embedding_model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    collection_name: str = "documents"

    # --- Retrieval ---
    top_k: int = 6

    # --- LLM (Ollama) ---
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gemma3:4b"
    ollama_temperature: float = 0.2


@lru_cache
def get_settings() -> Settings:
    return Settings()

