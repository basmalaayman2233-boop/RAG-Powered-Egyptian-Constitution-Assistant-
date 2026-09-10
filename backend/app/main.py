"""
FastAPI application entrypoint.

Uses a lifespan context manager so the vector store + embedding model are
loaded exactly ONCE at startup (per Phase 3's requirement), not on every
request.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.query import router as query_router
from app.core.config import get_settings
from app.services.retrieval import get_retrieval_service
from app.utils.logging_config import configure_logging

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    # Warm up the vector store + embedding model once, at startup.
    get_retrieval_service()
    yield
    # (no teardown needed for Chroma's PersistentClient)


app = FastAPI(title=settings.app_name, lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_router, tags=["rag"])
