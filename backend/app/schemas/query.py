"""
Request/response contracts for the /query endpoint.
"""
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's natural-language question")
    history: list[dict] | None = Field(
        default=None,
        description=(
            "Optional conversation history for follow-up question support. "
            "Each entry must be {'role': 'user'|'assistant', 'content': '...'}"
        ),
    )


class SourceChunk(BaseModel):
    document: str
    chunk_id: str
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
