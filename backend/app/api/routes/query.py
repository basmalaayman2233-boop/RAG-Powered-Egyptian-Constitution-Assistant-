"""
GET /health and POST /query.
Wiring only — the actual RAG logic lives in app/services/.
"""
from fastapi import APIRouter, Depends

from app.schemas.query import QueryRequest, QueryResponse
from app.services.generation import detect_greeting_or_out_of_scope, generate_answer
from app.services.retrieval import RetrievalService, get_retrieval_service

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> QueryResponse:
    # 1. Instant response for conversational greetings or obvious out-of-scope queries
    quick_reply = detect_greeting_or_out_of_scope(request.question)
    if quick_reply:
        return QueryResponse(answer=quick_reply, sources=[])

    # 2. Retrieve relevant chunks
    chunks = retrieval_service.retrieve(request.question)

    # 3. Generate answer — pass conversation history for follow-up support
    answer = generate_answer(
        question=request.question,
        chunks=chunks,
        history=request.history,
    )

    # 4. Suppress sources if the model concluded the query is out of scope or not found
    out_of_scope_indicators = [
        "خارج نطاق",
        "غير مذكورة في الوثائق",
        "غير مذكور في المستندات",
        "لم يتم العثور",
        "outside the scope",
        "not mentioned in the provided",
        "not found in the provided",
        "could not be found in the available",
    ]
    if any(indicator in answer.lower() for indicator in out_of_scope_indicators):
        sources = []
    else:
        sources = list(dict.fromkeys(chunk.document for chunk in chunks))

    return QueryResponse(answer=answer, sources=sources)
