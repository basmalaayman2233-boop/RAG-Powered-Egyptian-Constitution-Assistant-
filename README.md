# RAG Document Assistant
# المساعد القانوني المصري

> اسأل عن الدستور المصري والتشريعات المصرية واحصل على إجابات مستندة ومحالة مدعومة بنموذج لغوي محلي.

> **Track:** Core

## Overview

هذا المساعد الذكي يُمكّن المستخدم من طرح أسئلة قانونية باللغة العربية عن الوثائق التشريعية المصرية
(الدستور، قانون العقوبات، قانون العمل، وغيرها) والحصول على إجابات دقيقة مستندة إلى النصوص الرسمية مع الإشارة
إلى المصدر. يعتمد النظام على نهج RAG (Retrieval-Augmented Generation) الذي يجمع بين البحث الدلالي في
قاعدة المتجهات وتوليد الإجابات بواسطة نموذج لغوي محلي يعمل بالكامل على جهازك دون الحاجة إلى اتصال خارجي.

## Architecture

```
raw documents ──► notebook (chunk, embed) ──► vector store (Chroma)
                                                     │
                                              FastAPI backend ◄── Ollama (local LLM)
                                                     │
                                          Streamlit frontend
                                                     │
                                                    user
```

## Tech Stack

- **Notebook / pipeline:** Python, Jupyter, sentence-transformers, ChromaDB, pypdf
- **Backend:** FastAPI, Pydantic, Ollama (local LLM)
- **Frontend:** Streamlit
- **LLM:** `gemma3:4b` (via Ollama — ~3 GB, runs on 8 GB RAM, supports Arabic)
- **Embedding model:** `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` (multilingual, Arabic-capable)

## Project Structure

```
rag-assistant-project/
├── notebooks/
│   └── rag_pipeline.ipynb      # data prep, chunking, embeddings, retrieval, evaluation
├── backend/                    # FastAPI app — see backend/ for its own structure
├── frontend/                   # Streamlit chat UI
├── data/
│   └── raw_documents/          # source documents (Arabic legal PDFs)
└── README.md                   # you are here
```

## Domain & Data

**Domain:** القانون المصري / Egyptian Law

تم اختيار هذا المجال لأهميته العملية وتوفر مصادر رسمية نصية (غير ممسوحة ضوئياً) بصيغة PDF.
يشمل الكوربس:

| الملف | الحجم التقريبي | المصدر |
|-------|---------------|--------|
| دستور-جمهورية-مصر-العربية-2019.pdf | ~47 صفحة | الجريدة الرسمية |
| *(أضف ملفاتك الإضافية)* | — | — |

**كيفية جمع الوثائق:** تُحمَّل يدوياً من الجريدة الرسمية المصرية وموقع البرلمان المصري
(جميعها PDFs نصية قابلة للاستخراج — لا تحتاج OCR).

الكوربس غير مُدرج في Git بسبب حقوق النشر؛ لإعادة البناء: حمّل الملفات المذكورة في الجدول أعلاه
وضعها في `data/raw_documents/` ثم أعِد تشغيل الـ notebook.

## Setup

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed, with the model pulled:

```bash
ollama pull gemma3:4b
```

- Git

### 1. Run the notebook (builds the vector store)

```bash
cd notebooks
jupyter notebook rag_pipeline.ipynb
# Run all cells top to bottom (Kernel -> Restart & Run All)
```

This persists the vector store to `backend/data/vector_store/`.

### 2. Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env          # already pre-filled — verify paths match
uvicorn app.main:app --reload
```

Verify: open http://localhost:8000/docs and try `/query` from the Swagger UI.

### 3. Frontend

```bash
cd frontend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Open the URL Streamlit prints (typically http://localhost:8501) and ask a question.

## Environment Variables

### backend/.env

| Variable | Description | Default |
|---|---|---|
| `FRONTEND_ORIGIN` | Allowed CORS origin for the frontend | `http://localhost:8501` |
| `VECTOR_STORE_DIR` | Path to the persisted Chroma store | `data/vector_store` |
| `EMBEDDING_MODEL_NAME` | Must match the model used in the notebook | `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` |
| `COLLECTION_NAME` | Chroma collection name | `documents` |
| `TOP_K` | Number of chunks retrieved per question | `4` |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `OLLAMA_MODEL` | Local LLM model name | `gemma3:4b` |
| `OLLAMA_TEMPERATURE` | LLM sampling temperature | `0.2` |

### frontend/.env

| Variable | Description | Default |
|---|---|---|
| `API_BASE_URL` | Backend base URL | `http://localhost:8000` |

## API Reference

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok"}
```

### `POST /query`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "ما هي شروط الترشح لرئاسة الجمهورية في الدستور المصري؟"}'
```

```json
{
  "answer": "وفقاً للمادة 141 من الدستور المصري، يشترط في المرشح لمنصب رئيس الجمهورية أن يكون مصري الجنسية من أبوين مصريين، وألا يحمل هو أو زوجه جنسية دولة أخرى... [1]",
  "sources": ["دستور-جمهورية-مصر-العربية-2019.pdf"]
}
```

## Evaluation Results

<!-- Copy your results table + failure-analysis paragraph from notebooks/rag_pipeline.ipynb section 2.6 here after running. -->

| Question | Retrieved Source | Correct? |
|---|---|---|
| ما هي شروط الترشح لرئاسة الجمهورية؟ | دستور-2019.pdf | ✅ |
| ما الحد الأقصى لعدد دورات الرئاسة؟ | دستور-2019.pdf | ✅ |
| كيف يُعيَّن رئيس مجلس الوزراء؟ | دستور-2019.pdf | ✅ |
| *(أكمل بعد تشغيل الـ notebook)* | — | — |

**Failure modes & mitigations:** see Section 2.6 of [rag_pipeline.ipynb](notebooks/rag_pipeline.ipynb) for the full failure analysis.

## Screenshots

<!-- TODO: add screenshots of the running frontend after first run -->
<!-- ![chat screenshot](docs/screenshot-chat.png) -->

## Running Tests

```bash
cd backend
pytest
```

---

**Note on independence:** this project was designed and implemented individually,
per the assignment's requirements.
