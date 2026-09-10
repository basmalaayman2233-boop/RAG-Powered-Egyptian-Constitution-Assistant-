# -*- coding: utf-8 -*-
"""
RAG Indexing Pipeline — Egyptian Constitution (Article-Aware Edition)
=====================================================================
Key improvement: Instead of blind character-splitting, this pipeline:
  1. Detects article boundaries (مادة N / Article N patterns).
  2. Keeps each article as its own chunk (with overflow splitting if too long).
  3. Stores {"source": filename, "article": "139"} in Chroma metadata so the
     retrieval service can do exact article-number filtering.

Run this script whenever the PDF changes or after the first clone:
    py -3.12 run_pipeline.py
"""
import sys
import os
import re

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8")

from pathlib import Path

print("=" * 60)
print("STEP 1: Load & Inspect Documents")
print("=" * 60)

import pypdf

DATA_DIR = Path("data/raw_documents")

pdf_paths = sorted(DATA_DIR.glob("**/*.pdf"))
txt_paths = sorted(DATA_DIR.glob("**/*.txt"))

print(f"Found {len(pdf_paths)} PDF file(s) and {len(txt_paths)} text file(s)")


def strip_tashkeel(text: str) -> str:
    """Remove Arabic diacritics (tashkeel) for cleaner text."""
    return re.sub(r'[\u0617-\u061A\u064B-\u065F]', '', text)


def normalize_hindi_numerals(text: str) -> str:
    """Convert Eastern Arabic (Hindi) numerals to Western Arabic numerals."""
    table = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
    return text.translate(table)


docs = []
failed = []

for path in pdf_paths:
    try:
        reader = pypdf.PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        text = strip_tashkeel(text)
        text = normalize_hindi_numerals(text)
        if len(text.strip()) < 20:
            failed.append((path.name, "little/no extractable text — likely needs OCR"))
            print(f"  [SKIP] {path.name}: no extractable text")
            continue
        docs.append({"source": path.name, "text": text, "pages": len(reader.pages)})
        print(f"  [OK] {path.name}: {len(reader.pages)} pages, {len(text):,} chars extracted")
    except Exception as e:
        failed.append((path.name, str(e)[:120]))
        print(f"  [FAIL] {path.name}: {str(e)[:120]}")
        continue

total_pages = sum(d["pages"] for d in docs)
print(f"\nLoaded {len(docs)} document(s) | {total_pages} total pages")
if failed:
    print(f"Failed: {len(failed)}")
    for name, reason in failed:
        print(f"  [WARN] {name}: {reason}")

print("\n" + "=" * 60)
print("STEP 2: Article-Aware Chunking")
print("=" * 60)

# --- Configuration ---
# Maximum characters for a single article chunk before we sub-split it.
# Chosen to fit comfortably within the embedding model's 512-token window
# (≈ 4 chars/token → 512 * 4 = 2048 chars; we use 1800 to be safe).
MAX_ARTICLE_CHUNK = 1800
CHUNK_OVERLAP     = 200  # overlap when splitting long articles

# Regex that matches an Arabic article header in the PDF, e.g.:
#   "مادة 141"  "مادة (141)"  "مادة( 141)"  "المادة 141"
# The PDF may also contain English-style "Article 141" in bilingual versions.
ARTICLE_PATTERN = re.compile(
    r'(?:(?:ال)?مادة\s*\(?\s*(\d+)\s*\)?|Article\s+(\d+))',
    re.UNICODE
)


def fix_pdf_article_number(num_str: str) -> str:
    """
    PyPDF reverses 3-digit article numbers in RTL Arabic PDFs.
    Examples confirmed from the actual PDF:
        '931' → '139'  (Article 139 = President as head of state)
        '041' → '140'  (Article 140 = 6-year presidential term)
        '241' → '142'  (Article 142 = endorsement requirements)
        '141' → '141'  (palindrome — unaffected)

    Rule: reverse the digit STRING for 3-digit numbers only.
    1- and 2-digit numbers are NOT reversed by PyPDF.
    """
    if len(num_str) == 3:
        reversed_str = num_str[::-1]          # e.g. '931' → '139'
        return str(int(reversed_str))          # strip any leading zero
    return num_str


def split_into_articles(text: str, source: str) -> list[dict]:
    """
    Split document text on article boundaries.
    Returns a list of dicts:
      {"id": ..., "source": ..., "article": "139" or "PREAMBLE", "text": ...}

    If a segment (between two article headers) is longer than MAX_ARTICLE_CHUNK,
    it is further split with overlap so no embedding loses context.
    """
    chunks = []

    # Find all article boundary positions
    matches = list(ARTICLE_PATTERN.finditer(text))

    # Everything before the first article header is the preamble / table of contents
    if matches:
        preamble = text[:matches[0].start()].strip()
        if len(preamble) > 50:  # skip near-empty preambles
            for i, seg in enumerate(_split_long(preamble)):
                chunks.append({
                    "id": f"{source}::PREAMBLE::seg{i}",
                    "source": source,
                    "article": "PREAMBLE",
                    "text": seg,
                })
    else:
        # No article markers found → fall back to character-based splitting
        print(f"  [WARN] No article markers found in {source} — falling back to character chunking")
        for i, seg in enumerate(_split_long(text)):
            chunks.append({
                "id": f"{source}::chunk{i}",
                "source": source,
                "article": "UNKNOWN",
                "text": seg,
            })
        return chunks

    # Process each article segment
    # Track occurrence count per article number to handle cases where an article
    # number appears more than once in the PDF (e.g. due to table of contents or
    # repeated headers across chapters). This guarantees globally unique IDs.
    art_occurrence: dict[str, int] = {}
    for idx, match in enumerate(matches):
        art_num_raw = match.group(1) or match.group(2)  # whichever captured
        # Fix RTL digit reversal for 3-digit numbers (PDF extraction artefact)
        art_num = fix_pdf_article_number(art_num_raw)
        start = match.start()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        segment = text[start:end].strip()

        occurrence = art_occurrence.get(art_num, 0)
        art_occurrence[art_num] = occurrence + 1

        # Prefix with occurrence number when the article appears more than once
        occ_suffix = f"_occ{occurrence}" if occurrence > 0 else ""

        sub_segs = _split_long(segment)
        for i, seg in enumerate(sub_segs):
            chunk_id = f"{source}::article{art_num}{occ_suffix}::seg{i}"
            chunks.append({
                "id": chunk_id,
                "source": source,
                "article": art_num,
                "text": seg,
            })

    return chunks


def _split_long(text: str) -> list[str]:
    """
    If text is shorter than MAX_ARTICLE_CHUNK, return it as-is.
    Otherwise split with overlap to preserve cross-boundary context.
    """
    if len(text) <= MAX_ARTICLE_CHUNK:
        return [text]

    segs = []
    start = 0
    while start < len(text):
        end = start + MAX_ARTICLE_CHUNK
        seg = text[start:end].strip()
        if seg:
            segs.append(seg)
        start += MAX_ARTICLE_CHUNK - CHUNK_OVERLAP
    return segs


all_chunks: list[dict] = []
article_counts: dict[str, int] = {}

for doc in docs:
    art_chunks = split_into_articles(doc["text"], doc["source"])
    all_chunks.extend(art_chunks)
    for c in art_chunks:
        a = c["article"]
        article_counts[a] = article_counts.get(a, 0) + 1

avg_len = sum(len(c["text"]) for c in all_chunks) / max(len(all_chunks), 1)
unique_articles = len([k for k in article_counts if k not in ("PREAMBLE", "UNKNOWN")])
print(f"Produced {len(all_chunks)} chunks covering {unique_articles} unique articles")
print(f"Avg chunk length: {avg_len:.0f} chars")
print(f"Article distribution sample: { {k: v for k, v in list(article_counts.items())[:10]} }")

print("\n" + "=" * 60)
print("STEP 3: Embeddings & Vector Store")
print("=" * 60)

from sentence_transformers import SentenceTransformer
import chromadb

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
print(f"Loading embedding model: {EMBEDDING_MODEL_NAME} ...")
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Model loaded.")

VECTOR_STORE_DIR = "backend/data/vector_store"
COLLECTION_NAME  = "documents"

Path(VECTOR_STORE_DIR).mkdir(parents=True, exist_ok=True)
client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)

# Always start fresh so stale article-unaware chunks are removed
try:
    client.delete_collection(COLLECTION_NAME)
    print(f"  [OK] Deleted old collection '{COLLECTION_NAME}'")
except Exception:
    pass

collection = client.create_collection(COLLECTION_NAME)

batch_size = 32
total = len(all_chunks)
for i in range(0, total, batch_size):
    batch = all_chunks[i:i + batch_size]
    embeddings = embedder.encode([c["text"] for c in batch]).tolist()
    collection.add(
        ids=[c["id"] for c in batch],
        embeddings=embeddings,
        documents=[c["text"] for c in batch],
        metadatas=[
            {"source": c["source"], "article": c["article"]}
            for c in batch
        ],
    )
    pct = min(i + batch_size, total)
    print(f"  Embedded {pct}/{total} chunks ...", end="\r")

print(f"\n[OK] Persisted {collection.count()} chunks → {VECTOR_STORE_DIR}")
print("      Each chunk has metadata: {{source, article}}")

print("\n" + "=" * 60)
print("STEP 4: Smoke-test Retrieval (no LLM)")
print("=" * 60)

TEST_CASES = [
    # (question, expected_article_in_top_result)
    ("ما هي شروط الترشح لرئاسة الجمهورية؟",     "141"),
    ("ماذا تنص المادة 139؟",                      "139"),
    ("ما مدة دورة رئاسة الجمهورية؟",              "140"),
    ("كيف تعدل مواد الدستور؟",                    None),
    ("What are the eligibility requirements?",    "141"),
]

print(f"\n{'Q':<55} {'Top Article':<15} {'Match?'}")
print("-" * 80)
for q, expected_art in TEST_CASES:
    qe = embedder.encode([q]).tolist()
    results = collection.query(
        query_embeddings=qe,
        n_results=3,
        include=["metadatas", "documents", "distances"],
    )
    top_meta = results["metadatas"][0][0] if results["metadatas"][0] else {}
    top_art  = top_meta.get("article", "?")
    dist     = results["distances"][0][0] if results["distances"][0] else 9.9
    match    = "✓" if (expected_art is None or top_art == expected_art) else "✗"
    print(f"  {q[:53]:<55} art={top_art:<10} d={dist:.3f}  {match}")

# Article-specific sanity check
for art_num in ["139", "140", "141", "142"]:
    res = collection.get(where={"article": art_num}, include=["documents"])
    count = len(res["ids"])
    snippet = (res["documents"][0][:80].replace("\n", " ")) if res["documents"] else "(none)"
    print(f"\n  Article {art_num}: {count} chunk(s) | snippet: {snippet!r}")

print("\n" + "=" * 60)
print("[DONE] Article-aware pipeline complete! Vector store ready.")
print(f"   -> {VECTOR_STORE_DIR}")
print("=" * 60)
