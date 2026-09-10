"""
Retrieval service — Article-Aware Hybrid Search
================================================
Improvements over the previous version:
  1. Article-number detection: if the query mentions "المادة N" or "Article N",
     chunks whose metadata["article"] == N are boosted to the very top.
  2. Metadata-based article filtering as a hard constraint for article-specific queries.
  3. top_k increased to 6 for better multi-chunk article coverage.
  4. Cleaner RRF fusion that gives article metadata hits priority over pure semantics.
"""
from dataclasses import dataclass
import re

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings

settings = get_settings()


@dataclass
class RetrievedChunk:
    document: str
    chunk_id: str
    text: str
    distance: float
    article: str = ""   # NEW: article number from metadata ("139", "141", …, or "PREAMBLE")


# ---------------------------------------------------------------------------
# English → Arabic concept expansion (unchanged from previous version)
# ---------------------------------------------------------------------------
ENGLISH_TO_ARABIC_CONCEPTS = {
    "president": "رئيس الجمهورية",
    "presidential": "رئاسة الجمهورية",
    "candidate": "مترشح ترشح يترشح",
    "candidacy": "الترشح لرئاسة الجمهورية",
    "eligibility": "شروط الترشح يشترط",
    "requirements": "شروط الترشح يشترط",
    "amendment": "تعديل الدستور",
    "amendments": "تعديل مواد الدستور",
    "term": "مدة الرئاسة دورة",
    "terms": "دورات رئاسة ست سنوات",
    "rights": "الحقوق والحريات",
    "liberties": "الحريات العامة",
    "parliament": "مجلس النواب",
    "senate": "مجلس الشيوخ",
    "court": "المحكمة الدستورية العليا",
    "judiciary": "السلطة القضائية",
    "government": "الحكومة مجلس الوزراء",
    "prime minister": "رئيس مجلس الوزراء",
    "constitution": "الدستور المصري",
    "vote": "انتخاب استفتاء",
    "referendum": "استفتاء عام",
}


def normalize_text(text: str) -> str:
    """Normalizes Arabic text for robust matching."""
    t = re.sub(r'[\u0617-\u061A\u064B-\u065F\u0640]', '', text)
    t = re.sub(r'[إأآا]', 'ا', t)
    t = re.sub(r'[ىي]', 'ي', t)
    t = re.sub(r'[ة]', 'ه', t)
    hindi_to_arabic = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')
    return t.translate(hindi_to_arabic).lower()


def get_stems(word: str) -> set[str]:
    w = word
    stems = {w}
    for p in ['كال', 'فال', 'بال', 'وال', 'ولل', 'لل', 'ال']:
        if w.startswith(p) and len(w) > len(p) + 2:
            w = w[len(p):]
            stems.add(w)
            break
    for p in ['و', 'ف', 'ب', 'ك', 'ل', 'ي', 'ت', 'ن', 'م', 'س', 'است']:
        if w.startswith(p) and len(w) > len(p) + 2:
            w = w[len(p):]
            stems.add(w)
            break
    for s in ['كما', 'هما', 'هم', 'هن', 'كم', 'نا', 'ها', 'ية', 'يه', 'ات', 'ان', 'ين', 'ون', 'وا', 'ه', 'ي', 'ا']:
        if w.endswith(s) and len(w) > len(s) + 2:
            w = w[:-len(s)]
            stems.add(w)
            break
    if len(w) >= 3:
        stems.add(w)
    return stems


# ---------------------------------------------------------------------------
# Article-number extraction from a query
# ---------------------------------------------------------------------------
_ARTICLE_IN_QUERY = re.compile(
    r'(?:(?:ال)?ماد[ةه]\s*\(?\s*(\d+)\s*\)?|article\s+(\d+))',
    re.IGNORECASE | re.UNICODE,
)


def extract_article_number(question: str) -> str | None:
    """
    Returns the article number string if the question explicitly asks about a
    specific article, e.g. "ماذا تنص المادة 139؟" → "139".
    Returns None if no specific article is mentioned.
    """
    norm = normalize_text(question)
    m = _ARTICLE_IN_QUERY.search(norm)
    if m:
        return m.group(1) or m.group(2)
    return None


# ---------------------------------------------------------------------------
# Main Retrieval Service
# ---------------------------------------------------------------------------
class RetrievalService:
    """
    Hybrid retrieval: Semantic vector search + keyword/stem scoring + RRF fusion.
    For article-specific queries, the correct article chunks are guaranteed to
    be included via metadata filtering, regardless of semantic similarity.
    """

    def __init__(self) -> None:
        import os
        from pathlib import Path
        
        # Robust path resolution for vector_store_dir
        store_path = Path(settings.vector_store_dir)
        if not store_path.exists() or not any(store_path.iterdir()):
            # Try backend/data/vector_store or relative to this file
            alt1 = Path("backend") / settings.vector_store_dir
            alt2 = Path(__file__).resolve().parent.parent.parent / "data" / "vector_store"
            if alt1.exists() and any(alt1.iterdir()):
                store_path = alt1
            elif alt2.exists() and any(alt2.iterdir()):
                store_path = alt2

        self._client = chromadb.PersistentClient(path=str(store_path))
        self._collection = self._client.get_or_create_collection(settings.collection_name)
        self._embedder = SentenceTransformer(settings.embedding_model_name)

        # Cache all chunks for keyword/stem scoring
        all_data = self._collection.get(include=["documents", "metadatas"])
        self._all_ids   = all_data.get("ids", [])
        self._all_docs  = all_data.get("documents", [])
        self._all_metas = all_data.get("metadatas", [])
        self._norm_docs = [normalize_text(d) for d in self._all_docs]

        # Build article → [chunk_ids] index for fast metadata lookup
        self._article_index: dict[str, list[str]] = {}
        for cid, meta in zip(self._all_ids, self._all_metas):
            art = (meta or {}).get("article", "")
            if art:
                self._article_index.setdefault(art, []).append(cid)

    # ------------------------------------------------------------------
    def retrieve(self, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or settings.top_k
        if not self._all_ids:
            return []

        # ── Step 0: Detect if this is an article-specific question ──────────
        target_article = extract_article_number(question)

        # ── Step 1: Semantic vector search ──────────────────────────────────
        query_embedding = self._embedder.encode([question]).tolist()
        n_fetch = max(1, min(len(self._all_ids), max(20, top_k * 4)))
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=n_fetch,
            include=["distances", "metadatas"],
        )
        vector_ids   = results.get("ids",       [[]])[0]
        vector_dists = results.get("distances", [[]])[0]
        v_rank = {cid: rank for rank, cid in enumerate(vector_ids)}
        id_to_dist = dict(zip(vector_ids, vector_dists))

        # ── Step 2: Article-targeted mandatory retrieval ─────────────────────
        # If the user asks about a specific article, force all chunks of that
        # article to be candidates with the highest possible priority.
        forced_ids: list[str] = []
        if target_article and target_article in self._article_index:
            forced_ids = self._article_index[target_article]

        # ── Step 3: English concept expansion + keyword/stem scoring ─────────
        expanded_query = question
        lower_q = question.lower()
        for en_word, ar_trans in ENGLISH_TO_ARABIC_CONCEPTS.items():
            if en_word in lower_q:
                expanded_query += " " + ar_trans

        norm_q = normalize_text(expanded_query)
        q_words = [w for w in re.findall(r'\w+', norm_q) if len(w) > 2]
        q_stems_list = [get_stems(w) for w in q_words]

        scores: dict[str, float] = {}
        for cid, norm_doc in zip(self._all_ids, self._norm_docs):
            score = 0.0
            doc_words = set(re.findall(r'\w+', norm_doc))

            # Stem matching
            matched_terms = 0
            for stem_set in q_stems_list:
                if any(s in doc_words or any(s in dw for dw in doc_words) for s in stem_set):
                    matched_terms += 1
            score += matched_terms * 2.0

            # Legal phrase boosting
            if ("شروط" in norm_q or "يشترط" in norm_q or "eligibility" in lower_q or "requirement" in lower_q):
                if ("يترشح" in norm_doc or "ترشح" in norm_doc or "الترشح" in norm_doc) and ("رئيس" in norm_doc or "رئاس" in norm_doc):
                    score += 15.0
                if "يشترط فيمن يترشح" in norm_doc:
                    score += 25.0

            if ("تعديل" in norm_q or "amend" in lower_q) and ("تعديل" in norm_doc or "تعديل مواد" in norm_doc):
                score += 15.0

            if ("مده" in norm_q or "سنوات" in norm_q or "term" in lower_q) and ("ست سنوات" in norm_doc or "مده الرئاسه" in norm_doc):
                score += 15.0

            if ("حقوق" in norm_q or "حريات" in norm_q or "rights" in lower_q) and ("الحقوق" in norm_doc or "الحريات" in norm_doc):
                score += 15.0

            if score > 0:
                scores[cid] = score

        sorted_cids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        s_rank = {cid: rank for rank, cid in enumerate(sorted_cids)}

        # ── Step 4: RRF fusion over vector + keyword candidates ───────────────
        all_candidate_ids = set(vector_ids).union(sorted_cids[:20]).union(forced_ids)
        rrf_scores: dict[str, float] = {}

        for cid in all_candidate_ids:
            vr = v_rank.get(cid, 999)
            sr = s_rank.get(cid, 999)
            rrf = (1.0 / (20 + vr)) + (3.0 / (15 + sr))

            # Hard boost for forced (article-targeted) chunks
            if cid in forced_ids:
                rrf += 100.0

            rrf_scores[cid] = rrf

        best_ids = sorted(all_candidate_ids, key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        # ── Step 5: Assemble result chunks ────────────────────────────────────
        id_to_doc  = dict(zip(self._all_ids, self._all_docs))
        id_to_meta = dict(zip(self._all_ids, self._all_metas))

        chunks: list[RetrievedChunk] = []
        for cid in best_ids:
            meta = id_to_meta.get(cid) or {}
            chunks.append(
                RetrievedChunk(
                    document=meta.get("source", "unknown"),
                    chunk_id=cid,
                    text=id_to_doc.get(cid, ""),
                    distance=id_to_dist.get(cid, 0.0),
                    article=meta.get("article", ""),
                )
            )

        return chunks


# ---------------------------------------------------------------------------
# Singleton accessor (loaded once at startup)
# ---------------------------------------------------------------------------
_retrieval_service: RetrievalService | None = None


def get_retrieval_service() -> RetrievalService:
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service
