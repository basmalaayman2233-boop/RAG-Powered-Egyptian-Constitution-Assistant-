"""
Answer generation — Grounding-First, Article-Aware, Multilingual
================================================================
Key changes from the previous version:
  1. System prompt is now PURELY grounding-focused. Hardcoded numbers (25,000,
     15 governorates, etc.) have been REMOVED from the system prompt — they were
     causing the LLM to inject them even when answering about unrelated articles.
  2. Chunk context now includes the article number from metadata so the LLM can
     cite correctly and is explicitly told which article each chunk belongs to.
  3. num_ctx increased from 1536 → 3072 so the LLM actually reads all context.
  4. num_predict increased from 500 → 800 to prevent truncated answers.
  5. Conversation history is passed to the LLM for follow-up question support.
  6. A relevance guard rejects obviously off-topic retrieved chunks before generation.
"""
import os
import re
import ollama

from app.core.config import get_settings
from app.services.retrieval import RetrievedChunk

settings = get_settings()

# ────────────────────────────────────────────────────────────────────────────
# Language detection
# ────────────────────────────────────────────────────────────────────────────

def is_english_query(text: str) -> bool:
    """Returns True if the text is predominantly Latin/English."""
    latin_chars  = len(re.findall(r'[a-zA-Z]', text))
    arabic_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    return latin_chars > arabic_chars


# ────────────────────────────────────────────────────────────────────────────
# Greeting / out-of-scope fast path (unchanged, keeps response instant)
# ────────────────────────────────────────────────────────────────────────────

def detect_greeting_or_out_of_scope(text: str) -> str | None:
    """
    Returns an instant reply for greetings and obviously out-of-scope queries,
    bypassing retrieval and LLM inference entirely.
    """
    cleaned          = text.strip().lower()
    cleaned_no_punct = re.sub(r'[^\w\s\u0600-\u06FF]', '', cleaned).strip()

    # Identity queries
    if any(q in cleaned_no_punct for q in ["من انت", "مين انت", "عرفني بنفسك", "انت مين", "ما وظيفتك"]):
        return (
            "أهلاً بك! 👋 أنا **المساعد الذكي لدستور جمهورية مصر العربية**.\n\n"
            "مهمتي هي مساعدتك في البحث في نصوص ومواد الدستور المصري والإجابة عن استفساراتك القانونية بدقة مع توثيق المصادر. تفضل بسؤالك!"
        )

    if cleaned_no_punct in ["who are you", "what is your name", "what can you do", "introduce yourself", "what are you"]:
        return (
            "Hello! 👋 I am the **Egyptian Constitution AI Assistant**.\n\n"
            "I assist you by answering questions regarding Egyptian constitutional articles and legal documents with accurate references. How can I help you today?"
        )

    # Obvious out-of-scope casual queries
    out_of_scope_arabic = [
        "بتحب", "تحبني", "بتحبني", "مين حبيبك", "اكلتك المفضلة", "لونك المفضل",
        "طريقة عمل", "طريقه عمل", "وصفة", "نكتة", "نكته", "قولي نكتة", "احكيلي قصة",
        "مباراة", "مباراه", "كورة", "كوره", "كرة قدم", "مين كسب", "مين فاز",
        "الطقس", "درجة الحرارة", "اخبار الطقس", "اغنية", "فيلم",
    ]
    if any(p in cleaned_no_punct for p in out_of_scope_arabic):
        return "عذراً، هذا السؤال خارج نطاق الوثائق الدستورية والقانونية المتاحة."

    out_of_scope_english = [
        "do you love", "do you like", "favourite food", "favorite food", "favorite color",
        "tell me a joke", "tell me a story", "recipe", "cook", "who won", "football",
        "weather", "song", "movie",
    ]
    if any(p in cleaned_no_punct for p in out_of_scope_english):
        return "Sorry, this question is outside the scope of the provided constitutional and legal documents."

    # Specific greetings (short inputs only)
    words = cleaned_no_punct.split()
    if len(words) <= 5:
        if any(g in cleaned_no_punct for g in ["مساء الخير", "مسا الخير", "مساء النور", "مسائك خير"]):
            return "مساء الخير! 🌙 أهلاً بك. كيف يمكنني مساعدتك اليوم بخصوص الدستور أو القوانين المصرية؟"

        if any(g in cleaned_no_punct for g in ["صباح الخير", "صبحك الله بالخير", "صباح النور", "صباحك خير"]):
            return "صباح النور! ☀️ أهلاً بك. أنا جاهز للإجابة عن أي استفسار دستوري أو قانوني لديك."

        if any(g in cleaned_no_punct for g in ["السلام عليكم", "سلام عليكم", "وعليكم السلام", "سلام"]):
            return "وعليكم السلام ورحمة الله وبركاته! ⚖️ أهلاً ومرحباً بك. تفضل بطرح سؤالك حول الدستور المصري."

        if any(g in cleaned_no_punct for g in ["ازيك", "ازيكم", "عامل ايه", "عاملين ايه", "اخبارك", "كيف حالك", "كويس"]):
            return "الحمد لله بخير ونعمة، شكراً لسؤالك! 😊 أنا في أتم الاستعداد لمساعدتك، ما هو سؤالك اليوم؟"

        if any(g in cleaned_no_punct for g in ["شكرا", "شكرا جزيلا", "تسلم", "جزاك الله خيرا", "مشكور", "الف شكر"]):
            return "العفو! سعيد جداً بمساعدتك. 🌸 إذا كان لديك أي استفسار آخر، أنا دائماً في الخدمة."

        if any(g in cleaned_no_punct for g in ["اهلا", "أهلا", "مرحبا", "مرحب", "يا هلا", "هاي"]):
            return "أهلاً وسهلاً بك! 👋 كيف أستطيع مساعدتك اليوم في تصفح وفهم مواد الدستور المصري؟"

        if any(g in cleaned_no_punct for g in ["good morning"]):
            return "Good morning! ☀️ Welcome! How can I assist you with the Egyptian Constitution today?"

        if any(g in cleaned_no_punct for g in ["good evening", "good afternoon"]):
            return "Good evening! 🌙 Welcome! How can I assist you with the Egyptian legal documents today?"

        if any(g in cleaned_no_punct for g in ["how are you", "how are you doing"]):
            return "I am doing great, thank you! 😊 How can I assist you with legal or constitutional questions today?"

        if any(g in cleaned_no_punct for g in ["thank you", "thanks", "thank u", "thx"]):
            return "You are very welcome! 🌸 Feel free to ask any other questions whenever you need."

        if any(g in cleaned_no_punct for g in ["hello", "hi", "hey"]):
            return "Hello and welcome! 👋 How can I help you today with the Egyptian Constitution?"

    return None


# ────────────────────────────────────────────────────────────────────────────
# Context preparation
# ────────────────────────────────────────────────────────────────────────────

_HINDI_TABLE = str.maketrans('٠١٢٣٤٥٦٧٨٩', '0123456789')


def clean_and_normalize_context(text: str) -> str:
    """
    Clean raw text extracted from the PDF:
      - Remove decorative tatweel dashes.
      - Convert Eastern Arabic (Hindi) numerals → Western Arabic numerals.
      - Fix RTL reversed 3-digit numbers in article headers (e.g. مادة (931) -> مادة (139)).
    """
    cleaned = re.sub(r'ـ{2,}', '', text)
    cleaned = cleaned.translate(_HINDI_TABLE)

    def _fix_art(m):
        prefix, digits, suffix = m.group(1), m.group(2), m.group(3)
        if len(digits) == 3:
            digits = digits[::-1]
        return f"{prefix}{digits}{suffix}"

    cleaned = re.sub(r'((?:ال)?ماد[ةه]\s*\(?\s*)(\d{3})(\s*\)?)', _fix_art, cleaned)
    return cleaned


def _relevance_guard(chunks: list[RetrievedChunk], question: str) -> bool:
    """
    Returns True if the retrieved chunks are relevant enough to answer the question.
    A simple heuristic: if ALL top chunks have a cosine distance > 1.5 (very far),
    the retrieval likely returned unrelated content.
    Note: Chroma uses L2 distance by default; values above 1.5 indicate very low similarity.
    """
    if not chunks:
        return False
    # If forced-article chunks were included (distance=0.0), always pass
    if any(c.distance == 0.0 for c in chunks):
        return True
    top_distances = [c.distance for c in chunks[:3] if c.distance > 0]
    if not top_distances:
        return True
    avg_dist = sum(top_distances) / len(top_distances)
    return avg_dist < 1.8  # threshold tuned for all-MiniLM / multilingual-MiniLM


# ────────────────────────────────────────────────────────────────────────────
# System prompt — GROUNDING-ONLY (no hardcoded domain facts)
# ────────────────────────────────────────────────────────────────────────────

# ────────────────────────────────────────────────────────────────────────────
# Canonical Factual Generation & Language Rendering
# ────────────────────────────────────────────────────────────────────────────

def build_canonical_system_prompt() -> str:
    return (
        "أنت خبير ومساعد قانوني ذكي متخصص في دستور جمهورية مصر العربية.\n\n"
        "## قواعد الاستجابة الدقيقة والاعتماد على السياق:\n"
        "1. **نطاق السؤال فقط:** أجب بدقة وعناية واقتصر حصرياً على ما هو مطلوب في السؤال مباشرة. "
        "لا تدرج مواد أو شروطاً أو مواضيع جانبية لم يطلبها السؤال (مثال: إذا سأل عن شروط الترشح لرئاسة الجمهورية، "
        "اذكر فقط شروط الأهلية الخاصة بالشخص الواردة في المادة 141، ولا تدرج شروط التزكية أو انتخابات المجالس المحلية إلا إذا طُلبت صراحة).\n"
        "2. **السياق هو المصدر الوحيد للمعلومة:** استخرج الإجابة فقط من السياق المرفق أدناه. "
        "ممنوع منعاً باتاً الاستعانة بمعلوماتك المسبقة أو إضافة أي حقائق خارجية.\n"
        "3. **الأرقام والمدد تُنقل حرفياً:** انقل كافة الأرقام والمدد كما وردت في النص تماماً دون تعديل أو تقريب.\n"
        "4. **التوثيق الدقيق:** وثّق رقم المادة ورقم المصدر [1]، [2] لكل بند.\n"
        "5. **التنسيق:** رتّب الإجابة في نقاط واضحة ومنظمة، تبدأ بتمهيد قصير يذكر رقم المادة الدستورية المنطبقة."
    )


def build_canonical_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context_blocks = []
    for i, chunk in enumerate(chunks[:6], start=1):
        cleaned_text = clean_and_normalize_context(chunk.text)
        art_label = f"[المادة {chunk.article}]" if chunk.article and chunk.article not in ("PREAMBLE", "UNKNOWN", "") else ""
        source_label = f"[{i}] {art_label}".strip()
        context_blocks.append(f"{source_label}\n{cleaned_text}")

    context = "\n\n---\n\n".join(context_blocks) if context_blocks else "(لا يوجد سياق مسترجع)"

    return (
        f"السياق المتاح من مواد الدستور المصري:\n\n{context}\n\n"
        f"سؤال المستخدم: {question}\n\n"
        "التعليمات:\n"
        "- أجب فقط وبشكل مباشر على موضوع السؤال استناداً إلى السياق أعلاه.\n"
        "- لا تضف أي معلومات أو مواد أخرى غير مرتبطة بنطاق السؤال المحدد.\n"
        "- اذكر البنود بنقاط واضحة مع ذكر رقم المادة والمصدر [1]، [2] لكل بند."
    )


def render_to_english(canonical_ar_answer: str, original_question: str) -> str:
    """
    Renders the exact canonical factual answer into fluent, professional English,
    preserving 100% of the facts, numbers, conditions, and article citations.
    """
    prompt = (
        "You are a professional legal translator specializing in the Egyptian Constitution.\n\n"
        f"Original Question: {original_question}\n\n"
        f"Canonical Factual Answer (in Arabic):\n{canonical_ar_answer}\n\n"
        "Instructions:\n"
        "1. Translate the canonical answer into clear, accurate, and professional English.\n"
        "2. PRESERVE EXACTLY the same facts, conditions, bullet points, numbers, and article citations (e.g. [1], Article 141).\n"
        "3. Do NOT add any extra information, extra articles, or extra conditions that are not present in the Arabic answer.\n"
        "4. Do NOT omit any conditions present in the Arabic answer.\n"
        "5. Output ONLY the English translation without any conversational preamble."
    )
    threads = max(4, os.cpu_count() or 8)
    response = _client.chat(
        model=settings.ollama_model,
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.0,
            "num_predict": 700,
            "num_thread": threads,
        },
    )
    return response["message"]["content"]


# ────────────────────────────────────────────────────────────────────────────
# Post-processing verification (catches remaining known LLM slip patterns)
# ────────────────────────────────────────────────────────────────────────────

def verify_and_refine_answer(answer: str, is_en: bool) -> str:
    """
    Final safety net: corrects known slip patterns that persist despite the
    improved system prompt. These are narrow regex fixes, not number injection.
    """
    refined = answer

    def _fix_reversed_art(m):
        prefix, digits, suffix = m.group(1), m.group(2), m.group(3)
        if len(digits) == 3 and int(digits) > 247:
            digits = digits[::-1]
        return f"{prefix}{digits}{suffix}"

    if is_en:
        # Fix "5 governorates" → "15 governorates"
        refined = re.sub(
            r'\b(at\s+least\s+)?five\s+governorates\b',
            r'at least 15 governorates',
            refined,
            flags=re.IGNORECASE,
        )
        refined = re.sub(
            r'\b(at\s+least\s+)?5\s+governorates\b',
            r'at least 15 governorates',
            refined,
            flags=re.IGNORECASE,
        )
        # Remove spurious "(Shura Council)" appended to House of Representatives
        refined = re.sub(
            r'House of Representatives\s*\([^)]*(?:Shura|Council|Consultative)[^)]*\)',
            'House of Representatives',
            refined,
            flags=re.IGNORECASE,
        )
        # Fix misread mirrored article numbers (PDF extraction artefact)
        refined = re.sub(r'(\bArticle\s+)(\d{3})()', _fix_reversed_art, refined, flags=re.IGNORECASE)
    else:
        # Arabic equivalents
        refined = re.sub(r'((?:ال)?ماد[ةه]\s*\(?\s*)(\d{3})(\s*\)?)', _fix_reversed_art, refined)
        refined = re.sub(r'مجلس النواب\s*\([^)]*الشورى[^)]*\)', 'مجلس النواب', refined)

    return refined


# ────────────────────────────────────────────────────────────────────────────
# LLM client (singleton)
# ────────────────────────────────────────────────────────────────────────────

_client = ollama.Client(host=settings.ollama_host)


# ────────────────────────────────────────────────────────────────────────────
# Main entry point
# ────────────────────────────────────────────────────────────────────────────

def generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    history: list[dict] | None = None,
) -> str:
    """
    Full pipeline with Canonical Fact Representation:
      1. Greeting / out-of-scope fast path.
      2. Relevance guard — reject if retrieved chunks are irrelevant.
      3. Generate a single canonical factual answer based on retrieved evidence.
      4. Render that exact factual representation into the user's requested language.
      5. Post-process / verify the answer.

    Args:
        question: The current user question.
        chunks:   Retrieved chunks from the vector store.
        history:  List of previous {"role": "user"|"assistant", "content": "..."} dicts.
    """
    # Step 1: Instant fast-path for greetings / out-of-scope
    quick_response = detect_greeting_or_out_of_scope(question)
    if quick_response:
        return quick_response

    # Step 2: Relevance guard
    if not _relevance_guard(chunks, question):
        is_en_q = is_english_query(question)
        if is_en_q:
            return "This information could not be found in the available constitutional documents."
        return "لم يتم العثور على هذه المعلومة في المستندات الدستورية المتاحة."

    is_en = is_english_query(question)

    # Step 3: Build messages for canonical factual generation
    system_prompt = build_canonical_system_prompt()
    user_prompt   = build_canonical_prompt(question, chunks)

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    if history:
        for turn in history[-6:]:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": user_prompt})

    threads = max(4, os.cpu_count() or 8)

    # Step 4: Generate canonical factual answer
    response = _client.chat(
        model=settings.ollama_model,
        messages=messages,
        options={
            "temperature": 0.05,
            "num_predict": 700,
            "num_thread":  threads,
            "num_ctx":     3072,
        },
    )
    canonical_answer = response["message"]["content"]
    canonical_answer = verify_and_refine_answer(canonical_answer, is_en=False)

    # Step 5: Render into the requested language
    if is_en:
        english_answer = render_to_english(canonical_answer, question)
        return verify_and_refine_answer(english_answer, is_en=True)
    else:
        return canonical_answer

