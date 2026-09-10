"""
Streamlit chat UI for the Egyptian Constitution RAG assistant.
"""
import streamlit as st

from api_client import ApiError, ask_question, check_health

st.set_page_config(
    page_title="المساعد الذكي للدستور المصري",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        :root {
            --navy-1: #031d2e;
            --navy-2: #062d45;
            --navy-3: #0b3d58;
            --gold-1: #d9b368;
            --gold-2: #f0d5a1;
            --text-1: #f5ebd1;
            --text-2: #d8c8a1;
            --panel: rgba(13, 40, 58, 0.9);
            --panel-2: rgba(20, 55, 74, 0.95);
            --border: rgba(255, 255, 255, 0.35);
        }

        html, body {
            background: linear-gradient(180deg, var(--navy-1) 0%, var(--navy-2) 100%);
            color: var(--text-1);
        }

        .stApp, [data-testid="stAppViewContainer"] {
            background: linear-gradient(180deg, var(--navy-1) 0%, var(--navy-2) 100%);
            color: var(--text-1);
        }

        .block-container {
            background: transparent;
            color: var(--text-1);
        }

        h1, h2, h3, p, li, span, div {
            color: var(--text-1);
        }

        .stTitle {
            color: var(--gold-2) !important;
            font-weight: 800 !important;
            text-align: center;
        }

        .stCaption {
            color: var(--text-2) !important;
            text-align: center;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(6, 38, 58, 0.96), rgba(7, 31, 47, 0.96));
            border-right: 1px solid var(--border);
        }

        .stSidebar .stButton > button,
        .stSidebar button,
        .stButton > button,
        button[kind="primary"],
        button[kind="secondary"] {
            background: linear-gradient(135deg, rgba(217, 179, 104, 0.18), rgba(217, 179, 104, 0.06)) !important;
            border: 1px solid var(--border) !important;
            color: var(--gold-2) !important;
            border-radius: 14px !important;
            box-shadow: none !important;
            font-weight: 700 !important;
        }

        .stChatMessage {
            border-radius: 16px;
            border: 1px solid var(--border);
            background: rgba(19, 48, 67, 0.75);
        }

        .stChatMessage [data-testid="stChatMessageContent"] {
            background: transparent;
        }

        .stChatInput {
            border: 1.5px solid rgba(255, 255, 255, 0.55) !important;
            border-radius: 24px !important;
            background: #ffffff !important;
            box-shadow: 0 6px 24px rgba(0, 0, 0, 0.35) !important;
            overflow: hidden;
        }

        .stChatInput > div {
            background: #ffffff !important;
            border: none !important;
        }

        .stChatInput textarea,
        .stChatInput input {
            background: #ffffff !important;
            color: #0c2d3d !important;
            border: none !important;
            box-shadow: none !important;
            text-align: right !important;
            direction: rtl !important;
            unicode-bidi: plaintext !important;
            font-size: 1rem !important;
            padding-top: 0.85rem !important;
            padding-bottom: 0.85rem !important;
        }

        .stChatInput textarea::placeholder,
        .stChatInput input::placeholder {
            color: #9a8a68 !important;
            opacity: 1 !important;
            text-align: center !important;
        }

        .stChatInput button {
            background: linear-gradient(135deg, var(--gold-1), var(--gold-2)) !important;
            border: none !important;
            border-radius: 50% !important;
            color: #0c2d3d !important;
            font-weight: 900 !important;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.25) !important;
        }

        .stAlert,
        .stWarning,
        .stError {
            border: 1px solid rgba(217, 179, 104, 0.7);
            border-radius: 12px;
            background: rgba(19, 48, 67, 0.75);
        }

        .stExpander {
            border: 1px solid var(--border) !important;
            border-radius: 14px;
            background: rgba(13, 40, 58, 0.85) !important;
            color: var(--gold-2) !important;
            overflow: hidden;
        }

        [data-testid="stExpander"] {
            background: rgba(13, 40, 58, 0.85) !important;
            border: 1px solid var(--border) !important;
            border-radius: 14px;
            overflow: hidden;
        }

        [data-testid="stExpander"] > div,
        .stExpander > div {
            color: var(--gold-2) !important;
            background: rgba(13, 40, 58, 0.85) !important;
        }

        /* The clickable header row of the expander (summary/details) keeps a
           white background by default in Streamlit — force it to match. */
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] [data-testid="stExpanderToggleIcon"],
        [data-testid="stExpander"] div[role="button"] {
            background: rgba(13, 40, 58, 0.85) !important;
            color: var(--gold-2) !important;
        }
        [data-testid="stExpander"] summary:hover,
        [data-testid="stExpander"] div[role="button"]:hover {
            background: rgba(20, 55, 74, 0.95) !important;
        }
        [data-testid="stExpander"] svg {
            fill: var(--gold-2) !important;
            color: var(--gold-2) !important;
        }
        [data-testid="stExpander"] p,
        [data-testid="stExpander"] span {
            color: var(--text-1) !important;
        }

        .sidebar-empty-fill {
            text-align: center;
            margin-top: 60px;
            opacity: 0.9;
        }
        .sidebar-empty-fill .icon {
            font-size: 3.4rem;
            line-height: 1;
        }
        .sidebar-empty-fill .caption {
            color: var(--text-2);
            font-size: 0.85rem;
            margin-top: 10px;
        }
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {
            background: var(--navy-1) !important;
            background-image: none !important;
        }

        /* ── Fix: remove the white bar at the bottom (chat input wrapper) ──
           The white strip comes from several nested wrapper divs around the
           chat input that Streamlit renders with its own white background.
           We force every one of them back to navy, and only let the actual
           input pill itself (targeted separately below) be white. ── */
        [data-testid="stBottom"],
        [data-testid="stBottomBlockContainer"],
        [data-testid="stChatInputContainer"],
        div:has(> div > [data-testid="stChatInputContainer"]),
        div:has(> [data-testid="stChatInputContainer"]),
        div:has(> [data-testid="stBottomBlockContainer"]) {
            background: var(--navy-1) !important;
            background-image: none !important;
            border: none !important;
            box-shadow: none !important;
        }

        [data-testid="stBottom"] {
            border-top: 1px solid var(--border) !important;
            padding-top: 14px !important;
            padding-bottom: 14px !important;
        }

        [data-testid="stBottomBlockContainer"] {
            max-width: 760px !important;
            margin: 0 auto !important;
        }

        /* ── Force the sidebar to always stay visible, no matter what state
           Streamlit thinks it's in (collapsed/expanded). This overrides the
           width/transform/position Streamlit applies when the sidebar is
           "closed" — cover every attribute value and both div/section forms. ── */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"][aria-expanded="false"],
        [data-testid="stSidebar"][aria-expanded="true"] {
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
            transform: none !important;
            position: relative !important;
            left: 0 !important;
            right: auto !important;
            margin-left: 0 !important;
            min-width: 244px !important;
            max-width: 244px !important;
            width: 244px !important;
            flex-shrink: 0 !important;
            overflow: visible !important;
            pointer-events: auto !important;
        }
        [data-testid="stSidebarContent"],
        [data-testid="stSidebarUserContent"] {
            display: block !important;
            visibility: visible !important;
            width: 100% !important;
            overflow: visible !important;
        }

        /* No collapse is possible anymore, so the reopen arrow has nothing
           to do — hide it and the internal close button both. */
        [data-testid="collapsedControl"],
        [data-testid="stSidebarHeader"] button,
        [data-testid="stSidebarCollapseButton"] {
            display: none !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⚖️ المساعد الذكي لدستور جمهورية مصر العربية")
st.caption("اسأل أي سؤال عن مواد وبنود الدستور المصري للحصول على إجابة موثقة ودقيقة (يدعم العربية والإنجليزية).")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ خيارات المحادثة")
    if st.button("🗑️ مسح المحادثة (Clear Chat)", use_container_width=True):
        st.session_state.messages = []
        st.session_state.is_processing = False
        st.rerun()

    st.markdown(
        """
        <div class="sidebar-empty-fill">
            <div class="icon">⚖️</div>
            <div class="caption">دستور جمهورية مصر العربية</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Sample questions ──────────────────────────────────────────────────────────
with st.expander("💡 أسئلة مقترحة للتجربة / Sample Questions"):
    st.markdown("""
    **باللغة العربية:**
    - ما هي شروط الترشح لمنصب رئيس الجمهورية؟
    - ماذا تنص المادة 139؟
    - ما هي مدة دورة رئاسة الجمهورية؟
    - كيف يتم تعديل مواد الدستور المصري؟
    - ما هي الحقوق والحريات التي يكفلها الدستور للمواطنين؟

    **In English:**
    - What are the eligibility requirements for presidential candidates?
    - What does Article 139 say?
    - How are constitutional amendments approved in Egypt?
    - What are the citizen rights and liberties guaranteed by the constitution?
    """)

# ── Backend health check ──────────────────────────────────────────────────────
if "backend_connected" not in st.session_state:
    st.session_state.backend_connected = check_health()

if not st.session_state.backend_connected:
    st.warning("⚠️ السيرفر غير متصل. تأكد من تشغيل FastAPI Backend على المنفذ 8000.")

# ── Session state initialisation ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# ── Display chat history ──────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            st.caption("📚 **المصادر المرجعية / Sources:** " + ", ".join(message["sources"]))

# ── Input ─────────────────────────────────────────────────────────────────────
input_placeholder = (
    "جاري معالجة الإجابة، يرجى الانتظار..."
    if st.session_state.is_processing
    else "اطرح سؤالك هنا (عربي / English)..."
)
question = st.chat_input(input_placeholder, disabled=st.session_state.is_processing)

if question and not st.session_state.is_processing:
    st.session_state.is_processing = True
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("جاري معالجة الإجابة... / Generating answer..."):
            try:
                # Build history from the last 6 exchanges (12 messages) for follow-up support.
                # We exclude the message we just appended (the current question) from history.
                history_messages = st.session_state.messages[:-1]  # everything before current question
                history_for_api = [
                    {"role": m["role"], "content": m["content"]}
                    for m in history_messages[-12:]  # last 6 user+assistant pairs
                    if m.get("content")
                ]

                result  = ask_question(question, history=history_for_api or None)
                answer  = result.get("answer", "")
                sources = result.get("sources", [])

                st.markdown(answer)
                if sources:
                    st.caption("📚 **المصادر المرجعية / Sources:** " + ", ".join(sources))

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except ApiError as exc:
                error_message = f"حدث خطأ أثناء التواصل مع السيرفر.\n\n`{exc}`"
                st.error(error_message)
                st.session_state.messages.append({"role": "assistant", "content": error_message})
            finally:
                st.session_state.is_processing = False
                st.rerun()