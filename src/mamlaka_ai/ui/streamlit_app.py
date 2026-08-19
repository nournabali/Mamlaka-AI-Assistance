"""Streamlit interface for the bilingual Mamlaka AI assistant."""

from __future__ import annotations

import base64
import html
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable

import streamlit as st

from mamlaka_ai.config import settings
from mamlaka_ai.generation.answerer import Answerer, KIND_ERROR
from mamlaka_ai.prompts import refusals
from mamlaka_ai.retrieval.retriever import load_retriever
from mamlaka_ai.ui.suggestions import choose_suggested_questions
from mamlaka_ai.utils.citations import Citation, strip_citation_markers
from mamlaka_ai.utils.language import detect_language, text_direction


USER_AVATAR_PATH = settings.assets_dir / "mamlaka-user-avatar.png"
BOT_AVATAR_PATH = settings.assets_dir / "mamlaka-ai-avatar.png"
SIDEBAR_LOGO_PATH = settings.assets_dir / "mamlaka-ai-sidebar-logo-v3.png"
HEADER_LOGO_PATH = settings.assets_dir / "almamlaka-tv-logo.png"

st.set_page_config(
    page_title=settings.app_title,
    page_icon=str(BOT_AVATAR_PATH),
    layout="centered",
    initial_sidebar_state="expanded",
)


CSS = """
<style>
    :root {
        --mamlaka-red: #b51f2e;
        --mamlaka-deep: #74131e;
        --mamlaka-ink: #20242b;
        --mamlaka-soft: #f7f3f2;
    }
    .stApp { background: linear-gradient(180deg, #fff 0%, #fff 72%, #faf6f5 100%); }
    [data-testid="stHeader"] { background: rgba(255, 255, 255, .92); }
    [data-testid="stSidebar"] > div:first-child { background: var(--mamlaka-soft); }
    .brand-card {
        display: flex; align-items: center; gap: 1rem; padding: .85rem 1rem;
        border: 1px solid #eadcda; border-radius: 18px; background: #fff;
        box-shadow: 0 7px 25px rgba(93, 19, 27, .08); margin-bottom: .75rem;
    }
    .brand-card img { width: 112px; height: auto; object-fit: contain; }
    .brand-title { font-size: 1.32rem; font-weight: 760; color: var(--mamlaka-ink); }
    .scope-note {
        border-right: 4px solid var(--mamlaka-red); background: var(--mamlaka-soft);
        padding: .7rem .9rem; border-radius: 10px; color: #4b3b3c; font-size: .9rem;
        margin-bottom: 1rem;
    }
    .scope-note [dir="rtl"] { margin-top: .28rem; }
    .welcome-copy { line-height: 1.7; }
    .welcome-copy [dir="rtl"] { margin-top: .65rem; }
    .suggestion-title {
        color: #777; font-size: .8rem; font-weight: 650; margin: .85rem 0 .4rem;
    }
    div[data-testid="stHorizontalBlock"] .stButton > button {
        min-height: 3.2rem; white-space: normal; line-height: 1.3;
        background: #fff; border-color: #e1c8c5;
    }
    .message-copy { line-height: 1.75; overflow-wrap: anywhere; }
    .message-direction-marker { display: none; }
    [data-testid="stChatMessage"]:has(.message-direction-marker[data-direction="rtl"])
    [data-testid="stMarkdownContainer"] {
        direction: rtl; text-align: right; line-height: 1.75;
    }
    [data-testid="stChatMessage"]:has(.message-direction-marker[data-direction="rtl"])
    [data-testid="stMarkdownContainer"] ul,
    [data-testid="stChatMessage"]:has(.message-direction-marker[data-direction="rtl"])
    [data-testid="stMarkdownContainer"] ol {
        padding-right: 1.5rem; padding-left: 0;
    }
    .source-title { color: #777; font-size: .78rem; margin-top: .7rem; }
    .source-list { display: flex; flex-wrap: wrap; gap: .38rem; margin-top: .32rem; }
    .source-badge {
        display: inline-flex; align-items: center; border: 1px solid #e1c8c5;
        background: #fff7f6; color: #6f1720; border-radius: 999px;
        padding: .28rem .66rem; font-size: .78rem; line-height: 1.35;
        white-space: nowrap;
    }
    .status-card {
        padding: .15rem .72rem; border: 1px solid #e8e8e8; border-radius: 12px;
        background: #fafafa; font-size: .84rem;
    }
    .status-item { padding: .65rem 0; border-bottom: 1px solid #ececec; }
    .status-item:last-child { border-bottom: 0; }
    .status-label { color: #777; font-size: .72rem; text-transform: uppercase; letter-spacing: .04em; }
    .status-value { color: var(--mamlaka-ink); font-weight: 650; margin-top: .12rem; }
    .kb-badge {
        display: inline-block; width: 100%; box-sizing: border-box; margin-top: .25rem;
        padding: .62rem .72rem; border: 1px solid #eadcda; border-radius: 12px;
        background: #fff7f6;
    }
    .kb-badge-label { color: #777; font-size: .7rem; text-transform: uppercase; letter-spacing: .04em; }
    .kb-badge-value { color: #6f1720; font-size: .82rem; font-weight: 650; margin-top: .12rem;
    }
    div[data-testid="stChatInput"] textarea:focus { border-color: var(--mamlaka-red); }
    [data-testid^="stChatMessageAvatar"] {
        background: #fff !important; border: 1px solid #eadcda;
        overflow: hidden; box-shadow: 0 2px 8px rgba(93, 19, 27, .08);
    }
    [data-testid^="stChatMessageAvatar"] img {
        background: #fff !important; object-fit: cover;
    }
    .stButton > button { border-color: #d6a9ad; }
    .stButton > button:hover { color: var(--mamlaka-red); border-color: var(--mamlaka-red); }
    [data-testid="stDialog"] {
        align-items: center; justify-content: center; padding: 1rem;
    }
    [data-testid="stDialog"] > div {
        background: var(--mamlaka-soft); border: 1px solid #e1c8c5;
        border-top: 4px solid var(--mamlaka-red); border-radius: 18px;
        box-shadow: 0 18px 50px rgba(93, 19, 27, .18); margin: 0;
        min-width: 0; width: min(26.25rem, calc(100vw - 2rem)) !important;
        max-width: calc(100vw - 2rem);
    }
    [data-testid="stDialog"] [role="dialog"] {
        background: transparent; border: 0; border-radius: 0;
        box-shadow: none; margin: 0;
    }
    [data-testid="stDialog"] h2 { color: var(--mamlaka-deep); }
    [data-testid="stDialog"] button[aria-label="Close"] {
        top: 1.35rem; right: 1.15rem; color: var(--mamlaka-ink);
    }
    [data-testid="stDialog"] .dialog-copy {
        color: #4b3b3c; line-height: 1.65; margin-bottom: .5rem;
    }
    @media (min-width: 768px) {
        body:has([data-testid="stSidebar"][aria-expanded="true"])
        [data-testid="stDialog"] > div {
            transform: translateX(150px);
        }
    }
    @media (max-width: 640px) {
        .brand-card img { width: 82px; }
        .brand-title { font-size: 1.08rem; }
    }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_answerer() -> Answerer:
    """Load and cache the RAG engine."""
    return Answerer(load_retriever())


def _safe_multiline(text: str) -> str:
    """Escape model text and keep line breaks."""
    return html.escape(text or "").replace("\n", "<br>")


def _png_data_uri(path: Path) -> str:
    """Embed a local PNG so the browser never needs an external logo request."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _citation_from_dict(payload: Dict[str, Any]) -> Citation:
    return Citation(
        document_name=str(payload["document_name"]),
        page_number=int(payload["page_number"]),
        section=payload.get("section"),
    )


def _source_display_name(document_name: str) -> str:
    """Return a short display name for a source file."""
    stem = Path(document_name).stem
    if stem.startswith("Almamlaka_"):
        stem = stem[len("Almamlaka_") :]
    label = stem.replace("_", " ")
    return label.replace("Team Governance", "Team & Governance")


def _render_sources(citations: Iterable[Dict[str, Any]], language: str) -> None:
    citations = list(citations)
    if not citations:
        return
    title = "المصادر" if language == "ar" else "Sources"
    seen: set[tuple[str, int]] = set()
    badges = []
    for payload in citations:
        citation = _citation_from_dict(payload)
        key = (citation.document_name, citation.page_number)
        if key in seen:
            continue
        seen.add(key)
        source = html.escape(_source_display_name(citation.document_name))
        badges.append(
            f'<span class="source-badge" dir="ltr">📄 {source} · Page '
            f'{citation.page_number}</span>'
        )
    st.markdown(
        f'<div dir="{text_direction(language)}" class="source-title">{title}</div>'
        f'<div dir="{text_direction(language)}" class="source-list">{"".join(badges)}</div>',
        unsafe_allow_html=True,
    )


def _render_copy_button(text: str, language: str) -> None:
    """Render a browser-side copy button for an answer."""
    encoded = base64.b64encode((text or "").encode("utf-8")).decode("ascii")
    label = "نسخ الإجابة" if language == "ar" else "Copy answer"
    copied = "تم النسخ" if language == "ar" else "Copied"
    st.iframe(
        f"""
        <style>
          body {{ margin: 0; background: transparent; font-family: sans-serif; }}
          .copy-row {{ display: flex; justify-content: flex-end; }}
          button {{
            border: 1px solid #d6a9ad; border-radius: 8px; background: #fff;
            color: #6f1720; padding: 5px 10px; cursor: pointer; font-size: 12px;
          }}
          button:hover {{ border-color: #b51f2e; }}
        </style>
        <div class="copy-row"><button id="copy-answer">{label}</button></div>
        <script>
          const encoded = "{encoded}";
          const bytes = Uint8Array.from(atob(encoded), character => character.charCodeAt(0));
          const answer = new TextDecoder().decode(bytes);
          const button = document.getElementById("copy-answer");
          button.addEventListener("click", async () => {{
            try {{
              await navigator.clipboard.writeText(answer);
            }} catch (error) {{
              const area = document.createElement("textarea");
              area.value = answer;
              document.body.appendChild(area);
              area.select();
              document.execCommand("copy");
              area.remove();
            }}
            button.textContent = "{copied}";
            setTimeout(() => button.textContent = "{label}", 1400);
          }});
        </script>
        """,
        height=34,
    )


def _render_message(message: Dict[str, Any]) -> None:
    role = message["role"]
    language = message.get("language") or detect_language(message.get("content", ""))
    avatar = str(USER_AVATAR_PATH if role == "user" else BOT_AVATAR_PATH)
    display_content = message.get("content", "")
    if role == "assistant" and message.get("citations"):
        display_content = strip_citation_markers(display_content)
    with st.chat_message(role, avatar=avatar):
        if role == "assistant":
            st.markdown(
                f'<span class="message-direction-marker" '
                f'data-direction="{text_direction(language)}"></span>',
                unsafe_allow_html=True,
            )
            st.markdown(display_content)
            _render_copy_button(display_content, language)
        else:
            st.markdown(
                f'<div class="message-copy" dir="{text_direction(language)}">'
                f'{_safe_multiline(display_content)}</div>',
                unsafe_allow_html=True,
            )
        if role == "assistant":
            _render_sources(message.get("citations", []), language)
            if settings.debug_retrieval and message.get("debug"):
                with st.expander("Retrieval debug / تفاصيل الاسترجاع", expanded=False):
                    st.json(message["debug"])
                    rows = message.get("retrieved_chunks") or []
                    if rows:
                        st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_welcome() -> None:
    """Show the bilingual welcome outside conversation history."""
    with st.chat_message("assistant", avatar=str(BOT_AVATAR_PATH)):
        st.markdown(
            """
            <div class="welcome-copy">
              <div dir="ltr">
                Hello! I'm Mamlaka AI Assistant. I'm glad to help you.<br>
                Ask me anything about the Almamlaka TV in Arabic or English.
              </div>
              <div dir="rtl">
                السلام عليكم! أنا مساعد المملكة الذكي، ويسعدني مساعدتك.<br>
                اسألني عن قناة المملكة بالعربية أو الإنجليزية.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_suggested_questions() -> str | None:
    """Show suggested questions and return the selected one."""
    st.markdown(
        '<div class="suggestion-title">Suggested questions · أسئلة مقترحة</div>',
        unsafe_allow_html=True,
    )
    selected = None
    questions = st.session_state.suggested_questions
    for index, (column, question) in enumerate(
        zip(st.columns(len(questions)), questions)
    ):
        if column.button(question, key=f"suggested-question-{index}", use_container_width=True):
            selected = question
    return selected


def _history_for_answerer(messages: Iterable[Dict[str, Any]]) -> list[tuple[str, str]]:
    """Return role and content for reference resolution."""
    return [(m["role"], m["content"]) for m in messages]


if "messages" not in st.session_state:
    st.session_state.messages = []
if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = choose_suggested_questions()


def _clear_conversation() -> None:
    """Remove all messages and prepare a new suggestion selection."""
    st.session_state.messages = []
    st.session_state.suggested_questions = choose_suggested_questions()


@st.dialog(
    "مسح المحادثة · Clear conversation",
    width="small",
    dismissible=True,
    on_dismiss="rerun",
)
def _confirm_clear_dialog() -> None:
    """Ask the user to confirm before clearing the chat."""
    st.markdown(
        """
        <div class="dialog-copy">
          <div dir="rtl">هل أنت متأكد من رغبتك في مسح المحادثة؟</div>
          <div dir="ltr">Are you sure you want to clear this conversation?</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    confirm_column, cancel_column = st.columns(2)
    if confirm_column.button(
        "نعم · Yes, clear",
        key="confirm-clear",
        use_container_width=True,
        on_click=_clear_conversation,
    ):
        st.rerun()
    if cancel_column.button(
        "إلغاء · Cancel",
        key="cancel-clear",
        use_container_width=True,
    ):
        st.rerun()


with st.sidebar:
    st.image(str(SIDEBAR_LOGO_PATH), width=220)
    st.markdown("### مساعد المملكة الذكي")
    st.caption("Mamlaka AI")
    st.markdown(
        '<div class="status-card">'
        '<div class="status-item"><div class="status-label">Knowledge Base</div>'
        '<div class="status-value">3 Approved PDFs</div></div>'
        '<div class="status-item"><div class="status-label">Languages</div>'
        '<div class="status-value">Arabic • English</div></div>'
        '<div class="status-item"><div class="status-label">Sources</div>'
        '<div class="status-value">Cited</div></div>'
        '<div class="status-item"><div class="status-label">Grounding</div>'
        '<div class="status-value">Enabled</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    if st.button(
        "مسح المحادثة · Clear conversation",
        key="clear-conversation",
        use_container_width=True,
    ):
        _confirm_clear_dialog()
    st.markdown(
        '<div class="kb-badge">'
        '<div class="kb-badge-label">Knowledge Base</div>'
        '<div class="kb-badge-value">3 Approved Project PDFs</div>'
        '</div>',
        unsafe_allow_html=True,
    )


st.markdown(
    f"""
    <div class="brand-card">
      <img src="{_png_data_uri(HEADER_LOGO_PATH)}" alt="Almamlaka TV logo">
      <div>
        <div class="brand-title">Mamlaka AI · مساعد المملكة الذكي</div>
      </div>
    </div>
    <div class="scope-note">
      <div dir="ltr">I don’t make things up — I answer only from approved documents.</div>
      <div dir="rtl">لا أختلق المعلومات — أجيب حصراً من المستندات المعتمدة.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

suggested_question = None
if not st.session_state.messages:
    _render_welcome()
    suggested_question = _render_suggested_questions()

try:
    with st.spinner("Loading the document index · جارٍ تحميل فهرس المستندات…"):
        answerer = get_answerer()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"{refusals.message(refusals.INDEX_MISSING, 'en')}\n\nDetails: {exc}")
    st.stop()
except Exception as exc:  # startup/configuration failures should be actionable in the UI
    st.error(
        "The application could not start. Check the embedding model and configured LLM provider, "
        f"then reload the page.\n\nDetails: {exc}"
    )
    st.stop()

for stored_message in st.session_state.messages:
    _render_message(stored_message)

typed_question = st.chat_input("اسأل مساعد المملكة... · Ask Mamlaka AI...")
question = suggested_question or typed_question
if question:
    language = detect_language(question)
    user_message = {"role": "user", "content": question, "language": language}
    history = _history_for_answerer(st.session_state.messages)
    st.session_state.messages.append(user_message)
    _render_message(user_message)

    spinner_text = "جارٍ البحث في المستندات…" if language == "ar" else "Searching the documents…"
    with st.spinner(spinner_text):
        result = answerer.answer(question, history=history)

    retrieved_rows = []
    if settings.debug_retrieval and result.retrieval:
        retrieved_rows = result.retrieval.debug_rows()

    assistant_message = {
        "role": "assistant",
        "content": result.answer,
        "language": result.language,
        "kind": result.kind,
        "citations": [asdict(citation) for citation in result.citations],
        "debug": result.debug if settings.debug_retrieval else {},
        "retrieved_chunks": retrieved_rows,
    }
    st.session_state.messages.append(assistant_message)
    _render_message(assistant_message)

    if result.kind == KIND_ERROR:
        st.caption(
            "The question remains in your history; retry after fixing the configured LLM provider."
        )
