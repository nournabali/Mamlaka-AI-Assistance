"""Localized application messages and refusals."""

from __future__ import annotations

from typing import Dict

NOT_FOUND: Dict[str, str] = {
    "en": (
        "The information you asked about is not provided in the project documents. Can I help "
        "you with another question about the Almamlaka TV project?"
    ),
    "ar": (
        "المعلومات التي سألت عنها غير واردة في مستندات المشروع. هل يمكنني مساعدتك في سؤال آخر "
        "عن مشروع قناة المملكة؟"
    ),
}

CAPABILITY_REFUSAL: Dict[str, str] = {
    "en": (
        "Sorry, this request is outside the scope. I can help you with questions about the "
        "Almamlaka TV project based on the documents."
    ),
    "ar": (
        "آسف جداً، هذا الطلب خارج النطاق. يمكنني مساعدتك في الأسئلة المتعلقة بمشروع قناة "
        "المملكة بالاعتماد على المستندات."
    ),
}

INJECTION_REFUSAL: Dict[str, str] = {
    "en": (
        "Sorry, that request is outside my capabilities. I can only answer questions about the "
        "Almamlaka TV project using the three approved documents. Can I help you with anything "
        "else about the project?"
    ),
    "ar": (
        "عذراً، هذا الطلب خارج نطاق قدراتي. يمكنني فقط الإجابة عن الأسئلة المتعلقة بمشروع قناة "
        "المملكة استناداً إلى المستندات الثلاثة المعتمدة. هل يمكنني مساعدتك في أي سؤال آخر عن "
        "المشروع؟"
    ),
}

GUESS_REFUSAL: Dict[str, str] = {
    "en": (
        "Sorry, guessing or inventing an answer is outside my capabilities. I can only answer from "
        "the three approved Almamlaka TV documents. Can I help you with anything else about the "
        "project?"
    ),
    "ar": (
        "عذراً، التخمين أو اختلاق إجابة خارج نطاق قدراتي. يمكنني الإجابة فقط بالاعتماد على مستندات "
        "قناة المملكة الثلاثة المعتمدة. هل يمكنني مساعدتك في أي سؤال آخر عن المشروع؟"
    ),
}

FULL_DOCUMENT_REFUSAL: Dict[str, str] = {
    "en": (
        "Sorry, providing an entire PDF verbatim is outside my chat capabilities. I can summarize "
        "the document or answer questions about a specific section with citations. Would you like "
        "a summary instead?"
    ),
    "ar": (
        "عذراً، عرض محتوى ملف PDF كاملاً حرفياً خارج نطاق قدراتي في المحادثة. يمكنني تلخيص "
        "المستند أو الإجابة عن أسئلة حول قسم محدد مع ذكر المصادر. هل ترغب في ملخص بدلاً من ذلك؟"
    ),
}

UNCITED_ANSWER: Dict[str, str] = {
    "en": (
        "I found related passages in the documents but couldn't produce an answer with reliable "
        "source citations, so I'm not going to present it as fact. Could you rephrase the question, "
        "or ask about a more specific detail?"
    ),
    "ar": (
        "وجدت مقاطع ذات صلة في المستندات لكنني لم أتمكّن من صياغة إجابة مع إسناد موثوق إلى المصدر، "
        "ولذلك لن أقدّمها كمعلومة مؤكدة. هل يمكنك إعادة صياغة السؤال أو السؤال عن تفصيل أكثر تحديداً؟"
    ),
}

LLM_UNAVAILABLE: Dict[str, str] = {
    "en": (
        "I can't use the configured language-model provider right now, so I can't compose an "
        "answer. Provider: `{provider}`; model: `{model}`. Check its configuration and try again. "
        "Details: {detail}"
    ),
    "ar": (
        "لا أستطيع استخدام مزوّد النموذج اللغوي المحدَّد حالياً، ولذلك لا يمكنني صياغة إجابة. "
        "المزوّد: `{provider}`؛ النموذج: `{model}`. تحقّق من الإعدادات ثم أعد المحاولة. "
        "التفاصيل: {detail}"
    ),
}

LLM_ERROR: Dict[str, str] = {
    "en": "The language model returned an error while composing the answer: {detail}",
    "ar": "أعاد النموذج اللغوي خطأً أثناء صياغة الإجابة: {detail}",
}

INDEX_MISSING: Dict[str, str] = {
    "en": (
        "The document index has not been built yet, so there is nothing to search. "
        "Run `python scripts/build_index.py` and reload the app."
    ),
    "ar": (
        "لم يتم بناء فهرس المستندات بعد، ولذلك لا يوجد ما يمكن البحث فيه. "
        "نفّذ الأمر `python scripts/build_index.py` ثم أعد تحميل التطبيق."
    ),
}

GREETING: Dict[str, str] = {
    "en": (
        "Hello! I'm Mamlaka AI Assistant. I'm glad to help you.\n\n"
        "Ask me anything about the Almamlaka TV project in Arabic or English."
    ),
    "ar": (
        "السلام عليكم! أنا مساعد المملكة الذكي، ويسعدني مساعدتك.\n\n"
        "اسألني عن قناة المملكة بالعربية أو الإنجليزية."
    ),
}


def message(catalogue: Dict[str, str], language: str, **kwargs: object) -> str:
    template = catalogue.get(language) or catalogue["en"]
    return template.format(**kwargs) if kwargs else template
