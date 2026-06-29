"""Role-templated paraphrase prompts (Razavi 2025 ECIR PromptSET pattern).

PromptSET (arXiv:2502.06065) generates paraphrases by asking the model to
rewrite a query in a target *persona's* voice while preserving the answer set.
The role set below (4 design-doc-pinned + 4 added 2026-06-29 to widen surface
variety) all follow the same structure (persona + invariance constraint +
single-line output) per Razavi §3.1. Which roles a run actually uses is selected
by `config.paraphrases.templates`, so the set is trivially adjustable.

The system prompt is identical across roles; only the persona description and
a stylistic hint differ. Output is always one line, no preamble, no
explanations — that keeps post-processing trivial and avoids generator-side
"helpful" framing that would itself be a paraphrase artifact.
"""

from __future__ import annotations

from typing import Sequence

from ..models.schemas import ChatMessage
from .schemas import RoleName


ROLE_NAMES: Sequence[RoleName] = (
    "neutral", "journalist", "casual_user", "domain_expert",
    # 2026-06-29: four additional registers to widen surface-form variety. They
    # span the same principled axis (persona-conditioned rewrite preserving the
    # answer set); the NLI + gold-constraint filters still guarantee semantic
    # equivalence, so they add diversity WITHOUT answer-set bias. More personas =>
    # more distinct accepted paraphrases / fewer dropped questions, at the same
    # EVAL cost (the eval is capped by max_paraphrases, not by persona count;
    # only the one-off paraphrase-prep job does more generation).
    "student", "terse_keyword", "formal_academic", "second_language",
)


_PERSONA: dict[RoleName, str] = {
    "neutral": (
        "Rewrite the question in clear, neutral, encyclopedia-style English. "
        "Avoid colloquialisms, opinions, or stylistic flourish."
    ),
    "journalist": (
        "Rewrite the question as a news journalist would phrase it for a "
        "factual article. Be precise, attribute nothing, use a professional "
        "register, and prefer active voice."
    ),
    "casual_user": (
        "Rewrite the question as a casual user might type it into a chatbot "
        "or web search box. Informal but not slang-heavy; you may drop "
        "articles or use contractions; questions phrased as imperatives or "
        "even fragments are fine, as long as the meaning is preserved."
    ),
    "domain_expert": (
        "Rewrite the question as a domain expert would phrase it among "
        "colleagues. Precise terminology, slightly higher register, can "
        "assume some shared background but must remain self-contained."
    ),
    "student": (
        "Rewrite the question as an inquisitive student studying the topic "
        "would ask it: plain and direct, possibly opening with 'Can you tell "
        "me' or 'I want to know'. Keep every entity and constraint that fixes "
        "the answer."
    ),
    "terse_keyword": (
        "Rewrite the question as a terse keyword / search-style query: "
        "telegraphic, drop articles and filler words, but KEEP every named "
        "entity, relation, and constraint that determines the answer. It need "
        "not be a grammatical sentence."
    ),
    "formal_academic": (
        "Rewrite the question in a formal academic register, as in a scholarly "
        "reference work: complete sentences, precise wording, no contractions "
        "or colloquialisms."
    ),
    "second_language": (
        "Rewrite the question as a careful non-native English speaker would: "
        "simple, grammatical sentences and common vocabulary, phrasing that may "
        "be slightly literal, with the meaning fully preserved."
    ),
}


_SYSTEM = (
    "You are a careful paraphrase generator. Your task is to rewrite a "
    "single question so the answer set is preserved bit-for-bit. The "
    "rewritten question MUST have the same correct answer (and the same set "
    "of acceptable answers) as the original. You MAY change wording, word "
    "order, syntax, and tone. You MUST NOT add or remove information that "
    "would change which answers are correct. You MUST NOT answer the "
    "question.\n\n"
    "Output exactly one line: the rewritten question. No preamble, no "
    "explanation, no surrounding quotes, no trailing notes."
)


_USER_TEMPLATE = (
    "Persona to write as: {persona}\n\n"
    "Original question:\n"
    "{question}\n\n"
    "Rewritten question (one line, in the persona above, with the same "
    "answer set as the original):"
)


def build_paraphrase_messages(question: str, role: RoleName) -> list[ChatMessage]:
    """Build the (system, user) message pair for one paraphrase request.

    The system prompt is identical across roles; only the persona varies, so
    cache hits are still maximally reused within a question across samples.
    """
    if role not in _PERSONA:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLE_NAMES}")
    persona = _PERSONA[role]
    user_text = _USER_TEMPLATE.format(persona=persona, question=question.strip())
    return [
        ChatMessage(role="system", content=_SYSTEM),
        ChatMessage(role="user", content=user_text),
    ]


def list_persona_descriptions() -> dict[RoleName, str]:
    """Returned for documentation / write-up purposes only."""
    return dict(_PERSONA)
