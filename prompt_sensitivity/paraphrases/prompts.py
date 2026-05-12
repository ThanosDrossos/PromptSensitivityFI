"""Role-templated paraphrase prompts (Razavi 2025 ECIR PromptSET pattern).

PromptSET (arXiv:2502.06065) generates paraphrases by asking the model to
rewrite a query in a target *persona's* voice while preserving the answer set.
The four roles below are the design-doc-pinned set; their wording is mine but
the structure (persona + invariance constraint + single-line output) follows
Razavi §3.1.

The system prompt is identical across roles; only the persona description and
a stylistic hint differ. Output is always one line, no preamble, no
explanations — that keeps post-processing trivial and avoids generator-side
"helpful" framing that would itself be a paraphrase artifact.
"""

from __future__ import annotations

from typing import Sequence

from ..models.schemas import ChatMessage
from .schemas import RoleName


ROLE_NAMES: Sequence[RoleName] = ("neutral", "journalist", "casual_user", "domain_expert")


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
