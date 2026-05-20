"""Multi-hop QA prompt template. Sprint 5+.

ONE template across all four models. The only variables (per
Research_Design_v3 §6 anti-pattern rule):

  - {question}: the paraphrased question text
  - {context_block}: the ladder-selected paragraphs, formatted

System prompt is fixed; instruction "answer briefly" is baked into it.

When `paragraphs` is empty (level 0 of any ladder), the context block is
left out entirely so the model receives just the question — a true
closed-book condition.
"""

from __future__ import annotations

from collections.abc import Sequence

from ...data import HotpotParagraph
from ...models.schemas import ChatMessage


QA_SYSTEM_PROMPT = (
    "You answer factual multi-hop questions. Use ONLY the information in "
    "the provided context paragraphs (if any). If the context does not "
    "contain the answer, reply with the single word: unknown. Answer in a "
    "brief phrase or a single sentence — do not add reasoning, citations, "
    "or follow-up prose."
)


QA_USER_TEMPLATE_WITH_CONTEXT = (
    "Context:\n{context_block}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


QA_USER_TEMPLATE_NO_CONTEXT = (
    "Question: {question}\n\n"
    "Answer:"
)


# Re-exported for `prompts/__init__.py`.
QA_USER_TEMPLATE = QA_USER_TEMPLATE_WITH_CONTEXT


def _format_context_block(paragraphs: Sequence[HotpotParagraph]) -> str:
    """Render paragraphs as `Title: sentences...` blocks separated by blank lines."""
    parts: list[str] = []
    for p in paragraphs:
        body = p.joined().strip()
        if not body:
            continue
        parts.append(f"{p.title}: {body}")
    return "\n\n".join(parts)


def assemble_qa_messages(
    question: str,
    paragraphs: Sequence[HotpotParagraph],
) -> list[ChatMessage]:
    """Build (system, user) for one (paraphrase, ladder-level) pair.

    Args:
        question: the paraphrase text (the Variant question).
        paragraphs: ladder-selected paragraphs (already ordered by ladder).
                    Pass empty list for level 0.
    """
    question = question.strip()
    if paragraphs:
        context_block = _format_context_block(paragraphs)
        user_content = QA_USER_TEMPLATE_WITH_CONTEXT.format(
            context_block=context_block,
            question=question,
        )
    else:
        user_content = QA_USER_TEMPLATE_NO_CONTEXT.format(question=question)
    return [
        ChatMessage(role="system", content=QA_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_content),
    ]
