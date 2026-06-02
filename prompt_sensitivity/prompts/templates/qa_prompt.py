"""Multi-hop QA prompt templates. Sprint 5+ / v6 dual-ladder.

TWO templates, both model-agnostic (one template for all models, per
Research_Design_v3 §6 anti-pattern rule). The only per-cell variables are
`{question}` and `{context_block}`.

  1. BASELINE (`QA_SYSTEM_PROMPT` / `assemble_qa_messages`): "answer briefly,
     no reasoning". Used for HotpotQA / 2Wiki and for the binary final-answer
     score on every dataset.

  2. CHAIN-OF-THOUGHT (`QA_COT_SYSTEM_PROMPT` / `assemble_qa_cot_messages`):
     "work step by step, state each intermediate conclusion, then a final
     `Answer:` line". Used for MuSiQue, where the graded chain-completion
     scorer (v6 §2) needs the model to EXPOSE its intermediate answers. The
     `Answer:` line is parsed back out for the secondary binary score.

When `paragraphs` is empty (level 0 of any ladder), the context block is left
out entirely so the model receives just the question — a true closed-book
condition.
"""

from __future__ import annotations

import re
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
    """Build (system, user) for the BASELINE (no-reasoning) prompt.

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


# --------------------------------------------------------------------------- #
# Chain-of-thought template (MuSiQue graded chain-completion scoring)         #
# --------------------------------------------------------------------------- #


QA_COT_SYSTEM_PROMPT = (
    "You answer factual multi-hop questions by reasoning step by step. "
    "Use ONLY the information in the provided context paragraphs (if any). "
    "Work through the question one hop at a time: on each line, state one "
    "intermediate conclusion you can draw, naming the specific entity, date, "
    "or fact for that step. After the reasoning, write the final answer on a "
    "new line that starts exactly with 'Answer:'. If the context does not "
    "let you resolve a step, state what is missing for that step and continue."
)


QA_COT_USER_TEMPLATE_WITH_CONTEXT = (
    "Context:\n{context_block}\n\n"
    "Question: {question}\n\n"
    "Reason step by step, then end with a line starting 'Answer:'."
)


QA_COT_USER_TEMPLATE_NO_CONTEXT = (
    "Question: {question}\n\n"
    "Reason step by step, then end with a line starting 'Answer:'."
)


def assemble_qa_cot_messages(
    question: str,
    paragraphs: Sequence[HotpotParagraph],
) -> list[ChatMessage]:
    """Build (system, user) for the CHAIN-OF-THOUGHT prompt (MuSiQue).

    The model's full reasoning text is consumed by `chain_completion_score`;
    the final `Answer:` line is parsed by `parse_answer_line` for the
    secondary binary score.
    """
    question = question.strip()
    if paragraphs:
        context_block = _format_context_block(paragraphs)
        user_content = QA_COT_USER_TEMPLATE_WITH_CONTEXT.format(
            context_block=context_block,
            question=question,
        )
    else:
        user_content = QA_COT_USER_TEMPLATE_NO_CONTEXT.format(question=question)
    return [
        ChatMessage(role="system", content=QA_COT_SYSTEM_PROMPT),
        ChatMessage(role="user", content=user_content),
    ]


_ANSWER_LINE_RE = re.compile(r"(?im)^\s*answer\s*:\s*(.+?)\s*$")


def parse_answer_line(response: str) -> str:
    """Extract the final `Answer:` line from a CoT response.

    Returns the text after the last `Answer:` marker. Falls back to the whole
    (stripped) response if no marker is present — so the binary scorer always
    has something to score.
    """
    if not response:
        return ""
    matches = _ANSWER_LINE_RE.findall(response)
    if matches:
        return matches[-1].strip()
    return response.strip()
