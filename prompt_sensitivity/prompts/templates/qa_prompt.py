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


# The CoT system prompt asks for a line starting exactly with "Answer:", but the
# local 7-8B instruct models (Qwen2.5 / Mistral / Llama-3.1) routinely dress it
# up — markdown-bold (`**Answer:**`, `**Answer**:`), heading (`### Answer`), or
# "Final Answer:". The old `^answer:` regex missed all of these and fell back to
# the WHOLE reasoning essay, which the asymmetric gold->answer NLI scorer
# (scoring/nli_with_gold.py) then scored 0 — so final_answer_f_mean read ~0 in
# cells where chain-F was already high. We therefore (a) tolerate leading
# markdown / blockquote / bullet glyphs and emphasis around the label, (b) accept
# "Final Answer", (c) read the value off the following line when the label line
# is bare, and (d) fall back to the LAST non-empty line (the model's conclusion)
# rather than the entire essay. See Dataset_Evaluation_v6_Dual_Ladder §2 (the
# `Answer:` line is the secondary binary score; chain-completion is primary).
_ANSWER_LABEL_RE = re.compile(
    r"(?im)^[ \t>#*_\-]*(?:final\s+answer|answer)[ \t*_]*[:\-–—][ \t]*(.*)$"
)

# "the answer is X" / "answer was X" prose, for responses that skip the label.
_ANSWER_PROSE_RE = re.compile(
    r"(?is)\b(?:the\s+)?(?:final\s+)?answer\s+(?:is|was|would\s+be)\b[:\s]*(.+?)[.\n]"
)

# Markdown emphasis, quotes, and trailing punctuation to peel off an extracted span.
_ANSWER_STRIP = " \t\r\n*_`\"'.,;:!?()[]"


def _clean_answer(span: str) -> str:
    """Strip surrounding markdown emphasis / quotes / punctuation from a span."""
    return span.strip().strip(_ANSWER_STRIP).strip()


# A cleaned span that is JUST the label word (e.g. a refusal ending in a bare
# "Answer:") is not an answer — reject it so we never extract the literal "Answer".
_LABEL_NOISE = {"answer", "final answer", "the answer", "the final answer"}


def _is_label_noise(s: str) -> bool:
    return s.strip().lower() in _LABEL_NOISE


def parse_answer_line(response: str) -> str:
    """Extract the final answer from a CoT response, tolerantly.

    Resolution order:
      1. The LAST explicit answer-label line (`Answer:`, `**Answer:**`,
         `Final Answer:`, `### Answer:`, ...). If the label line is bare
         (value on the next line), take the next non-empty line.
      2. A "the answer is X" prose statement (last occurrence).
      3. The last non-empty line of the response.

    Surrounding markdown / quotes / trailing punctuation are stripped. Never
    returns the whole multi-line essay — that systematically scored 0 under the
    gold->answer NLI scorer (see the module-level note).
    """
    if not response:
        return ""
    text = response.strip()

    # 1. explicit answer-label line(s) — take the last.
    label_matches = _ANSWER_LABEL_RE.findall(text)
    if label_matches:
        cand = _clean_answer(label_matches[-1])
        if cand and not _is_label_noise(cand):
            return cand
        # Label present but value sits on a following non-empty, non-label line.
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if _ANSWER_LABEL_RE.match(line):
                for nxt in lines[i + 1:]:
                    c = _clean_answer(nxt)
                    if c and not _is_label_noise(c):
                        return c

    # 2. "the answer is X" prose — last occurrence.
    prose = list(_ANSWER_PROSE_RE.finditer(text + "\n"))
    if prose:
        cand = _clean_answer(prose[-1].group(1))
        if cand and not _is_label_noise(cand):
            return cand

    # 3. fallback: last non-empty, non-label line (NOT the whole essay). A response
    #    that is only a bare "Answer:" (a refusal) yields "" rather than "Answer".
    for line in reversed(text.splitlines()):
        c = _clean_answer(line)
        if c and not _is_label_noise(c):
            return c
    return ""
