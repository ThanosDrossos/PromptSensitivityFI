"""Prompt assembly — model-agnostic templates (anti-pattern: no silent prompt engineering).

Sprint-5 brief §6: "All prompts assembled from a single template per model,
documented in `code/prompts/templates/`. The only variables are: question
text, context paragraphs."

Two templates (v6 dual-ladder):
  - baseline (`assemble_qa_messages`): no reasoning, brief answer.
  - chain-of-thought (`assemble_qa_cot_messages`): step-by-step + `Answer:`
    line, for MuSiQue graded chain-completion scoring.

Both are single templates shared across all four models; the gateway handles
tokeniser-specific chat templating.
"""

from .templates.qa_prompt import (
    assemble_qa_messages,
    assemble_qa_cot_messages,
    parse_answer_line,
    QA_SYSTEM_PROMPT,
    QA_COT_SYSTEM_PROMPT,
    QA_USER_TEMPLATE,
)

__all__ = [
    "assemble_qa_messages",
    "assemble_qa_cot_messages",
    "parse_answer_line",
    "QA_SYSTEM_PROMPT",
    "QA_COT_SYSTEM_PROMPT",
    "QA_USER_TEMPLATE",
]
