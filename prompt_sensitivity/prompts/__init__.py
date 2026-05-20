"""Prompt assembly — single template per model (anti-pattern: no silent prompt engineering).

Sprint-5 brief §6: "All prompts assembled from a single template per model,
documented in `code/prompts/templates/`. The only variables are: question
text, context paragraphs, optional 'answer briefly' instruction."

For Sprint 5 we use ONE template across all four models. The kit.gpt-4.1
and Llama/Teuken/Qwen all accept the same chat-completion message shape;
no per-model branching is needed at the messaging layer (the gateway
handles tokeniser-specific chat templating).
"""

from .templates.qa_prompt import (
    assemble_qa_messages,
    QA_SYSTEM_PROMPT,
    QA_USER_TEMPLATE,
)

__all__ = ["assemble_qa_messages", "QA_SYSTEM_PROMPT", "QA_USER_TEMPLATE"]
