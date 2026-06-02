"""Reasoning ladder. Dataset_Evaluation_v6_Dual_Ladder.md §5.

A SEPARATE manipulation from the three context ladders. Instead of feeding
raw context paragraphs, it feeds the gold reasoning chain progressively: at
rung k the model receives the first k decomposition hops (each rendered as
"sub-question -> sub-answer") as scaffolding and must finish the rest.

To avoid handing over the answer, the FINAL hop is always withheld: the
maximum scaffold is `n_hops - 1` hops. So a 4-hop question produces rungs with
{0, 1, 2, 3} hops provided — never the full chain.

The headline experiment (v6 §5) compares this ladder's FI_in curve against the
context ladder's on the SAME MuSiQue questions: does feeding the reasoning lift
the chain-completion fraction faster per item than feeding the paragraphs that
contain it?

This builder does NOT select paragraphs (`paragraph_indices` stays empty). The
e2e driver renders the scaffold text from `question.question_decomposition`
sliced to `hops_provided`. We record `gold_count` as the number of gold
paragraphs *associated with the provided hops* purely for analytics.
"""

from __future__ import annotations

from ..data import MultiHopQuestion
from .schemas import LadderRow


def build_reasoning_ladder(question: MultiHopQuestion) -> list[LadderRow]:
    """Reasoning ladder rows for a MuSiQue question.

    Rungs: hops_provided = 0, 1, ..., n_hops-1 (final hop withheld). Returns
    n_hops rows. Raises ValueError for questions without a decomposition
    (HotpotQA / 2Wiki) — the caller must guard on `question.has_decomposition()`.
    """
    decomposition = question.question_decomposition
    if not decomposition:
        raise ValueError(
            f"question {question.id!r} has no decomposition; reasoning ladder "
            "is MuSiQue-only. Guard with question.has_decomposition()."
        )
    n_hops = len(decomposition)

    rows: list[LadderRow] = []
    # hops_provided 0..n_hops-1 — never the full chain (withhold final hop).
    for hops_provided in range(n_hops):
        provided = decomposition[:hops_provided]
        gold_para_idxs = {
            hop.supporting_paragraph_idx
            for hop in provided
            if hop.supporting_paragraph_idx is not None
        }
        gold_count = sum(
            1 for i in gold_para_idxs
            if 0 <= i < len(question.paragraphs) and question.paragraphs[i].is_gold
        )
        rows.append(
            LadderRow(
                question_id=question.id,
                ladder_type="reasoning",
                level_idx=hops_provided,
                level=hops_provided,          # the "amount" axis is #hops fed
                paragraph_indices=[],          # reasoning ladder feeds hops, not paragraphs
                paragraph_titles=[],
                gold_count=gold_count,
                permutation=None,
                ladder_family="reasoning",
                hops_provided=hops_provided,
            )
        )
    return rows


def render_reasoning_scaffold(question: MultiHopQuestion, hops_provided: int) -> str:
    """Render the first `hops_provided` hops as 'sub-question -> sub-answer' lines.

    Used by the e2e driver to build the scaffold block injected into the
    prompt. `#k` placeholders in sub-questions are left as the dataset wrote
    them (the scaffold already supplies the referenced sub-answers on earlier
    lines, so they are self-resolving in context).
    """
    lines: list[str] = []
    for hop in question.question_decomposition[:hops_provided]:
        lines.append(f"- {hop.sub_question.strip()} -> {hop.sub_answer.strip()}")
    return "\n".join(lines)
