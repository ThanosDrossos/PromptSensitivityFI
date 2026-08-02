"""Render the L_mid human-review file (FINAL_PHASE_PLAN C1 HARD GATE).

Reads the midlevel cache + AmbigQA and writes a markdown review of the first
N accepted rewrites: the ambiguous question, every interpretation marked
ADMITTED/EXCLUDED (with the judge's verdict), and the generated L_mid text.
A human must eyeball these BEFORE the full multilevel run is launched —
the gate script prints the file path as its last line.

    python -m prompt_sensitivity.scripts.show_midlevel \
        --mid-cache data/midlevel_questions.parquet \
        --n 10 --out data/ml_review_sample.md
"""

from __future__ import annotations

import argparse

from loguru import logger

from ..config import load_config
from ..data.load_ambigqa import load_ambigqa
from ..logging_setup import configure_logging
from ..specificity.midlevel import load_mid_cache


def render_review(mids: dict, questions: dict, n: int) -> str:
    accepted = [m for m in mids.values() if m.outcome == "accepted"]
    failed = [m for m in mids.values() if m.outcome != "accepted"]
    lines = [
        "# L_mid human review sample (C1 hard gate)",
        "",
        f"Cache coverage: **{len(accepted)} accepted / {len(failed)} failed** "
        f"({len(mids)} attempted). Review the rewrites below; the full run is "
        "GO only if the admitted/excluded readings match your own reading of "
        "each L_mid text.",
        "",
    ]
    for k, mid in enumerate(accepted[:n], 1):
        q = questions.get(mid.question_id)
        if q is None:
            continue
        lines += [f"## {k}. qid={mid.question_id}  (m0={q.m0()}, "
                  f"attempts={mid.attempts})", "",
                  f"**Ambiguous Q:** {q.question}", "",
                  f"**L_mid rewrite:** {mid.text}", ""]
        subset = set(mid.subset)
        for i, interp in enumerate(q.interpretations):
            want = "ADMITTED " if i in subset else "EXCLUDED "
            vote = ""
            if i < len(mid.judge_admits):
                agree = mid.judge_admits[i] == (i in subset)
                vote = "judge ok" if agree else "JUDGE DISAGREES"
            lines.append(f"- [{want}] {interp.disambiguated_question}  ({vote})")
        lines.append("")
    if failed:
        lines += ["## Failed (excluded from the arm)", ""]
        lines += [f"- qid={m.question_id} after {m.attempts} attempts"
                  for m in failed[:20]]
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mid-cache", type=str, default="data/midlevel_questions.parquet")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", type=str, default="data/ml_review_sample.md")
    args = ap.parse_args()

    configure_logging("show_midlevel")
    config = load_config()
    root = config.repo_root()

    mids = load_mid_cache(root / args.mid_cache)
    if not mids:
        logger.error("no midlevel cache at {} — run the ML prep first", args.mid_cache)
        return 1
    acfg = config.sampling.ambigqa
    questions = {
        q.id: q for q in load_ambigqa(
            hf_dataset=acfg.hf_dataset, hf_config=acfg.hf_config, split=acfg.split,
            min_interpretations=acfg.min_interpretations,
            include_single_answer_anchor=acfg.include_single_answer_anchor,
        )
    }
    out_path = root / args.out
    out_path.write_text(render_review(mids, questions, args.n), encoding="utf-8")
    logger.info("review file written")
    print(f"REVIEW FILE: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
