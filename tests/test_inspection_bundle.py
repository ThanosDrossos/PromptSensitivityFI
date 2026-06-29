"""Item 2: the per-run start->finish inspection bundle renderer (pure function)."""

from __future__ import annotations

from prompt_sensitivity.scripts.e2e_smoke import _fmt, _render_inspection_md


def _rec():
    return {
        "question_id": "q1", "dataset": "musique", "n_hops": 2,
        "question": "Who leads the place?", "gold_answer": "Xanana",
        "decomposition": [{"hop": 0, "sub_question": "what place?", "sub_answer": "Timor"}],
        "ladder_family": "context", "ladder_type": "random", "level": 0,
        "model_key": "qwen_2_5_7b", "scoring_mode": "chain_completion",
        "paraphrases": [{
            "paraphrase_idx": 0, "paraphrase": "Who is the leader?",
            "prompt_user": "Context...\nQuestion: Who is the leader?",
            "f_response": "Step 1...\nAnswer: Xanana", "final_answer": "Xanana", "score": 1.0,
        }],
        "hsem": {"k": 2, "distinct_clusters": 1,
                 "per_paraphrase": {0: {"clusters": [0, 0], "answers": ["Xanana", "Xanana"]}}},
        "metrics": {"f_mean": 1.0, "final_answer_f_mean": 1.0, "h_sem_mean": 0.0,
                    "a_q": 1, "aufi_in": 0.0, "fi_out_mean": 0.0},
    }


def test_render_inspection_md_contains_every_step(tmp_path):
    md = _render_inspection_md([_rec()], tmp_path)  # no paraphrase parquet -> roles blank, no crash
    for needle in ["Run inspection bundle", "q1", "Gold answer:** Xanana",
                   "Gold reasoning chain", "Who is the leader?", "context ladder",
                   "Prompt put to the model", "F-response", "H_sem", "|A_q|=1"]:
        assert needle in md, f"missing: {needle!r}"


def test_fmt_handles_none_and_nan():
    assert _fmt(None) == "—"
    assert _fmt(0.5) == "0.500"
