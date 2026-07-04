"""AmbigQA loader (HF `ambig_qa`, config `light`, split `validation`).

Schema VERIFIED against the live HF dataset (2026-07-04, 2002 validation rows):

    {id: str, question: str,
     annotations: {type:    list[str],              # parallel lists — one entry
                   answer:  list[list[str]],        # per annotator annotation
                   qaPairs: list[{question: list[str],
                                  answer:   list[list[str]]}]}}

i.e. HF flattens the record's annotation LIST into a dict of PARALLEL lists
(`type[i]` <-> `answer[i]` <-> `qaPairs[i]`), and each `qaPairs[i]` is itself
parallel lists (`question[j]` <-> `answer[j]`, the latter being the accepted
variant list for that disambiguated question). A record can carry BOTH a
`singleAnswer` and a `multipleQAs` annotation (different annotators).

The canonical GitHub release (`shmsw25/AmbigQA`) instead ships annotations as a
list of dicts (`[{type, answer} | {type, qaPairs:[{question: str, answer:
[str]}]}]`). `parse_ambigqa_record` normalises BOTH forms so the fixture (release
form) and the HF stream parse identically.

Mapping: the first `multipleQAs` annotation with >=1 usable pair wins ->
interpretations; records without one fall back to `singleAnswer` (m0 = 1,
dropped by default via `min_interpretations=2`, per the pivot spec §3.2).
"""

from __future__ import annotations

from typing import Any

from loguru import logger

from .ambigqa_schemas import AmbigInterpretation, AmbigQuestion


def _norm_answers(raw: Any) -> list[str]:
    """Normalise an answer field (str | list[str]) -> stripped non-empty list."""
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    return [a.strip() for a in raw if isinstance(a, str) and a.strip()]


def _iter_annotations(annotations: Any):
    """Yield (type, single_answers, qa_pairs) per annotation, both dataset forms.

    qa_pairs is normalised to list[(disambiguated_question: str, answers: list[str])].
    """
    if annotations is None:
        return
    # --- HF flattened form: dict of parallel lists ---------------------------
    if isinstance(annotations, dict):
        types = annotations.get("type") or []
        answers = annotations.get("answer") or []
        qa_pairs_raw = annotations.get("qaPairs") or []
        for i, ann_type in enumerate(types):
            single = _norm_answers(answers[i]) if i < len(answers) else []
            pairs: list[tuple[str, list[str]]] = []
            if i < len(qa_pairs_raw) and isinstance(qa_pairs_raw[i], dict):
                qp = qa_pairs_raw[i]
                qs = qp.get("question") or []
                ans = qp.get("answer") or []
                for j, dq in enumerate(qs):
                    variants = _norm_answers(ans[j]) if j < len(ans) else []
                    if isinstance(dq, str) and dq.strip() and variants:
                        pairs.append((dq.strip(), variants))
            yield str(ann_type), single, pairs
        return
    # --- GitHub release form: list of dicts ----------------------------------
    if isinstance(annotations, list):
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            ann_type = str(ann.get("type", ""))
            single = _norm_answers(ann.get("answer"))
            pairs = []
            for qp in ann.get("qaPairs") or []:
                if not isinstance(qp, dict):
                    continue
                dq = qp.get("question")
                variants = _norm_answers(qp.get("answer"))
                if isinstance(dq, str) and dq.strip() and variants:
                    pairs.append((dq.strip(), variants))
            yield ann_type, single, pairs


def parse_ambigqa_record(record: dict) -> AmbigQuestion | None:
    """One raw record (either form) -> AmbigQuestion, or None if unusable.

    The first `multipleQAs` annotation with a usable pair defines the
    interpretations (deterministic across runs — annotation order is stable in
    the release). Duplicate disambiguated questions are dropped (they would
    inflate m0, the headline ambiguity count). Records with no usable
    `multipleQAs` fall back to `singleAnswer` as a single interpretation
    anchored on the original question text (m0 = 1).
    """
    qid = str(record.get("id") or "").strip()
    question = str(record.get("question") or "").strip()
    if not qid or not question:
        return None

    interps: list[AmbigInterpretation] = []
    single_fallback: list[str] | None = None
    for ann_type, single, pairs in _iter_annotations(record.get("annotations")):
        if ann_type == "multipleQAs" and pairs and not interps:
            seen: set[str] = set()
            for dq, variants in pairs:
                key = " ".join(dq.lower().split())
                if key in seen:
                    continue
                seen.add(key)
                interps.append(
                    AmbigInterpretation(disambiguated_question=dq, answers=variants)
                )
        elif ann_type == "singleAnswer" and single and single_fallback is None:
            single_fallback = single

    if not interps:
        if single_fallback is None:
            return None
        interps = [
            AmbigInterpretation(disambiguated_question=question, answers=single_fallback)
        ]
    return AmbigQuestion(id=qid, question=question, interpretations=interps)


def accepts(
    q: AmbigQuestion, *, min_interpretations: int = 2,
    include_single_answer_anchor: bool = False,
) -> bool:
    """Filter predicate: keep genuinely ambiguous records (m0 >= min), plus
    optionally the unambiguous singleAnswer anchors (m0 == 1)."""
    if q.m0() >= min_interpretations:
        return True
    return include_single_answer_anchor and q.m0() == 1


def load_ambigqa(
    *,
    hf_dataset: str = "ambig_qa",
    hf_config: str = "light",
    split: str = "validation",
    min_interpretations: int = 2,
    include_single_answer_anchor: bool = False,
    cache_dir: str | None = None,
) -> list[AmbigQuestion]:
    """Load + parse + filter AmbigQA. Test split is not public -> validation."""
    from datasets import load_dataset

    ds = load_dataset(hf_dataset, hf_config, split=split, cache_dir=cache_dir)
    out: list[AmbigQuestion] = []
    n_unparsed = 0
    for record in ds:
        q = parse_ambigqa_record(record)
        if q is None:
            n_unparsed += 1
            continue
        if accepts(
            q,
            min_interpretations=min_interpretations,
            include_single_answer_anchor=include_single_answer_anchor,
        ):
            out.append(q)
    logger.info(
        "AmbigQA {}/{}: {} rows -> {} kept (min_interpretations={}, anchors={}, unparsed={})",
        hf_dataset, split, len(ds), len(out),
        min_interpretations, include_single_answer_anchor, n_unparsed,
    )
    return out
