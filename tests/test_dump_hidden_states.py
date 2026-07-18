"""Hidden-state dump for FI probes: vector codec, cache-only universe reads,
resume keys. GPU-free — the provider forward itself is exercised by the
cluster smoke, not here."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prompt_sensitivity.scripts.dump_hidden_states import (
    decode_vec,
    done_cells,
    encode_vec,
    universe_texts,
    validate_hidden_dump,
)


def _dump_frame(n_para=3, layers=(7, 28), dim=8, model="qwen_2_5_7b"):
    rng = np.random.default_rng(0)
    return pd.DataFrame([
        {"question_id": "q1", "spec_level": lvl, "model_key": model,
         "paraphrase_idx": p, "paraphrase": f"p{p}?", "context_mode": "uniform_evidence",
         "position": "tbg", "layer_idx": layer, "layer_frac": layer / 28,
         "dim": dim, "dtype": "float16",
         "vec": encode_vec(rng.normal(size=dim).astype(np.float32))}
        for lvl in (0, 1) for p in range(n_para) for layer in layers
    ])


def test_validate_hidden_dump_accepts_wellformed():
    assert validate_hidden_dump(_dump_frame(), decode_stride=1) == []


def test_validate_hidden_dump_catches_corruption():
    # torn blob (wrong byte length)
    df = _dump_frame()
    df.loc[0, "vec"] = df.loc[0, "vec"][:-2]
    assert any("dim*2" in p for p in validate_hidden_dump(df))

    # a hole in the paraphrase indices of one (cell, layer)
    df2 = _dump_frame()
    df2 = df2[~((df2.spec_level == 0) & (df2.paraphrase_idx == 1) & (df2.layer_idx == 7))]
    assert any("not contiguous" in p for p in validate_hidden_dump(df2))

    # mixed dims within one model (e.g. two runs with different configs merged)
    df3 = pd.concat([_dump_frame(dim=8), _dump_frame(dim=16)], ignore_index=True)
    assert any("mixed dims" in p for p in validate_hidden_dump(df3))

    # non-finite vector
    df4 = _dump_frame()
    bad = np.full(8, np.nan, dtype=np.float32)
    df4.loc[0, "vec"] = encode_vec(bad)
    assert any("non-finite" in p for p in validate_hidden_dump(df4, decode_stride=1))

    # missing schema / empty
    assert validate_hidden_dump(pd.DataFrame({"question_id": []}))
    assert validate_hidden_dump(_dump_frame().iloc[0:0]) == ["empty dump"]


def test_vec_roundtrip_float16():
    v = np.random.default_rng(0).normal(size=3584).astype(np.float32)
    blob = encode_vec(v)
    assert len(blob) == 3584 * 2                    # float16, half of float32
    out = decode_vec(blob, 3584)
    assert out.dtype == np.float16
    np.testing.assert_array_equal(out, v.astype(np.float16))
    with pytest.raises(ValueError):
        decode_vec(blob, 4096)                       # dim mismatch must not pass silently


def test_universe_texts_orders_filters_and_reports_missing():
    para = pd.DataFrame([
        # deliberately out of order + a rejected row that must be excluded
        {"question_id": "q1", "spec_level": 0, "outcome": "accepted", "paraphrase_idx": 1, "text": "b"},
        {"question_id": "q1", "spec_level": 0, "outcome": "accepted", "paraphrase_idx": 0, "text": "a"},
        {"question_id": "q1", "spec_level": 0, "outcome": "constraint_mismatch", "paraphrase_idx": 9, "text": "X"},
        {"question_id": "q1", "spec_level": 1, "outcome": "singleton_fallback", "paraphrase_idx": 0, "text": "orig"},
    ])
    assert universe_texts(para, "q1", 0, 10) == ["a", "b"]     # idx order, rejected dropped
    assert universe_texts(para, "q1", 0, 1) == ["a"]           # max_n cap
    assert universe_texts(para, "q1", 1, 10) == ["orig"]       # fallback rows count
    assert universe_texts(para, "q2", 0, 10) is None           # missing -> None (skip, never generate)


def test_done_cells_resume_keys():
    assert done_cells(None) == set()
    assert done_cells(pd.DataFrame()) == set()
    df = pd.DataFrame([
        {"question_id": "q1", "spec_level": 0, "layer_idx": 7},
        {"question_id": "q1", "spec_level": 0, "layer_idx": 28},   # same cell, extra layer row
        {"question_id": "q2", "spec_level": 1, "layer_idx": 7},
    ])
    assert done_cells(df) == {("q1", 0), ("q2", 1)}
