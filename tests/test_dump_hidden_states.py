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
)


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
