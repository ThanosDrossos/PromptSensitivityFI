"""The gate's assert logic (check_parquet) on synthetic parquets — no models."""

from __future__ import annotations

import pandas as pd

from prompt_sensitivity.scripts.smoke_specificity import check_parquet


def _df(f0=0.4, f1=0.7, spec0=0.0, spec1=1.5):
    rows = []
    for q in ["q1", "q2"]:
        rows.append({"question_id": q, "model_key": "m", "spec_level": 0,
                     "fi_spec": spec0, "m_valid": 3, "f_mean": f0, "aufi_in": 0.8,
                     "fi_out_mean": 1.0, "h_sem_mean": 1.2, "a_q": 4})
        rows.append({"question_id": q, "model_key": "m", "spec_level": 1,
                     "fi_spec": spec1, "m_valid": 1, "f_mean": f1, "aufi_in": 0.5,
                     "fi_out_mean": 1.5, "h_sem_mean": 0.7, "a_q": 3})
    return pd.DataFrame(rows)


def test_gate_passes_on_expected_directions(tmp_path):
    p = tmp_path / "ok.parquet"
    _df().to_parquet(p, index=False)
    assert check_parquet(p) == 0


def test_gate_fails_when_fi_spec_not_increasing(tmp_path):
    p = tmp_path / "bad_spec.parquet"
    _df(spec0=1.5, spec1=1.5).to_parquet(p, index=False)
    assert check_parquet(p) == 1


def test_gate_fails_when_accuracy_drops(tmp_path):
    p = tmp_path / "bad_acc.parquet"
    _df(f0=0.7, f1=0.4).to_parquet(p, index=False)
    assert check_parquet(p) == 1


def test_gate_fails_on_missing_columns(tmp_path):
    p = tmp_path / "bad_cols.parquet"
    _df().drop(columns=["h_sem_mean"]).to_parquet(p, index=False)
    assert check_parquet(p) == 1
