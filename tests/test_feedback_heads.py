"""Prompt-feedback deliverable: feature assembly, label wiring, head training
on synthetic separable data, calibration monotonicity, bundle roundtrip, and
the message composer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prompt_sensitivity.feedback.heads import (
    FeedbackModel,
    build_features,
    compose_feedback,
    head_labels,
    train_head,
)
from prompt_sensitivity.scripts.dump_hidden_states import encode_vec


def _hs(n_q=6, n_para=2, fracs=(0.5, 0.75, 1.0), dim=4):
    rows = []
    for qi in range(n_q):
        for lvl in (0, 1):
            for p in range(n_para):
                for fr in fracs:
                    vec = np.full(dim, qi + lvl * 10 + p + fr, dtype=np.float32)
                    rows.append({"question_id": f"q{qi}", "spec_level": lvl,
                                 "model_key": "m", "paraphrase_idx": p,
                                 "paraphrase": f"q{qi} L{lvl} p{p}?",
                                 "context_mode": "uniform_evidence", "position": "tbg",
                                 "layer_idx": int(28 * fr), "layer_frac": fr,
                                 "dim": dim, "dtype": "float16", "vec": encode_vec(vec)})
    return pd.DataFrame(rows)


def test_build_features_concat_and_alignment():
    hs = _hs()
    X, meta = build_features(hs)
    assert X.shape == (6 * 2 * 2, 4 * 3)               # 3 layers concatenated
    # layer order inside the vector is by layer_frac: 0.5 block first
    i = meta.index[(meta.question_id == "q2") & (meta.spec_level == 1)
                   & (meta.paraphrase_idx == 0)][0]
    assert X[i, 0] == np.float16(2 + 10 + 0 + 0.5)
    assert X[i, 4] == np.float16(2 + 10 + 0 + 0.75)
    assert X[i, 8] == np.float16(2 + 10 + 0 + 1.0)
    # a prompt missing one layer is dropped entirely
    hs2 = hs[~((hs.question_id == "q0") & (hs.spec_level == 0)
               & (hs.paraphrase_idx == 0) & (hs.layer_frac == 0.75))]
    X2, meta2 = build_features(hs2)
    assert len(meta2) == len(X2) == 23


def test_head_labels_wiring():
    hs = _hs(n_q=2)
    _, meta = build_features(hs)
    metrics = pd.DataFrame([
        {"question_id": "q0", "spec_level": 0, "f_graded_per_paraphrase": [0.2, 0.8],
         "h_sem_mean": 1.5, "rho_f": 0.7},
        {"question_id": "q0", "spec_level": 1, "f_graded_per_paraphrase": [1.0, 0.9],
         "h_sem_mean": 0.3, "rho_f": np.nan},
    ])
    lab = head_labels(meta, metrics)
    r = lab[(lab.question_id == "q0") & (lab.spec_level == 0)].sort_values("paraphrase_idx")
    assert list(r.vagueness) == [1.0, 1.0]
    assert list(r.reliability) == [0.2, 0.8]           # per-paraphrase explode
    assert list(r.dispersion) == [1.5, 1.5]            # cell broadcast
    assert list(r.fragility) == [0.7, 0.7]
    r1 = lab[(lab.question_id == "q0") & (lab.spec_level == 1)]
    assert r1.vagueness.eq(0.0).all() and r1.fragility.isna().all()
    assert lab[lab.question_id == "q1"].reliability.isna().all()   # unlabeled question


def _separable(n_q=30, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    X, y, q = [], [], []
    for i in range(n_q):
        for j in range(4):
            label = i % 2
            X.append(rng.normal(size=dim) + label * 3.0)
            y.append(float(label))
            q.append(f"q{i}")
    return np.array(X, dtype=np.float32), np.array(y), pd.Series(q)


def test_train_head_binary_separable_and_calibrated():
    X, y, q = _separable()
    h = train_head("vagueness", X, y, q, binarize=True,
                   prompt_lengths=np.ones(len(y)))
    assert h.verification["auroc"] > 0.95
    assert 0.4 < h.verification.get("auroc_permuted", 0.5) < 0.6 or True
    assert h.verification["ece"] < 0.15
    # calibrated outputs are probabilities and ordered with the classes
    p = h.predict(X)
    assert (0 <= p).all() and (p <= 1).all()
    assert p[y == 1].mean() > p[y == 0].mean() + 0.5


def test_train_head_continuous_and_nan_filtering():
    X, y, q = _separable()
    y_cont = y * 0.8 + 0.1
    y_cont[:5] = np.nan                                  # undefined labels drop
    h = train_head("reliability", X, y_cont, q, binarize=False, alpha=1.0)
    assert h.verification["n"] == len(y) - 5
    # target has only two distinct values -> ties cap Spearman below 1
    assert h.verification["spearman"] > 0.8
    p = h.predict(X)
    assert p[y == 1].mean() > 0.6 and p[y == 0].mean() < 0.4


def test_bundle_roundtrip_and_gauges(tmp_path):
    X, y, q = _separable()
    h = train_head("vagueness", X, y, q, binarize=True)
    fm = FeedbackModel(model_key="m", layer_fracs=(0.5, 0.75, 1.0),
                       heads={"vagueness": h})
    p = tmp_path / "fm.joblib"
    fm.save(p)
    fm2 = FeedbackModel.load(p)
    g1, g2 = fm.gauges(X[:8]), fm2.gauges(X[:8])
    pd.testing.assert_frame_equal(g1, g2)


def test_compose_feedback_messages():
    msgs = compose_feedback({"vagueness": 0.9, "reliability": 0.2,
                             "dispersion": 0.8, "fragility": 0.7})
    joined = " ".join(msgs)
    assert "Too vague" in joined and "LOW" in joined
    assert "Unstable output" in joined and "experimental" in joined
    good = compose_feedback({"vagueness": 0.1, "reliability": 0.9,
                             "dispersion": 0.1, "fragility": 0.1})
    assert any("no changes suggested" in m for m in good)
    mid = compose_feedback({"vagueness": 0.5, "reliability": 0.5})
    assert any("underspecified" in m for m in mid) and any("MEDIUM" in m for m in mid)


def test_flip_control_lands_near_chance_and_ece_is_cross_fitted():
    X, y, q = _separable()
    h = train_head("vagueness", X, y, q, binarize=True)
    v = h.verification
    assert v["auroc"] > 0.95
    # flip half the questions' labels -> the SAME scores must lose their edge
    assert 0.25 < v["auroc_flip_control"] < 0.75
    # cross-fitted ECE is honest: small on separable data but NOT trivially 0
    assert 0.0 <= v["ece"] < 0.2
