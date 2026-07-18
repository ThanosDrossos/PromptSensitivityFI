"""FI probes: binarization threshold, feature assembly, label joins, and the
question-group leakage rule. Probe heads themselves are sklearn; we test our
wiring, not sklearn."""

from __future__ import annotations

import numpy as np
import pandas as pd

from prompt_sensitivity.scripts.dump_hidden_states import encode_vec
from prompt_sensitivity.scripts.train_fi_probes import (
    assemble_features,
    gamma_star,
    group_folds,
    join_labels,
    permute_labels_by_question,
)


def test_gamma_star_splits_bimodal_at_the_gap():
    v = np.array([0.0, 0.1, 0.2, 3.0, 3.1, 3.2])   # two tight modes, big gap
    g = gamma_star(v)
    assert 0.2 < g < 3.0
    # degenerate inputs don't crash
    assert gamma_star(np.array([1.0])) == 1.0
    assert gamma_star(np.array([2.0, 2.0])) == 2.0


def _hs_frame(n_q=3, n_para=2, layers=(7, 28), dim=4):
    rows = []
    for qi in range(n_q):
        for lvl in (0, 1):
            for p in range(n_para):
                for layer in layers:
                    vec = np.full(dim, qi * 100 + lvl * 10 + p, dtype=np.float32)
                    rows.append({"question_id": f"q{qi}", "spec_level": lvl,
                                 "model_key": "m", "paraphrase_idx": p,
                                 "paraphrase": f"q{qi}L{lvl}p{p}?",
                                 "context_mode": "uniform_evidence", "position": "tbg",
                                 "layer_idx": layer, "layer_frac": layer / 28,
                                 "dim": dim, "dtype": "float16", "vec": encode_vec(vec)})
    return pd.DataFrame(rows)


def test_assemble_features_layer_slice_aligned_with_meta():
    hs = _hs_frame()
    X, meta = assemble_features(hs, 7)
    assert X.shape == (3 * 2 * 2, 4)                    # q x level x paraphrase
    # row i's vector encodes its identity -> meta must line up exactly
    for i, r in meta.iterrows():
        qi = int(r.question_id[1:])
        assert X[i, 0] == qi * 100 + r.spec_level * 10 + r.paraphrase_idx


def test_join_labels_cell_broadcast_and_per_paraphrase_explode():
    hs = _hs_frame()
    _, meta = assemble_features(hs, 7)
    metrics = pd.DataFrame([
        {"question_id": "q0", "spec_level": 0, "model_key": "m",
         "aufi_in": 1.5, "f_graded_per_paraphrase": [0.1, 0.9]},
        {"question_id": "q0", "spec_level": 1, "model_key": "m",
         "aufi_in": 0.5, "f_graded_per_paraphrase": [1.0, 1.0]},
    ])
    y = join_labels(meta, metrics, "aufi_in")
    m0 = (meta.question_id == "q0").to_numpy()
    assert np.isnan(y[~m0]).all()                        # unlabeled cells -> NaN
    assert set(y[m0]) == {1.5, 0.5}                      # broadcast to both paraphrases
    yg = join_labels(meta, metrics, "f_graded_per_paraphrase")
    r = meta[(meta.question_id == "q0") & (meta.spec_level == 0)]
    assert yg[r.index[r.paraphrase_idx == 0][0]] == 0.1  # exploded by paraphrase_idx
    assert yg[r.index[r.paraphrase_idx == 1][0]] == 0.9


def test_group_folds_never_split_a_question():
    qids = pd.Series([f"q{i}" for i in range(10) for _ in range(4)])
    for tr, te in group_folds(qids, 5):
        assert set(qids.iloc[tr]) & set(qids.iloc[te]) == set()
        assert len(tr) + len(te) == len(qids)


def test_permute_labels_by_question_destroys_association_keeps_structure():
    qids = pd.Series(["a"] * 3 + ["b"] * 3 + ["c"] * 3)
    y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype=float)
    yp = permute_labels_by_question(y, qids, seed=0)
    # each question still has ONE consistent label value...
    for g in ("a", "b", "c"):
        vals = set(yp[(qids == g).to_numpy()])
        assert len(vals) == 1
    # ...drawn from the original label set (a permutation, not new values)
    assert set(yp) == {1.0, 2.0, 3.0}
