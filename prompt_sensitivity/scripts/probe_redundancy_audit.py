"""Is the underspecification (FI_spec) probe head redundant with the three axes?

Supervisor's challenge (2026-08-11): the measurement model claims THREE axes, but
the probe suite ships FOUR heads. If three suffice, the fourth must be redundant.

Three tests, all on identical features (hidden state, last prompt token, layer
0.5), identical question-grouped folds:

  T1  DIRECTION. Train each head, compare weight vectors by |cosine|. If the
      underspecification direction is the competence direction, |cos| -> 1.
  T2  RECONSTRUCTION (the decisive one). Can the three axis heads' out-of-fold
      predictions reproduce the underspecification label? Fit a logistic model
      on {accuracy-hat, H_sem-hat, rho_F-hat} and compare its AUROC with the
      direct hidden-state head. If the direct head adds nothing, the fourth head
      is redundant and should be dropped.
  T3  SUBSTITUTION. Single best axis head used directly as an underspecification
      detector (best orientation), i.e. the cheapest possible replacement.

    uv run python -m prompt_sensitivity.scripts.probe_redundancy_audit
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from ..config import load_config

_MODELS = ["qwen_2_5_7b", "llama_3_1_8b", "mistral_7b_v03"]
_LAYER = 0.5
_SEED = 42


def _fit_weights(X, y):
    m = LogisticRegression(C=0.01, max_iter=2000)
    m.fit(X, y)
    return m.coef_.ravel()


def _oof_logistic(X, y, groups, n_splits=5):
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        m = LogisticRegression(C=0.01, max_iter=2000)
        m.fit(X[tr], y[tr])
        oof[te] = m.decision_function(X[te])
    return oof


def _auc(y, s):
    """AUROC in its better orientation (a baseline is allowed its sign)."""
    a = roc_auc_score(y, s)
    return max(a, 1 - a)


def load_features(config, model_key):
    """Hidden states at layer 0.5 joined to the cell-level axis targets."""
    root = config.repo_root()
    h = pd.read_parquet(root / f"data/hidden_states_{model_key}.parquet")
    h = h[np.isclose(h["layer_frac"], _LAYER)].reset_index(drop=True)
    v = pd.read_parquet(root / f"data/specificity_v3_{model_key}.parquet")

    from ..analysis.rho_f_hierarchical import fit_hierarchical_rho_f
    cells = [list(x) if x is not None else None for x in v["f_graded_per_paraphrase"]]
    k = int(v["n_samples_per_prompt"].iloc[0])
    v = v.assign(rho_f_hier=fit_hierarchical_rho_f(cells, k).rho_mean)

    keep = ["question_id", "spec_level", "f_graded_mean", "h_sem_mean", "rho_f_hier"]
    d = h.merge(v[keep], on=["question_id", "spec_level"], how="inner")
    dt = np.dtype(d["dtype"].iloc[0])          # vectors are raw bytes + declared dtype
    X = np.vstack([np.frombuffer(b, dtype=dt).astype(np.float32) for b in d["vec"]])
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    return d, X


def main() -> int:
    config = load_config()
    L = ["# Probe-head redundancy audit (supervisor feedback, 2026-08-11)", ""]
    L.append("Question: the measurement model has three axes, the probe suite has "
             "four heads. Is the fourth (underspecification / FI_spec) redundant?")
    L.append("")
    L.append("All heads: same features (hidden state, last prompt token, layer 0.5), "
             "same question-grouped 5-fold CV, logistic C = 0.01.")
    L.append("")

    rows_cos, rows_rec = [], []
    for mk in _MODELS:
        logger.info("model {}", mk)
        d, X = load_features(config, mk)
        groups = d["question_id"].to_numpy()

        # binary targets: the dial, and each axis split at its median
        y_spec = (d["spec_level"] == 0).astype(int).to_numpy()      # 1 = underspecified
        tgt = {
            "competence": d["f_graded_mean"].to_numpy(),
            "dispersion": d["h_sem_mean"].to_numpy(),
            "sensitivity": d["rho_f_hier"].to_numpy(),
        }
        y_axis = {k: (v > np.median(v)).astype(int) for k, v in tgt.items()}

        # ---- T1 direction ------------------------------------------------
        w_spec = _fit_weights(X, y_spec)
        cos = {}
        for k, y in y_axis.items():
            w = _fit_weights(X, y)
            cos[k] = float(abs(np.dot(w_spec, w) /
                               (np.linalg.norm(w_spec) * np.linalg.norm(w))))
        rows_cos.append({"model": mk, **cos})

        # ---- T2/T3 reconstruction + substitution ---------------------------
        oof_spec = _oof_logistic(X, y_spec, groups)
        auc_direct = _auc(y_spec, oof_spec)
        oof_axis = {k: _oof_logistic(X, y, groups) for k, y in y_axis.items()}
        Z = np.column_stack([oof_axis[k] for k in ("competence", "dispersion", "sensitivity")])
        oof_rec = _oof_logistic(Z, y_spec, groups)
        auc_rec = _auc(y_spec, oof_rec)
        subs = {k: _auc(y_spec, oof_axis[k]) for k in oof_axis}

        # T4 (construct level, capacity-matched): the TRUE measured axis values.
        # If underspecification is a function of the three axes, their measured
        # values -- not noisy probe estimates -- must predict it.
        Zt = np.column_stack([tgt["competence"], tgt["dispersion"], tgt["sensitivity"]])
        Zt = (Zt - Zt.mean(0)) / (Zt.std(0) + 1e-9)
        auc_true = _auc(y_spec, _oof_logistic(Zt, y_spec, groups))

        # T5: how much of the direct head is just prompt length?
        lens = d["paraphrase"].str.len().to_numpy(dtype=float).reshape(-1, 1)
        auc_len = _auc(y_spec, _oof_logistic(
            (lens - lens.mean()) / (lens.std() + 1e-9), y_spec, groups))

        rows_rec.append({"model": mk, "direct": auc_direct, "reconstructed": auc_rec,
                         "true_axes": auc_true, "length": auc_len,
                         **{f"sub_{k}": v for k, v in subs.items()}})

    L.append("## T1 - Is it the same direction in representation space?")
    L.append("")
    L.append("|cosine| between the underspecification weight vector and each axis head:")
    L.append("")
    L.append("| model | vs competence | vs dispersion | vs sensitivity |")
    L.append("|---|---|---|---|")
    for r in rows_cos:
        L.append(f"| {r['model']} | {r['competence']:.3f} | {r['dispersion']:.3f} | "
                 f"{r['sensitivity']:.3f} |")
    L.append("")

    L.append("## T2/T3 - Can the three axis heads replace it?")
    L.append("")
    L.append("| model | direct head (AUROC) | reconstructed from 3 axis heads | "
             "from TRUE axis values | best single axis head | length only |")
    L.append("|---|---|---|---|---|---|")
    for r in rows_rec:
        best = max(r["sub_competence"], r["sub_dispersion"], r["sub_sensitivity"])
        L.append(f"| {r['model']} | **{r['direct']:.3f}** | {r['reconstructed']:.3f} | "
                 f"{r['true_axes']:.3f} | {best:.3f} | {r['length']:.3f} |")
    L.append("")
    L.append("| model | sub: competence | sub: dispersion | sub: sensitivity |")
    L.append("|---|---|---|---|")
    for r in rows_rec:
        L.append(f"| {r['model']} | {r['sub_competence']:.3f} | "
                 f"{r['sub_dispersion']:.3f} | {r['sub_sensitivity']:.3f} |")
    L.append("")
    gap = np.mean([r["direct"] - r["reconstructed"] for r in rows_rec])
    L.append(f"Mean advantage of the direct head over the best reconstruction: "
             f"**{gap:+.3f} AUROC**.")
    L.append("")

    out = config.repo_root() / "data/probe_redundancy_audit.md"
    out.write_text("\n".join(L), encoding="utf-8")
    try:
        print("\n".join(L))
    except UnicodeEncodeError:
        print(f"(console cannot render; see {out})")
    logger.info("wrote {}", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
