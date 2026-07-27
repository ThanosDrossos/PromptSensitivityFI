"""Multi-head prompt-feedback probes (deliverable design 2026-07-28).

Four linear heads on ONE shared feature vector (the target model's TBG hidden
states at 50/75/100% depth, concatenated), one per metric axis:

  vagueness    P(prompt is ambiguous)        label: spec_level == 0 (free from
               the design)                    -> "this prompt is too vague"
  reliability  E[P(correct | prompt)]        label: f_graded per prompt
               (isotonic-recalibrated)        -> "predicted reliability: X%"
  dispersion   P(high output entropy)        label: h_sem_mean > gamma*
                                              -> "answers will likely vary"
  fragility    P(high rho_F)  [EXPERIMENTAL] label: rho_f > gamma*; per-prompt
               decodability is weak (0.55-0.65 AUROC) and the k=20 arm showed
               that is signal scarcity, not label noise -> shipped with an
               explicit experimental flag, never as a headline gauge.

Hyperparameters are FIXED from the probe sweeps (train_fi_probes nested CV:
logistic C=0.01, ridge alpha=1e5); training here does plain question-grouped
5-fold OOF for verification + isotonic calibration fitted ON the OOF scores
(leakage-free), then refits each head on all rows for the shipped bundle.

Every verification row carries the two standing controls: question-permuted
labels (expected ~0.5) and the prompt-length baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..scripts.dump_hidden_states import decode_vec
from ..scripts.train_fi_probes import (
    gamma_star,
    group_folds,
    permute_labels_by_question,
)

DEFAULT_LAYER_FRACS = (0.5, 0.75, 1.0)
_C, _ALPHA = 0.01, 1e5


# --------------------------------------------------------------------------- #
# Features + labels                                                           #
# --------------------------------------------------------------------------- #


def build_features(
    hs: pd.DataFrame, layer_fracs=DEFAULT_LAYER_FRACS
) -> tuple[np.ndarray, pd.DataFrame]:
    """Dump rows -> (X (n_prompts, dim*len(fracs)) float32, aligned meta).

    Layers are matched by layer_frac (as written by the dump) and concatenated
    in the given order; a prompt missing any requested layer is dropped.
    """
    hs = hs.copy()
    hs["question_id"] = hs["question_id"].astype(str)
    want = [round(f, 4) for f in layer_fracs]
    hs = hs[hs["layer_frac"].round(4).isin(want)]
    key = ["question_id", "spec_level", "paraphrase_idx"]
    blocks, meta_rows = [], []
    for k, grp in hs.sort_values(key + ["layer_frac"]).groupby(key, sort=True):
        if grp["layer_frac"].round(4).nunique() != len(want):
            continue
        vecs = [decode_vec(r.vec, r.dim).astype(np.float32)
                for _, r in grp.sort_values("layer_frac").iterrows()]
        blocks.append(np.concatenate(vecs))
        meta_rows.append({"question_id": k[0], "spec_level": int(k[1]),
                          "paraphrase_idx": int(k[2]),
                          "paraphrase": grp.paraphrase.iloc[0]})
    return np.stack(blocks), pd.DataFrame(meta_rows)


def head_labels(meta: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Per-prompt label frame for all four heads (NaN where undefined)."""
    m = metrics.copy()
    m["question_id"] = m["question_id"].astype(str)
    key = ["question_id", "spec_level"]
    out = meta.copy()
    out["vagueness"] = (out.spec_level == 0).astype(float)
    per = m[key + ["f_graded_per_paraphrase"]].dropna(
        subset=["f_graded_per_paraphrase"]).explode("f_graded_per_paraphrase")
    per["paraphrase_idx"] = per.groupby(key).cumcount()
    per["reliability"] = per.f_graded_per_paraphrase.astype(float)
    out = out.merge(per[key + ["paraphrase_idx", "reliability"]],
                    on=key + ["paraphrase_idx"], how="left")
    for src, dst in [("h_sem_mean", "dispersion"), ("rho_f", "fragility")]:
        out = out.merge(m[key + [src]].rename(columns={src: dst}), on=key, how="left")
    return out


# --------------------------------------------------------------------------- #
# Training + calibration                                                      #
# --------------------------------------------------------------------------- #


@dataclass
class TrainedHead:
    name: str
    kind: str                    # "binary" | "continuous"
    scaler_mean: np.ndarray
    scaler_sd: np.ndarray
    model: Any                   # LogisticRegression | Ridge (fitted on all rows)
    calibrator: Any              # IsotonicRegression score -> P(label) / E[label]
    threshold_label: float | None    # gamma* used to binarize (None for vagueness)
    verification: dict = field(default_factory=dict)
    experimental: bool = False

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xs = (X - self.scaler_mean) / self.scaler_sd
        raw = (self.model.decision_function(Xs) if self.kind == "binary"
               else self.model.predict(Xs))
        return np.clip(self.calibrator.predict(raw), 0.0, 1.0)


def train_head(
    name: str, X: np.ndarray, y_raw: np.ndarray, qids: pd.Series,
    *, binarize: bool, n_splits: int = 5, seed: int = 42,
    prompt_lengths: np.ndarray | None = None, experimental: bool = False,
    C: float = _C, alpha: float = _ALPHA,
) -> TrainedHead:
    """Grouped-OOF verification + isotonic calibration + full refit."""
    from scipy.stats import spearmanr
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import roc_auc_score

    keep = np.isfinite(y_raw)
    X, y_raw = X[keep], y_raw[keep]
    qids = qids[keep].reset_index(drop=True)
    thr = None
    if binarize and set(np.unique(y_raw)) - {0.0, 1.0}:
        thr = gamma_star(y_raw)
        y = (y_raw > thr).astype(int)
    else:
        y = y_raw.astype(int) if binarize else y_raw

    def make_model():
        # Defaults fixed from the train_fi_probes nested-CV sweeps on the real
        # ~11k-dim standardized features; pass dim-appropriate values elsewhere.
        return (LogisticRegression(C=C, max_iter=2000) if binarize
                else Ridge(alpha=alpha))

    oof = np.full(len(y), np.nan)
    for tr, te in group_folds(qids, n_splits, seed=seed):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        mdl = make_model()
        mdl.fit((X[tr] - mu) / sd, y[tr])
        Xte = (X[te] - mu) / sd
        oof[te] = mdl.decision_function(Xte) if binarize else mdl.predict(Xte)

    ver: dict[str, float] = {"n": int(len(y)), "n_questions": int(qids.nunique())}
    rng = np.random.default_rng(seed)
    if binarize:
        ver["auroc"] = float(roc_auc_score(y, oof))
        # Two null controls — each meaningful for a different label geometry:
        #  * permuted: shuffle label patterns BETWEEN questions. Meaningful for
        #    cell-level labels (dispersion/fragility); VACUOUS for a
        #    within-question label like vagueness, where every question carries
        #    the identical [L0=1, L1=0] pattern and permutation is a no-op
        #    (caught on the 2026-07-27 verification run: permuted == real).
        #  * flip: invert ALL labels of a random half of questions. Meaningful
        #    exactly in the within-question case; expected ~0.5.
        yperm = permute_labels_by_question(y.astype(float), qids).astype(int)
        if len(set(yperm)) > 1 and (yperm != y).any():
            ver["auroc_permuted"] = float(roc_auc_score(yperm, oof))
        uq = sorted(qids.unique())
        flip_q = set(rng.choice(uq, size=len(uq) // 2, replace=False))
        yflip = np.where(qids.isin(flip_q).to_numpy(), 1 - y, y)
        if len(set(yflip)) > 1:
            ver["auroc_flip_control"] = float(roc_auc_score(yflip, oof))
        if prompt_lengths is not None:
            ln = prompt_lengths[keep]
            ver["auroc_length_baseline"] = float(
                max(roc_auc_score(y, ln), roc_auc_score(y, -ln)))
    else:
        ver["spearman"] = float(spearmanr(y, oof).statistic)

    # Isotonic calibration on the leakage-free OOF scores. The SHIPPED
    # calibrator uses all OOF pairs; the REPORTED ECE is cross-fitted (2 folds
    # by question: fit on one half's OOF, score the other) — fitting and
    # evaluating isotonic on the same points gives ECE ~ 0 by construction
    # (the 2026-07-27 run reported a meaningless 0.000 everywhere).
    target = y.astype(float)
    eces = []
    for tr2, te2 in group_folds(qids, 2, seed=seed + 7):
        iso2 = IsotonicRegression(out_of_bounds="clip")
        iso2.fit(oof[tr2], target[tr2])
        cal2 = np.clip(iso2.predict(oof[te2]), 0, 1)
        t2 = target[te2]
        which = np.digitize(cal2, np.linspace(0, 1, 11)) - 1
        eces.append(np.nansum([
            abs(cal2[which == b].mean() - t2[which == b].mean()) * (which == b).mean()
            for b in range(10) if (which == b).any()
        ]))
    ver["ece"] = float(np.mean(eces))
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof, target)

    mu, sd = X.mean(0), X.std(0) + 1e-8
    final = make_model()
    final.fit((X - mu) / sd, y)
    return TrainedHead(name=name, kind="binary" if binarize else "continuous",
                       scaler_mean=mu, scaler_sd=sd, model=final, calibrator=iso,
                       threshold_label=thr, verification=ver,
                       experimental=experimental)


# --------------------------------------------------------------------------- #
# Bundle + feedback composition                                               #
# --------------------------------------------------------------------------- #


@dataclass
class FeedbackModel:
    model_key: str
    layer_fracs: tuple
    heads: dict[str, TrainedHead]
    meta: dict = field(default_factory=dict)

    def gauges(self, X: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({name: h.predict(X) for name, h in self.heads.items()})

    def save(self, path) -> None:
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "FeedbackModel":
        import joblib
        return joblib.load(path)


THRESHOLDS = {"vagueness": 0.65, "dispersion": 0.60, "fragility": 0.60}
RELIABILITY_BANDS = (0.35, 0.65)     # low < .35 <= medium < .65 <= high


def compose_feedback(g: dict[str, float]) -> list[str]:
    """Calibrated gauges -> user-facing messages (the deliverable strings)."""
    msgs: list[str] = []
    if g.get("vagueness", 0) >= THRESHOLDS["vagueness"]:
        msgs.append("⚠ Too vague: the prompt likely admits multiple readings — "
                    "state exactly which case/entity/time you mean.")
    elif g.get("vagueness", 0) >= 0.45:
        msgs.append("~ Possibly underspecified — consider naming the intended "
                    "interpretation explicitly.")
    r = g.get("reliability")
    if r is not None:
        lo, hi = RELIABILITY_BANDS
        band = "LOW" if r < lo else ("MEDIUM" if r < hi else "HIGH")
        msgs.append(f"Predicted answer reliability: {band} (≈{r*100:.0f}%).")
    if g.get("dispersion", 0) >= THRESHOLDS["dispersion"]:
        msgs.append("⚠ Unstable output expected: re-asking will likely yield "
                    "different answers.")
    if g.get("fragility", 0) >= THRESHOLDS["fragility"]:
        msgs.append("(experimental) Wording-sensitive: small rephrasings may "
                    "change the outcome.")
    if len(msgs) == 1 and r is not None and r >= RELIABILITY_BANDS[1]:
        msgs.append("Prompt looks specific and stable — no changes suggested.")
    return msgs
