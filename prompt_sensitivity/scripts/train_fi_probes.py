"""FI probes — predict FI labels from single-prompt TBG hidden states.

The capstone (FI_PROBES_PLAN.md): transfer Semantic Entropy Probes (Kossen et
al. 2024, arXiv:2406.15927) to Functional Information. Features = the eval
model's OWN hidden state at the last templated prompt token (TBG, captured by
scripts/dump_hidden_states.py); labels = FI metrics from a specificity run.

Probe family — LINEAR ONLY, three complementary heads per (layer, target):
  * logistic  L2 logistic regression on the SEP-binarized label (gamma* =
              within-class-variance-minimizing threshold), scored by AUROC.
              SEP-faithful headline (their exact probe: L2 + LBFGS).
  * ridge     ridge regression on the CONTINUOUS label (bits), scored by
              Spearman. Keeps the information binarization throws away —
              viable because our labels are already real-valued.
  * massmean  difference-of-class-means direction (Marks & Tegmark 2024),
              scored by AUROC. Near-zero variance; the strongest choice when
              n is tiny and a lower bound on what the two trained heads add.
No MLP: with ~100 cells / 50 question groups, probe capacity would measure
memorization, not the representation (Hewitt & Liang 2019; Belinkov 2022).

Leakage rule: ALL splits are GroupKFold by question_id — paraphrases of one
question (which share cell-level labels) never straddle train/test.
Controls: question-level label permutation (expected AUROC ~0.5) and a
prompt-length baseline, so a real probe result must beat both.

    python -m prompt_sensitivity.scripts.train_fi_probes \
        --features data/hidden_states_qwen_2_5_7b.parquet \
        --labels data/specificity_v2_metrics.parquet \
        --targets aufi_in,h_sem_mean
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from ..config import load_config
from ..logging_setup import configure_logging
from .dump_hidden_states import decode_vec, validate_hidden_dump

_C_GRID = (1e-3, 1e-2, 1e-1, 1.0)
_RIDGE_ALPHA_GRID = (1e2, 1e3, 1e4, 1e5)


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)                                                   #
# --------------------------------------------------------------------------- #


def gamma_star(values: np.ndarray) -> float:
    """SEP's binarization threshold: the split minimizing the sample-weighted
    within-class variance (Kossen et al. §3.2; Otsu's criterion). Candidate
    thresholds are midpoints between consecutive sorted unique values."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    uniq = np.unique(v)
    if uniq.size < 2:
        return float(uniq[0]) if uniq.size else 0.0
    best_t, best_score = None, np.inf
    for t in (uniq[:-1] + uniq[1:]) / 2.0:
        lo, hi = v[v <= t], v[v > t]
        score = lo.size * lo.var() + hi.size * hi.var()
        if score < best_score:
            best_t, best_score = float(t), float(score)
    return best_t


def assemble_features(
    hs: pd.DataFrame, layer_idx: int
) -> tuple[np.ndarray, pd.DataFrame]:
    """One layer's dump rows -> (X (n, dim) float32, meta df aligned row-wise).

    Meta carries question_id / spec_level / model_key / paraphrase_idx /
    paraphrase for label joins and grouping. Deterministic order.
    """
    sub = hs[hs["layer_idx"] == layer_idx].sort_values(
        ["question_id", "spec_level", "paraphrase_idx"]
    ).reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"no rows for layer_idx={layer_idx}")
    X = np.stack([
        decode_vec(r.vec, r.dim).astype(np.float32) for r in sub.itertuples()
    ])
    meta = sub[["question_id", "spec_level", "model_key", "paraphrase_idx", "paraphrase"]]
    return X, meta.copy()


def join_labels(meta: pd.DataFrame, metrics: pd.DataFrame, target: str) -> np.ndarray:
    """Cell-level label broadcast onto the cell's paraphrase rows.

    Per-paraphrase targets (f_graded_per_paraphrase, v3+) are exploded by
    paraphrase_idx; cell scalars (aufi_in, h_sem_mean, f_mean, ...) repeat
    across the cell's rows. Rows without a label come back NaN (caller drops).
    """
    key = ["question_id", "spec_level", "model_key"]
    m = metrics.copy()
    m["question_id"] = m["question_id"].astype(str)
    if target == "f_graded_per_paraphrase":
        m = m[key + [target]].dropna(subset=[target])
        m = m.explode(target).reset_index(drop=True)
        m["paraphrase_idx"] = m.groupby(key).cumcount()
        m[target] = m[target].astype(float)
        joined = meta.merge(m, on=key + ["paraphrase_idx"], how="left")
    else:
        joined = meta.merge(m[key + [target]], on=key, how="left")
    return joined[target].to_numpy(dtype=float)


def group_folds(question_ids: pd.Series, n_splits: int, seed: int = 42):
    """GroupKFold by question — the leakage rule. Shuffled group assignment
    (GroupKFold itself is deterministic-unshuffled; we shuffle groups first)."""
    from sklearn.model_selection import GroupKFold

    rng = np.random.default_rng(seed)
    groups = question_ids.astype(str).to_numpy()
    uniq = np.array(sorted(set(groups)))
    perm = {g: i for i, g in enumerate(rng.permutation(uniq))}
    mapped = np.array([perm[g] for g in groups])
    gkf = GroupKFold(n_splits=min(n_splits, uniq.size))
    yield from gkf.split(np.zeros(len(groups)), groups=mapped)


def permute_labels_by_question(
    y: np.ndarray, question_ids: pd.Series, seed: int = 0
) -> np.ndarray:
    """Control task: shuffle labels BETWEEN questions, keeping each question's
    rows consistent (label structure preserved, association destroyed)."""
    rng = np.random.default_rng(seed)
    q = question_ids.astype(str).to_numpy()
    uniq = sorted(set(q))
    src = dict(zip(uniq, rng.permutation(uniq)))
    first_rows = {g: np.where(q == g)[0] for g in uniq}
    out = np.empty_like(y)
    for g in uniq:
        donor_rows = first_rows[src[g]]
        mine = first_rows[g]
        # donor question's label pattern, tiled/cut to my row count
        pattern = y[donor_rows]
        out[mine] = np.resize(pattern, mine.size)
    return out


# --------------------------------------------------------------------------- #
# Probe heads                                                                  #
# --------------------------------------------------------------------------- #


@dataclass
class ProbeResult:
    layer_idx: int
    target: str
    head: str
    score_name: str          # auroc | spearman
    score: float
    n: int
    n_questions: int
    detail: str = ""


def _standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = train.mean(0), train.std(0) + 1e-8
    return (train - mu) / sd, (test - mu) / sd


def eval_logistic(X, y_bin, qids, *, n_splits=5, seed=42) -> tuple[float, str]:
    """Nested CV: outer GroupKFold pooled-OOF AUROC; inner GroupKFold picks C."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    oof = np.full(len(y_bin), np.nan)
    picked = []
    for tr, te in group_folds(qids, n_splits, seed=seed):
        best_c, best_auc = _C_GRID[0], -np.inf
        for c in _C_GRID:
            aucs = []
            for itr, ite in group_folds(qids.iloc[tr].reset_index(drop=True), 3, seed=seed + 1):
                Xtr, Xte = _standardize(X[tr][itr], X[tr][ite])
                ytr, yte = y_bin[tr][itr], y_bin[tr][ite]
                if len(set(ytr)) < 2 or len(set(yte)) < 2:
                    continue
                clf = LogisticRegression(C=c, max_iter=2000, solver="lbfgs")
                clf.fit(Xtr, ytr)
                aucs.append(roc_auc_score(yte, clf.decision_function(Xte)))
            if aucs and np.mean(aucs) > best_auc:
                best_c, best_auc = c, float(np.mean(aucs))
        Xtr, Xte = _standardize(X[tr], X[te])
        if len(set(y_bin[tr])) < 2:
            continue
        clf = LogisticRegression(C=best_c, max_iter=2000, solver="lbfgs")
        clf.fit(Xtr, y_bin[tr])
        oof[te] = clf.decision_function(Xte)
        picked.append(best_c)
    mask = np.isfinite(oof)
    from sklearn.metrics import roc_auc_score as _auc
    score = float(_auc(y_bin[mask], oof[mask])) if len(set(y_bin[mask])) > 1 else np.nan
    return score, f"C={sorted(set(picked))}"


def eval_ridge(X, y, qids, *, n_splits=5, seed=42) -> tuple[float, str]:
    """Nested CV ridge on the continuous label; pooled-OOF Spearman."""
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge

    oof = np.full(len(y), np.nan)
    picked = []
    for tr, te in group_folds(qids, n_splits, seed=seed):
        best_a, best_r = _RIDGE_ALPHA_GRID[0], -np.inf
        for a in _RIDGE_ALPHA_GRID:
            rs = []
            for itr, ite in group_folds(qids.iloc[tr].reset_index(drop=True), 3, seed=seed + 1):
                Xtr, Xte = _standardize(X[tr][itr], X[tr][ite])
                reg = Ridge(alpha=a)
                reg.fit(Xtr, y[tr][itr])
                r = spearmanr(y[tr][ite], reg.predict(Xte)).statistic
                if np.isfinite(r):
                    rs.append(r)
            if rs and np.mean(rs) > best_r:
                best_a, best_r = a, float(np.mean(rs))
        Xtr, Xte = _standardize(X[tr], X[te])
        reg = Ridge(alpha=best_a)
        reg.fit(Xtr, y[tr])
        oof[te] = reg.predict(Xte)
        picked.append(best_a)
    mask = np.isfinite(oof)
    score = float(spearmanr(y[mask], oof[mask]).statistic)
    return score, f"alpha={sorted(set(picked))}"


def eval_massmean(X, y_bin, qids, *, n_splits=5, seed=42) -> tuple[float, str]:
    """Difference-of-class-means direction (Marks & Tegmark 2024), OOF AUROC."""
    from sklearn.metrics import roc_auc_score

    oof = np.full(len(y_bin), np.nan)
    for tr, te in group_folds(qids, n_splits, seed=seed):
        Xtr, Xte = _standardize(X[tr], X[te])
        ytr = y_bin[tr]
        if len(set(ytr)) < 2:
            continue
        w = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
        oof[te] = Xte @ w
    mask = np.isfinite(oof)
    score = float(roc_auc_score(y_bin[mask], oof[mask])) if len(set(y_bin[mask])) > 1 else np.nan
    return score, ""


# --------------------------------------------------------------------------- #
# Driver                                                                       #
# --------------------------------------------------------------------------- #


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default="data/hidden_states_qwen_2_5_7b.parquet")
    ap.add_argument("--labels", default="data/specificity_v2_metrics.parquet")
    ap.add_argument("--targets", default="aufi_in,h_sem_mean",
                    help="metric columns; f_graded_per_paraphrase for per-prompt (v3+)")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--out", default=None,
                    help="results parquet (default data/probe_results_<features-stem>.parquet)")
    args = ap.parse_args()

    configure_logging("train_fi_probes")
    config = load_config()
    root = config.repo_root()
    hs_path = root / args.features
    hs = pd.read_parquet(hs_path)
    problems = validate_hidden_dump(hs)
    if problems:
        logger.error("feature dump invalid: {}", problems)
        return 1
    hs["question_id"] = hs["question_id"].astype(str)
    metrics = pd.read_parquet(root / args.labels)
    layers = sorted(hs["layer_idx"].unique())
    logger.info("features: {} rows, layers {}, {} questions",
                len(hs), layers, hs.question_id.nunique())

    results: list[ProbeResult] = []
    baselines_done: set[str] = set()
    for layer in layers:
        X, meta = assemble_features(hs, layer)
        for target in [t.strip() for t in args.targets.split(",") if t.strip()]:
            y = join_labels(meta, metrics, target)
            keep = np.isfinite(y)
            if keep.sum() < 40:
                logger.warning("layer {} target {}: only {} labeled rows — skipped",
                               layer, target, int(keep.sum()))
                continue
            Xk, yk = X[keep], y[keep]
            qk = meta.loc[keep, "question_id"].reset_index(drop=True)
            g = gamma_star(yk)
            ybin = (yk > g).astype(int)
            nq = qk.nunique()
            common = dict(layer_idx=int(layer), target=target, n=int(keep.sum()),
                          n_questions=int(nq))

            auc, det = eval_logistic(Xk, ybin, qk, n_splits=args.n_splits)
            results.append(ProbeResult(head="logistic", score_name="auroc",
                                       score=auc, detail=f"gamma*={g:.3f} {det}", **common))
            rho, det = eval_ridge(Xk, yk, qk, n_splits=args.n_splits)
            results.append(ProbeResult(head="ridge", score_name="spearman",
                                       score=rho, detail=det, **common))
            mm, _ = eval_massmean(Xk, ybin, qk, n_splits=args.n_splits)
            results.append(ProbeResult(head="massmean", score_name="auroc",
                                       score=mm, detail=f"gamma*={g:.3f}", **common))
            logger.info("layer {:>2} {:<14} logistic AUROC={:.3f} ridge rho={:.3f} massmean AUROC={:.3f}",
                        layer, target, auc, rho, mm)

            if target not in baselines_done:
                baselines_done.add(target)
                # control 1: question-permuted labels (expected ~0.5)
                yperm = permute_labels_by_question(ybin.astype(float), qk).astype(int)
                if len(set(yperm)) > 1:
                    ctrl, _ = eval_massmean(Xk, yperm, qk, n_splits=args.n_splits)
                    results.append(ProbeResult(head="control_permuted", score_name="auroc",
                                               score=ctrl, **common))
                # control 2: prompt-length baseline
                length = meta.loc[keep, "paraphrase"].str.len().to_numpy(dtype=float)[:, None]
                lb, _ = eval_massmean(length, ybin, qk, n_splits=args.n_splits)
                results.append(ProbeResult(head="baseline_length", score_name="auroc",
                                           score=lb, **common))

    out = pd.DataFrame([r.__dict__ for r in results])
    out_path = root / (args.out or f"data/probe_results_{hs_path.stem}.parquet")
    out.to_parquet(out_path, index=False)
    print()
    print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
