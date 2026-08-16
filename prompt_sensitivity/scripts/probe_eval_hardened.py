"""R7 — hardened probe evaluation. Replaces the evaluation the review broke.

WHAT WAS WRONG (review 2026-08-06 §3.6-3.9, V10):
  * `control_permuted` / `baseline_length` existed ONLY at the shallowest layer;
    the headline layers had no control at all, and the single permutation drawn
    ranged 0.372-0.583 across targets (null SD ~.08 — comparable to the margins).
  * The best layer was selected post hoc over 4 layers x 2 head types.
  * The OOD length baseline was reported in its sign-flipped orientation (.457;
    honest value .543), and no text-surface baseline existed at all.
  * No operating-point analysis (the shipped threshold flags most prompts).

WHAT THIS DOES (per model, from the local parquets — no generation, no cluster):
  IN-DISTRIBUTION (149 questions, grouped by question):
    1. Per-layer AUROC for massmean and logistic heads at EVERY dumped layer,
       each with a 200-draw question-permutation null for massmean (closed form,
       so a full distribution is affordable everywhere) and a 25-draw null for
       the expensive logistic at its selected layer. Vagueness additionally gets
       the flip control (within-question labels make permutation vacuous —
       heads.py documents why).
    2. Honest headline: layer selection INSIDE nested CV (outer 5-fold by
       question; inner 3-fold picks the layer), so the reported AUROC never sees
       its own test fold, not even through the layer choice.
    3. Text baselines under the SAME grouped-OOF protocol: prompt length,
       TF-IDF word 1-2gram, TF-IDF char 3-5gram, first-word-rate.
  OUT-OF-DISTRIBUTION (1,852 annotator-labelled questions, frozen everything):
    4. The shipped vagueness head vs frozen text baselines (trained on the same
       L0-vs-L1 universe the head saw, no holdout labels), length in its BETTER
       orientation, PR-AUC, and the confusion matrix at the shipped threshold.

NOT COVERED HERE (needs generation => cluster): the "ask an LLM whether the
question is ambiguous" baseline. Flagged in the report as the one missing
comparison.

Outputs: data/probe_eval_hardened_{model}.parquet + data/probe_eval_hardened.md.
CONTRACT: read-only over parquets/joblibs; `metrics/` and `feedback/` untouched.
"""

from __future__ import annotations

import argparse
import glob
import sys

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score

from ..config import load_config
from ..feedback.heads import FeedbackModel, build_features, head_labels
from ..scripts.train_fi_probes import gamma_star, group_folds, permute_labels_by_question

_MODELS = ["qwen_2_5_7b", "llama_3_1_8b", "mistral_7b_v03"]
_TARGETS = {  # name -> (binary?, permutation null meaningful?)
    "vagueness": (True, False),  # within-question label -> use flip control
    "dispersion": (True, True),
    "fragility": (True, True),
    "reliability": (False, True),
}
_SHIPPED_THRESHOLD = 0.65  # heads.THRESHOLDS["vagueness"]


# --------------------------------------------------------------------------- #
# fast heads                                                                   #
# --------------------------------------------------------------------------- #


def massmean_oof(X, y, qids, *, n_splits=5, seed=42) -> np.ndarray:
    """Question-grouped OOF decision scores of the mass-mean probe.

    Closed form (difference of class means on standardized features), which is
    what makes a 200-draw permutation null affordable at every layer.
    """
    oof = np.full(len(y), np.nan)
    for tr, te in group_folds(qids, n_splits, seed=seed):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        Xtr = (X[tr] - mu) / sd
        w = Xtr[y[tr] == 1].mean(0) - Xtr[y[tr] == 0].mean(0)
        oof[te] = ((X[te] - mu) / sd) @ w
    return oof


def logistic_oof(X, y, qids, *, n_splits=5, seed=42, C=0.01) -> np.ndarray:
    oof = np.full(len(y), np.nan, dtype=float)
    for tr, te in group_folds(qids, n_splits, seed=seed):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        mdl = LogisticRegression(C=C, max_iter=2000)
        mdl.fit((X[tr] - mu) / sd, y[tr])
        oof[te] = mdl.decision_function((X[te] - mu) / sd)
    return oof


def ridge_oof(X, y, qids, *, n_splits=5, seed=42, alpha=1e5) -> np.ndarray:
    oof = np.full(len(y), np.nan, dtype=float)
    for tr, te in group_folds(qids, n_splits, seed=seed):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        mdl = Ridge(alpha=alpha)
        mdl.fit((X[tr] - mu) / sd, y[tr])
        oof[te] = mdl.predict((X[te] - mu) / sd)
    return oof


def permutation_null(head_fn, X, y, qids, *, n_perm, seed=42, score="auroc") -> np.ndarray:
    """Null AUROC/Spearman distribution: permute label BLOCKS between questions,
    refit the head each time. This is the distribution V10 said was missing."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_perm):
        yp = permute_labels_by_question(y.astype(float), qids, seed=int(rng.integers(1 << 31)))
        if score == "auroc":
            ypb = yp.astype(int)
            if len(np.unique(ypb)) < 2:
                continue
            s = head_fn(X, ypb, qids, seed=seed + i)
            out.append(roc_auc_score(ypb, s))
        else:
            s = head_fn(X, yp, qids, seed=seed + i)
            out.append(stats.spearmanr(yp, s).statistic)
    return np.asarray(out)


def flip_control(oof, y, qids, *, n_draws=200, seed=42) -> np.ndarray:
    """Null for within-question labels: invert all labels of a random half of
    questions, rescore the SAME oof (the flip breaks any real signal)."""
    rng = np.random.default_rng(seed)
    uq = np.array(sorted(qids.unique()))
    out = []
    for _ in range(n_draws):
        flip = set(rng.choice(uq, size=len(uq) // 2, replace=False))
        yf = np.where(qids.isin(flip).to_numpy(), 1 - y, y)
        if len(np.unique(yf)) > 1:
            out.append(roc_auc_score(yf, oof))
    return np.asarray(out)


# --------------------------------------------------------------------------- #
# baselines                                                                    #
# --------------------------------------------------------------------------- #


def text_baseline_oof(texts, y, qids, kind, *, n_splits=5, seed=42) -> np.ndarray:
    """TF-IDF / first-word / length baselines under the SAME grouped protocol."""
    texts = pd.Series(texts).astype(str)
    oof = np.full(len(y), np.nan)
    if kind == "length":
        ln = texts.str.len().to_numpy(dtype=float)
        return ln  # orientation resolved at scoring time via max(auc, 1-auc)
    for tr, te in group_folds(qids, n_splits, seed=seed):
        if kind == "first_word":
            fw = texts.str.strip().str.split().str[0].str.lower()
            rates = pd.Series(y[tr], index=fw.iloc[tr]).groupby(level=0).mean()
            oof[te] = fw.iloc[te].map(rates).fillna(float(y[tr].mean())).to_numpy()
            continue
        vec = (
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
            if kind == "tfidf_word"
            else TfidfVectorizer(
                analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True
            )
        )
        Xtr = vec.fit_transform(texts.iloc[tr])
        mdl = LogisticRegression(max_iter=2000)
        mdl.fit(Xtr, y[tr])
        oof[te] = mdl.decision_function(vec.transform(texts.iloc[te]))
    return oof


def _auc_best_orientation(y, s) -> float:
    a = roc_auc_score(y, s)
    return float(max(a, 1 - a))


# --------------------------------------------------------------------------- #
# in-distribution evaluation                                                   #
# --------------------------------------------------------------------------- #


def eval_model_in_distribution(
    config, model_key, *, n_perm_mass=200, n_perm_logistic=25, seed=42
) -> pd.DataFrame:
    hs = pd.read_parquet(config.repo_root() / f"data/hidden_states_{model_key}.parquet")
    metrics = pd.read_parquet(config.repo_root() / f"data/specificity_v3_{model_key}.parquet")
    layer_fracs = sorted(hs["layer_frac"].round(4).unique())
    rows = []

    # per-layer features once
    feats: dict[float, tuple[np.ndarray, pd.DataFrame]] = {}
    for lf in layer_fracs:
        feats[lf] = build_features(hs, layer_fracs=(lf,))

    for target, (binary, perm_ok) in _TARGETS.items():
        # labels aligned per layer (meta identical across layers, but be exact)
        per_layer_oof: dict[float, np.ndarray] = {}
        aligned: dict[float, tuple] = {}
        for lf in layer_fracs:
            X, meta = feats[lf]
            lab = head_labels(meta, metrics)
            y_raw = lab[target].to_numpy(dtype=float)
            keep = np.isfinite(y_raw)
            Xk, yk = X[keep], y_raw[keep]
            qk = lab.question_id[keep].reset_index(drop=True)
            tk = lab.paraphrase[keep].reset_index(drop=True)
            thr = None
            if binary and set(np.unique(yk)) - {0.0, 1.0}:
                thr = gamma_star(yk)
                yk = (yk > thr).astype(int)
            elif binary:
                yk = yk.astype(int)
            aligned[lf] = (Xk, yk, qk, tk)

            if binary:
                oof_m = massmean_oof(Xk, yk, qk, seed=seed)
                oof_l = logistic_oof(Xk, yk, qk, seed=seed)
                null_m = (
                    permutation_null(
                        lambda X_, y_, q_, seed: roc_null_scores_massmean(X_, y_, q_, seed),
                        Xk,
                        yk,
                        qk,
                        n_perm=n_perm_mass,
                        seed=seed,
                    )
                    if perm_ok
                    else flip_control(oof_m, yk, qk, seed=seed)
                )
                real_m = roc_auc_score(yk, oof_m)
                real_l = roc_auc_score(yk, oof_l)
                p_m = float((np.sum(null_m >= real_m) + 1) / (len(null_m) + 1))
                rows.append(
                    dict(
                        model=model_key,
                        target=target,
                        layer_frac=lf,
                        head="massmean",
                        auroc=real_m,
                        null_mean=float(null_m.mean()),
                        null_sd=float(null_m.std()),
                        null_p=p_m,
                        null_kind="permutation" if perm_ok else "flip",
                        n=len(yk),
                    )
                )
                rows.append(
                    dict(
                        model=model_key,
                        target=target,
                        layer_frac=lf,
                        head="logistic",
                        auroc=real_l,
                        null_mean=np.nan,
                        null_sd=np.nan,
                        null_p=np.nan,
                        null_kind="(null at selected layer only)",
                        n=len(yk),
                    )
                )
                per_layer_oof[lf] = oof_l
            else:
                oof_r = ridge_oof(Xk, yk, qk, seed=seed)
                real = float(stats.spearmanr(yk, oof_r).statistic)
                null_r = permutation_null(
                    lambda X_, y_, q_, seed: ridge_oof(X_, y_, q_, seed=seed),
                    Xk,
                    yk,
                    qk,
                    n_perm=max(10, n_perm_logistic),
                    seed=seed,
                    score="spearman",
                )
                rows.append(
                    dict(
                        model=model_key,
                        target=target,
                        layer_frac=lf,
                        head="ridge",
                        auroc=real,
                        null_mean=float(null_r.mean()),
                        null_sd=float(null_r.std()),
                        null_p=float((np.sum(null_r >= real) + 1) / (len(null_r) + 1)),
                        null_kind="permutation(spearman)",
                        n=len(yk),
                    )
                )
                per_layer_oof[lf] = oof_r

        # honest headline: layer selection INSIDE nested CV (logistic/ridge)
        Xs = {lf: aligned[lf][0] for lf in layer_fracs}
        _, y0, q0, t0 = aligned[layer_fracs[0]]
        nested = nested_layer_selection(Xs, y0, q0, binary=binary, seed=seed)
        sel_layer = nested["selected_mode"]
        rows.append(
            dict(
                model=model_key,
                target=target,
                layer_frac=np.nan,
                head="nested_cv",
                auroc=nested["score"],
                null_mean=np.nan,
                null_sd=np.nan,
                null_p=np.nan,
                null_kind=f"layer selected in inner CV (mode {sel_layer})",
                n=len(y0),
            )
        )
        # expensive null at the nested-selected layer
        if binary:
            Xsel, ysel, qsel, _ = aligned[sel_layer]
            if perm_ok:
                null_l = permutation_null(
                    lambda X_, y_, q_, seed: logistic_oof(X_, y_, q_, seed=seed, n_splits=3),
                    Xsel,
                    ysel,
                    qsel,
                    n_perm=n_perm_logistic,
                    seed=seed,
                )
            else:
                null_l = flip_control(per_layer_oof[sel_layer], ysel, qsel, seed=seed)
            real_l = roc_auc_score(ysel, per_layer_oof[sel_layer])
            rows.append(
                dict(
                    model=model_key,
                    target=target,
                    layer_frac=sel_layer,
                    head="logistic+null",
                    auroc=real_l,
                    null_mean=float(null_l.mean()),
                    null_sd=float(null_l.std()),
                    null_p=float((np.sum(null_l >= real_l) + 1) / (len(null_l) + 1)),
                    null_kind="permutation" if perm_ok else "flip",
                    n=len(ysel),
                )
            )

        # text baselines under the SAME protocol (target-independent features,
        # target-dependent fit)
        _, yb, qb, tb = aligned[layer_fracs[0]]
        if binary:
            for kind in ("length", "tfidf_word", "tfidf_char", "first_word"):
                s = text_baseline_oof(tb, yb, qb, kind, seed=seed)
                a = _auc_best_orientation(yb, s) if kind == "length" else roc_auc_score(yb, s)
                rows.append(
                    dict(
                        model=model_key,
                        target=target,
                        layer_frac=np.nan,
                        head=f"baseline_{kind}",
                        auroc=float(a),
                        null_mean=np.nan,
                        null_sd=np.nan,
                        null_p=np.nan,
                        null_kind="",
                        n=len(yb),
                    )
                )
        logger.info("{} / {}: done", model_key, target)
    return pd.DataFrame(rows)


def roc_null_scores_massmean(X, y, qids, seed):
    return massmean_oof(X, y, qids, seed=seed)


def nested_layer_selection(Xs: dict, y, qids, *, binary, seed=42) -> dict:
    """Outer 5-fold grouped CV; inner 3-fold picks the layer; returns the outer
    OOF score — an AUROC that never saw its own test fold, even via the layer."""
    layer_fracs = sorted(Xs)
    oof = np.full(len(y), np.nan)
    chosen = []
    for tr, te in group_folds(qids, 5, seed=seed):
        best_lf, best = None, -np.inf
        q_tr = qids.iloc[tr].reset_index(drop=True)
        for lf in layer_fracs:
            Xin = Xs[lf][tr]
            s = (
                massmean_oof(Xin, y[tr], q_tr, n_splits=3, seed=seed)
                if binary
                else ridge_oof(Xin, y[tr], q_tr, n_splits=3, seed=seed)
            )
            score = roc_auc_score(y[tr], s) if binary else stats.spearmanr(y[tr], s).statistic
            if score > best:
                best, best_lf = score, lf
        chosen.append(best_lf)
        mu, sd = Xs[best_lf][tr].mean(0), Xs[best_lf][tr].std(0) + 1e-8
        if binary:
            mdl = LogisticRegression(C=0.01, max_iter=2000)
            mdl.fit((Xs[best_lf][tr] - mu) / sd, y[tr])
            oof[te] = mdl.decision_function((Xs[best_lf][te] - mu) / sd)
        else:
            mdl = Ridge(alpha=1e5)
            mdl.fit((Xs[best_lf][tr] - mu) / sd, y[tr])
            oof[te] = mdl.predict((Xs[best_lf][te] - mu) / sd)
    score = float(roc_auc_score(y, oof)) if binary else float(stats.spearmanr(y, oof).statistic)
    mode = max(set(chosen), key=chosen.count)
    return {"score": score, "selected_mode": mode, "chosen_per_fold": chosen}


# --------------------------------------------------------------------------- #
# OOD (frozen protocol)                                                        #
# --------------------------------------------------------------------------- #


def eval_ood(config, model_key) -> dict:
    """Frozen vagueness head + frozen baselines on the annotator-labelled holdout."""
    hold = pd.read_parquet(config.repo_root() / f"data/vagueness_holdout_{model_key}.parquet")
    # The dump contains all 2,002 annotator-labelled questions INCLUDING the 150
    # the heads trained on (their L0 prompts are literal training examples, all
    # labelled ambiguous). Zero-shot means they must be excluded — same rule as
    # eval_vagueness_holdout.py; without it n=2,002 and the AUROC is inflated.
    seen: set[str] = set()
    for f in glob.glob(str(config.repo_root() / "data/specificity_v3_*.parquet")):
        seen |= set(pd.read_parquet(f, columns=["question_id"])["question_id"].astype(str))
    hold = hold[~hold["question_id"].astype(str).isin(seen)].reset_index(drop=True)
    bundle = FeedbackModel.load(config.repo_root() / f"data/feedback_model_{model_key}.joblib")
    # the holdout dump already carries spec_level/paraphrase_idx keys, so the
    # standard feature builder applies unchanged
    X, meta = build_features(hold, layer_fracs=bundle.layer_fracs)
    # label per question (one prompt per question in the holdout dump)
    lab = hold.drop_duplicates(["question_id"]).set_index("question_id")["ambiguous"].astype(int)
    meta["y"] = meta.question_id.map(lab)
    y = meta.y.to_numpy(dtype=int)
    scores = bundle.heads["vagueness"].predict(X)

    out = {"model": model_key, "n": len(y), "n_ambiguous": int(y.sum())}
    out["auroc_head"] = float(roc_auc_score(y, scores))
    out["prauc_head"] = float(average_precision_score(y, scores))
    out["prevalence"] = float(y.mean())
    # confusion at the shipped threshold — the operating point users actually get
    pred = (scores >= _SHIPPED_THRESHOLD).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    out.update(
        dict(
            threshold=_SHIPPED_THRESHOLD,
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            flagged_frac=float(pred.mean()),
            precision=float(tp / max(tp + fp, 1)),
            recall=float(tp / max(tp + fn, 1)),
        )
    )
    # frozen baselines: length (better orientation) + TF-IDF/first-word trained
    # on the SAME L0-vs-L1 universe the head trained on (no holdout labels)
    texts = meta.paraphrase.astype(str)
    out["auroc_length"] = _auc_best_orientation(y, texts.str.len().to_numpy(dtype=float))
    uni = pd.read_parquet(config.repo_root() / "data/paraphrases_ambigqa.parquet")
    uni = uni[uni.outcome.isin(["accepted", "singleton_fallback"])]
    ytr = (uni.spec_level == 0).astype(int).to_numpy()
    for kind, vec in [
        ("tfidf_word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        (
            "tfidf_char",
            TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True),
        ),
    ]:
        Xtr = vec.fit_transform(uni.text.astype(str))
        mdl = LogisticRegression(max_iter=2000)
        mdl.fit(Xtr, ytr)
        out[f"auroc_{kind}_frozen"] = float(
            roc_auc_score(y, mdl.decision_function(vec.transform(texts)))
        )
    fw_tr = uni.text.astype(str).str.strip().str.split().str[0].str.lower()
    rates = pd.Series(ytr, index=fw_tr).groupby(level=0).mean()
    fw_te = texts.str.strip().str.split().str[0].str.lower()
    out["auroc_first_word_frozen"] = float(
        roc_auc_score(y, fw_te.map(rates).fillna(float(ytr.mean())).to_numpy())
    )
    return out


# --------------------------------------------------------------------------- #
# report                                                                       #
# --------------------------------------------------------------------------- #


def render(indist: pd.DataFrame, ood: list[dict]) -> str:
    L = ["# R7 — hardened probe evaluation", ""]
    L.append(
        "Controls at EVERY layer (null distributions, not single draws); layer selection inside"
    )
    L.append(
        "nested CV; text baselines under matched protocols; operating-point analysis on the OOD"
    )
    L.append("holdout. Missing by design (needs generation → cluster): the ask-an-LLM baseline.")
    L.append("")
    for m in indist.model.unique():
        L.append(f"## {m} — in-distribution (grouped by question)")
        L.append("")
        for target in indist[indist.model == m].target.unique():
            sub = indist[(indist.model == m) & (indist.target == target)]
            L.append(f"### {target}")
            L.append("")
            L.append("| head | layer | score | null mean ± sd | p |")
            L.append("|---|---|---|---|---|")
            for _, r in sub.iterrows():
                # r["head"], not r.head — attribute access resolves to the
                # pandas Series METHOD and prints its repr into the table.
                nm = (
                    "—"
                    if np.isnan(r["null_mean"])
                    else f"{r['null_mean']:.3f} ± {r['null_sd']:.3f}"
                )
                pv = "—" if np.isnan(r["null_p"]) else f"{r['null_p']:.3f}"
                lf = "—" if np.isnan(r["layer_frac"]) else f"{r['layer_frac']:g}"
                L.append(f"| {r['head']} | {lf} | {r['auroc']:.3f} | {nm} | {pv} |")
            L.append("")
    L.append("## OOD — frozen heads on the annotator-labelled holdout")
    L.append("")
    L.append(
        "| model | AUROC head | PR-AUC (prev.) | length | TF-IDF w/c (frozen) | first-word (frozen) |"
    )
    L.append("|---|---|---|---|---|---|")
    for o in ood:
        L.append(
            f"| {o['model']} | {o['auroc_head']:.3f} | {o['prauc_head']:.3f} "
            f"({o['prevalence']:.2f}) | {o['auroc_length']:.3f} | "
            f"{o['auroc_tfidf_word_frozen']:.3f} / {o['auroc_tfidf_char_frozen']:.3f} | "
            f"{o['auroc_first_word_frozen']:.3f} |"
        )
    L.append("")
    L.append("### Operating point at the shipped threshold (0.65)")
    L.append("")
    L.append("| model | flagged | precision | recall | TP/FP/FN/TN |")
    L.append("|---|---|---|---|---|")
    for o in ood:
        L.append(
            f"| {o['model']} | {o['flagged_frac']:.1%} | {o['precision']:.3f} | "
            f"{o['recall']:.3f} | {o['tp']}/{o['fp']}/{o['fn']}/{o['tn']} |"
        )
    L.append("")
    L.append(
        "**Claim discipline:** the head's OOD advantage is over *frozen* baselines "
        "(zero-shot transfer / label efficiency). With in-domain labels, bag-of-words "
        "matches it — never claim detection quality beyond that. The fragility head's "
        "null table above is the honest version of the axis-2 probe result."
    )
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=_MODELS)
    ap.add_argument("--n-perm-mass", type=int, default=200)
    ap.add_argument("--n-perm-logistic", type=int, default=25)
    args = ap.parse_args()
    config = load_config()
    frames, ood = [], []
    for m in args.models:
        df = eval_model_in_distribution(
            config, m, n_perm_mass=args.n_perm_mass, n_perm_logistic=args.n_perm_logistic
        )
        df.to_parquet(config.repo_root() / f"data/probe_eval_hardened_{m}.parquet", index=False)
        frames.append(df)
        ood.append(eval_ood(config, m))
        logger.info("{}: in-distribution + OOD done", m)
    md = render(pd.concat(frames), ood)
    out = config.repo_root() / "data/probe_eval_hardened.md"
    out.write_text(md, encoding="utf-8")
    logger.info("wrote {}", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
