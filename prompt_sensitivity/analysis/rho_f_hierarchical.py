"""R2 — partially-pooled rho_F. A hierarchical beta-binomial replacement for the
per-cell method-of-moments ICC.

WHY (review 2026-08-06 §2.3, R2). `metrics.sensitivity_v2.rho_f` is a one-way
ANOVA ICC estimated independently in each cell. It has two defects that together
produce the paper's headline independence result:

  1. It is UNDEFINED when a cell has no variance (every sample of every paraphrase
     correct, or all wrong): MS_B = MS_W = 0. Those cells are exactly the accuracy
     extremes, so complete-case analysis deletes 34-55% of the data NON-RANDOMLY
     and range-restricts accuracy. Imputing 0 instead moves rho_F ~ accuracy from
     +.03/+.22/+.00 to +.26/+.48/+.41 -- i.e. the orthogonality claim is an
     artifact of the missingness rule.
  2. It is NOISY. The disjoint-paraphrase split-half reliability (mean of 200
     random 5/5 splits) is only .215/.350/.405 (qwen/llama/mistral), Spearman-
     Brown .35/.52/.58 -- not the .81/.92/.95 reported, which compares k=10
     against a k=20 set that CONTAINS it.

THE MODEL. For cell c with N paraphrases, each answered correctly y_i of k times:

    y_i  ~  Binomial(k, p_i)
    p_i  ~  Beta(mu_c, rho_c)        [mean mu_c, intra-class correlation rho_c]

parameterised by s_c = (1 - rho_c)/rho_c, alpha = mu_c*s_c, beta = (1-mu_c)*s_c.
For binary outcomes the beta-binomial overdispersion parameter **is** the ICC:
rho_c = 0 recovers the plain binomial (phrasing irrelevant, all wobble is decoding
noise); rho_c = 1 means each paraphrase is deterministically right or wrong. So
this estimates the SAME estimand as `rho_f`, with two differences that matter:

  * mu_c is integrated out under a flat prior, and rho_c gets a Beta(a, b) prior
    whose hyper-parameters are estimated from ALL cells by type-II maximum
    likelihood (empirical Bayes). Cells therefore borrow strength from each other.
  * The posterior mean of rho_c is DEFINED EVERYWHERE, including all-correct and
    all-wrong cells, where it shrinks to what the rest of the data says a cell
    like that looks like. Coverage becomes 100% and the missingness rule stops
    driving the result.

We also return the posterior SD, so per-cell uncertainty is reported rather than
implied, and `sigma2_between` (the ABSOLUTE noise-corrected variance component),
which is comparable across models in a way the share is not: rho_F is monotone
decreasing in decoding noise, and mean within-paraphrase p(1-p) is the exact
inverse of the rho_F model ordering (qwen .0247 < mistral .0378 < llama .0598), so
a reviewer will otherwise read the model ranking as a decoding-entropy ranking.

CONTRACT: `metrics/` is untouched. This lives in `analysis/` and consumes the same
persisted `f_graded_per_paraphrase` column.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp
from scipy.stats import beta as beta_dist

# Grid bounds. rho is bounded away from 0 and 1 because s = (1-rho)/rho diverges
# at 0 and the Beta parameters vanish at 1; the caps are far outside the range
# any real cell resolves at k=10, N=10.
_RHO_LO, _RHO_HI = 1e-3, 1.0 - 1e-3
_MU_LO, _MU_HI = 1e-3, 1.0 - 1e-3


def _betabinom_logpmf_table(k: int, rho_grid: np.ndarray, mu_grid: np.ndarray) -> np.ndarray:
    """(k+1, R, M) table of log P(y | k, mu, rho).

    Only k+1 distinct outcomes exist, so the whole grid costs (k+1)*R*M
    evaluations rather than one per paraphrase.
    """
    s = (1.0 - rho_grid) / rho_grid                       # (R,)
    a = mu_grid[None, :] * s[:, None]                     # (R, M)
    b = (1.0 - mu_grid)[None, :] * s[:, None]             # (R, M)
    y = np.arange(k + 1)[:, None, None]                   # (k+1, 1, 1)
    a = a[None, :, :]
    b = b[None, :, :]
    # log C(k, y) + log B(y+a, k-y+b) - log B(a, b)
    log_choose = gammaln(k + 1) - gammaln(y + 1) - gammaln(k - y + 1)
    log_beta_num = gammaln(y + a) + gammaln(k - y + b) - gammaln(k + a + b)
    log_beta_den = gammaln(a) + gammaln(b) - gammaln(a + b)
    return log_choose + log_beta_num - log_beta_den


def _cell_counts_matrix(counts: list[np.ndarray], k: int) -> np.ndarray:
    """(C, k+1) histogram of per-paraphrase success counts, one row per cell."""
    out = np.zeros((len(counts), k + 1), dtype=np.float64)
    for c, ys in enumerate(counts):
        for y in ys:
            out[c, int(y)] += 1.0
    return out


@dataclass(frozen=True)
class HierFit:
    """Result of one empirical-Bayes fit over a set of cells."""

    rho_mean: np.ndarray          # (C,) posterior mean of rho_c — DEFINED EVERYWHERE
    rho_sd: np.ndarray            # (C,) posterior SD
    rho_grid: np.ndarray          # (R,)
    log_lik: np.ndarray           # (C, R) per-cell marginal log-likelihood over rho
    prior_a: float                # fitted Beta prior on rho
    prior_b: float
    mu_a: float                   # fitted Beta prior on mu (removes the degenerate-cell artifact)
    mu_b: float
    n_cells: int

    def posterior(self) -> np.ndarray:
        """(C, R) normalised posterior over the rho grid."""
        logp = self.log_lik + beta_dist.logpdf(self.rho_grid, self.prior_a, self.prior_b)[None, :]
        logp -= logsumexp(logp, axis=1, keepdims=True)
        return np.exp(logp)

    def sample(self, n_draws: int, *, seed: int = 42) -> np.ndarray:
        """(n_draws, C) draws of rho_c from each cell's posterior.

        Use these for MULTIPLE IMPUTATION rather than correlating the posterior
        means. Correlating shrunken point estimates understates uncertainty and
        can bias the correlation toward 0 (all cells pulled to a common value);
        re-running an analysis on each draw and pooling propagates the per-cell
        uncertainty properly, which is what the coverage problem demands. Cells
        that carry no information contribute noise rather than a fabricated value.
        """
        rng = np.random.default_rng(seed)
        post = self.posterior()                          # (C, R)
        cdf = np.cumsum(post, axis=1)
        cdf /= cdf[:, -1:][:, :]
        u = rng.random((n_draws, post.shape[0]))
        idx = np.array([
            np.searchsorted(cdf[c], u[:, c], side="left") for c in range(post.shape[0])
        ]).T                                             # (n_draws, C)
        idx = np.clip(idx, 0, len(self.rho_grid) - 1)
        return self.rho_grid[idx]


def fit_hierarchical_rho_f(
    per_paraphrase_rates: list[list[float] | np.ndarray | None],
    k: int,
    *,
    n_rho: int = 80,
    n_mu: int = 40,
) -> HierFit:
    """Empirical-Bayes beta-binomial rho_F over a collection of cells.

    `per_paraphrase_rates[c]` is the cell's `f_graded_per_paraphrase` (each entry a
    multiple of 1/k). Cells that are None/empty get the prior mean, which is the
    honest answer when a cell carries no information.
    """
    if k < 2:
        raise ValueError("need k >= 2 samples per paraphrase")
    rho_grid = np.linspace(_RHO_LO, _RHO_HI, n_rho)
    mu_grid = np.linspace(_MU_LO, _MU_HI, n_mu)

    counts: list[np.ndarray] = []
    for rates in per_paraphrase_rates:
        if rates is None or len(rates) == 0:
            counts.append(np.zeros(0, dtype=int))
            continue
        arr = np.asarray(rates, dtype=float)
        y = np.rint(arr * k).astype(int)
        if np.any(y < 0) or np.any(y > k):
            raise ValueError("per-paraphrase rates must lie in [0, 1]")
        counts.append(y)

    tab = _betabinom_logpmf_table(k, rho_grid, mu_grid)      # (k+1, R, M)
    hist = _cell_counts_matrix(counts, k)                     # (C, k+1)

    # (C, R*M) = (C, k+1) @ (k+1, R*M): the cell log-likelihood at every (rho, mu).
    flat = tab.reshape(k + 1, -1)
    ll = (hist @ flat).reshape(len(counts), len(rho_grid), len(mu_grid))

    # Empirical Bayes over FOUR hyper-parameters: a Beta prior on rho AND a Beta
    # prior on mu, both fitted to all cells jointly.
    #
    # The mu prior is not cosmetic. Under a FLAT mu prior a degenerate cell (all
    # paraphrases always wrong) has marginal likelihood 1/(N+1) at rho=1 versus
    # 1/(Nk+1) at rho=0 -- a 9.2x pull toward rho=1 at N=k=10 that is pure
    # artifact of integrating mu over regions where all-zeros is unlikely under
    # low rho. Fitting mu's prior to the actual accuracy distribution removes it,
    # so degenerate cells fall back to what comparable cells look like instead of
    # being scored as maximally phrasing-sensitive.
    informative = np.array([len(c) > 0 for c in counts])
    ll_inf = ll[informative]

    def _log_evidence(theta: np.ndarray) -> np.ndarray:
        a_r, b_r, a_m, b_m = np.exp(theta)
        lp_rho = beta_dist.logpdf(rho_grid, a_r, b_r)
        lp_mu = beta_dist.logpdf(mu_grid, a_m, b_m)
        if not (np.all(np.isfinite(lp_rho)) and np.all(np.isfinite(lp_mu))):
            return None
        # integrate mu (weighted by its prior), then rho
        over_mu = logsumexp(ll_inf + lp_mu[None, None, :], axis=2)
        return logsumexp(over_mu + lp_rho[None, :], axis=1)

    def _neg_evidence(theta: np.ndarray) -> float:
        if not np.all(np.isfinite(theta)) or np.any(theta > 20.0):
            return 1e12
        ev = _log_evidence(theta)
        if ev is None or not np.all(np.isfinite(ev)):
            return 1e12
        return float(-np.sum(ev))

    best = minimize(_neg_evidence, x0=np.zeros(4), method="Nelder-Mead",
                    options={"maxiter": 500, "xatol": 1e-2, "fatol": 1e-2})
    prior_a, prior_b, mu_a, mu_b = (float(x) for x in np.exp(best.x))

    log_prior_mu = beta_dist.logpdf(mu_grid, mu_a, mu_b)
    log_lik = logsumexp(ll + log_prior_mu[None, None, :], axis=2)   # (C, R)

    log_prior = beta_dist.logpdf(rho_grid, prior_a, prior_b)
    logp = log_lik + log_prior[None, :]
    logp -= logsumexp(logp, axis=1, keepdims=True)
    post = np.exp(logp)

    rho_mean = post @ rho_grid
    rho_var = post @ (rho_grid ** 2) - rho_mean ** 2
    rho_sd = np.sqrt(np.maximum(rho_var, 0.0))

    return HierFit(
        rho_mean=rho_mean, rho_sd=rho_sd, rho_grid=rho_grid, log_lik=log_lik,
        prior_a=prior_a, prior_b=prior_b, mu_a=mu_a, mu_b=mu_b, n_cells=len(counts),
    )


def sigma2_between(rates: list[float] | np.ndarray | None, k: int) -> float:
    """ABSOLUTE noise-corrected between-paraphrase variance component,
    sigma2_B = (MS_B - MS_W)/k, clamped at 0.

    Report this ALONGSIDE the share. rho_F is a ratio normalised by a
    noise-dependent denominator, so a model that decodes more randomly gets a
    smaller rho_F for the same absolute phrasing effect; sigma2_B does not have
    that property and therefore settles the "is the model ranking just decoding
    entropy?" objection. (Verified 2026-08-06: it gives the SAME ordering,
    qwen .067 > mistral .039 > llama .022, so the objection fails -- but only
    because we can show it.)
    """
    if rates is None or len(rates) < 2:
        return float("nan")
    p = np.asarray(rates, dtype=float)
    n = len(p)
    mean = p.mean()
    msb = k * float(((p - mean) ** 2).sum()) / (n - 1)
    msw = float((k * p * (1.0 - p)).sum()) / (n * (k - 1))
    return float(max(0.0, (msb - msw) / k))
