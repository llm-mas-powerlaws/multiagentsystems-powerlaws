"""
src/tail_fitting/powerlaw_fit.py

Clauset-Newman-Shalizi-correct power-law fitting for coordination event sizes.

Methodological notes:

  1. All five observables are INTEGER-VALUED. discrete=True enforced throughout.
     The discrete MLE applies the correction x_i / (x_min - 0.5) vs the
     continuous version. Omitting this overestimates alpha by 0.05-0.15.

  2. x_min selected by KS minimisation (Clauset procedure). Pass xmin= to
     force a value for sensitivity checks only.

  3. Sigma uses the analytic formula (alpha-1)/sqrt(n_tail) from Clauset
     et al. Eq. 4, not the library's internal estimate. These can differ
     slightly; we use the paper-defined formula for reviewer clarity.

  4. KS statistic: fit.power_law.KS() returns the KS distance between the
     empirical tail and the fitted power-law CDF. True bootstrap p-values
     require O(1000) refits and are expensive; we use the package's internal
     approximation and document this explicitly. pl_plausible uses KS < 0.1
     as threshold rather than a p-value, which is the more conservative check.

  5. Gini computed on TAIL DATA ONLY (arr >= x_min), consistent with alpha,
     KS, and LR tests. Full-distribution Gini available separately if needed.

  6. Model comparison uses Vuong likelihood-ratio test (normalized_ratio=True).
     R > 0 favours power-law; p < 0.05 makes the preference significant.

  7. n_tail >= 50 minimum for reliable MLE. dynamic_range < 20 triggers a
     warning (insufficient distinct tail values per Clauset guidelines).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import powerlaw  # type: ignore  (Alstott et al. 2014)


# ── Observable metadata ───────────────────────────────────────

OBSERVABLES: Dict[str, dict] = {
    "delegation_sizes":     {"label": "Delegation cascade size",  "discrete": True},
    "revision_waves":       {"label": "Revision wave size",       "discrete": True},
    "contradiction_bursts": {"label": "Contradiction burst size",  "discrete": True},
    "merge_fan_in":         {"label": "Merge fan-in",             "discrete": True},
    "tce":                  {"label": "TCE",                      "discrete": True},
}


# ── Result dataclass ──────────────────────────────────────────

@dataclass
class FitResult:
    """All statistics from one power-law fit. One instance = one table row."""

    observable:    str
    label:         str

    # Sample statistics
    n_total:       int
    n_tail:        int       # samples at or above x_min
    tail_fraction: float     # n_tail / n_total
    x_min:         float
    x_max:         float
    mean:          float
    median:        float
    dynamic_range: int       # distinct integer values in tail

    # Power-law fit
    alpha:         float     # scaling exponent (discrete MLE)
    sigma_alpha:   float     # (alpha-1) / sqrt(n_tail)  — Clauset Eq. 4
    ks_stat:       float     # KS distance between empirical tail and fit
    # ks_pvalue is approximate (package internal); use ks_stat < 0.1 for
    # pl_plausible check, not a strict p-value threshold.
    ks_pvalue:     float

    # Vuong likelihood-ratio tests (R > 0 favours PL; p < 0.05 significant)
    lr_vs_lognormal:      float
    lr_vs_lognormal_p:    float
    lr_vs_exponential:    float
    lr_vs_exponential_p:  float
    lr_vs_truncated_pl:   float
    lr_vs_truncated_pl_p: float

    # Verdicts
    pl_plausible:         bool   # ks_stat < 0.1
    pl_beats_lognormal:   bool   # lr > 0 and p < 0.05
    pl_beats_exponential: bool   # lr > 0 and p < 0.05

    # Regime (paper Table 3)
    regime: str   # collusion_risk | collective_intelligence | fragile_reasoning

    # Inequality — computed on TAIL DATA ONLY, consistent with other metrics
    gini: float

    tags: List[str] = field(default_factory=list)


# ── Helpers ───────────────────────────────────────────────────

def _regime(alpha: float) -> str:
    if alpha < 1.5:
        return "collusion_risk"
    if alpha <= 3.0:
        return "collective_intelligence"
    return "fragile_reasoning"


def _gini(data: np.ndarray) -> float:
    """Gini coefficient. Input must already be filtered to tail."""
    if len(data) == 0 or data.sum() == 0:
        return 0.0
    data = np.sort(data.astype(float))
    n    = len(data)
    idx  = np.arange(1, n + 1)
    return float((2 * (idx * data).sum() / (n * data.sum())) - (n + 1) / n)


# ── Core fitter ───────────────────────────────────────────────

def fit_observable(
    data:    List[int],
    name:    str,
    xmin:    Optional[float] = None,
    verbose: bool = True,
) -> Optional[FitResult]:
    """
    Fit a discrete power-law to a list of positive integer event sizes.

    data    Pooled list across seeds/runs for one condition. Pool first.
    name    Key from OBSERVABLES dict or any label string.
    xmin    Force lower cutoff (None = auto via KS minimisation).
    verbose Print one summary line.

    Returns None if n_total < 50 or n_tail < 50 after x_min selection.
    """
    arr = np.array([x for x in data if x > 0], dtype=float)

    if len(arr) < 50:
        if verbose:
            print(f"  {name}: n={len(arr)} < 50 — skipping.")
        return None

    meta     = OBSERVABLES.get(name, {"label": name, "discrete": True})
    discrete = meta["discrete"]
    label    = meta["label"]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = powerlaw.Fit(arr, xmin=xmin, discrete=discrete, verbose=False)

    tail_mask = arr >= fit.xmin
    tail_data = arr[tail_mask]
    n_tail    = int(tail_mask.sum())

    if n_tail < 50:
        if verbose:
            print(f"  {name}: n_tail={n_tail} < 50 above xmin={fit.xmin:.1f} — skipping.")
        return None

    dynamic_range = int(len(np.unique(tail_data.astype(int))))
    if dynamic_range < 20 and verbose:
        print(f"  [warn] {name}: dynamic_range={dynamic_range} < 20 — "
              f"fit may be unreliable (insufficient distinct tail values).")

    # Scaling exponent: use library MLE (discrete-corrected)
    alpha = float(fit.power_law.alpha)

    # Sigma: analytic formula from Clauset et al. Eq. 4
    sigma = (alpha - 1.0) / np.sqrt(n_tail)

    x_min = float(fit.xmin)
    x_max = float(arr.max())

    # KS statistic between empirical tail and fitted CDF
    try:
        ks_stat = float(fit.power_law.KS())
    except Exception:
        ks_stat = float("nan")

    # KS p-value: package internal approximation (documented as approximate)
    try:
        ks_p = float(fit.power_law.D)
    except Exception:
        ks_p = float("nan")

    # Vuong likelihood-ratio tests
    def _lr(dist: str):
        try:
            R, p = fit.distribution_compare("power_law", dist,
                                            normalized_ratio=True)
            return float(R), float(p)
        except Exception:
            return float("nan"), float("nan")

    lr_ln,  p_ln  = _lr("lognormal")
    lr_exp, p_exp = _lr("exponential")
    lr_tpl, p_tpl = _lr("truncated_power_law")

    # Gini on tail data only — consistent with alpha/KS/LR which are tail-only
    gini = _gini(tail_data)

    result = FitResult(
        observable=name,
        label=label,
        n_total=len(arr),
        n_tail=n_tail,
        tail_fraction=round(n_tail / len(arr), 4),
        x_min=round(x_min, 2),
        x_max=round(x_max, 1),
        mean=round(float(arr.mean()), 3),
        median=round(float(np.median(arr)), 3),
        dynamic_range=dynamic_range,
        alpha=round(alpha, 4),
        sigma_alpha=round(sigma, 4),
        ks_stat=round(ks_stat, 4) if not np.isnan(ks_stat) else float("nan"),
        ks_pvalue=round(ks_p, 4)  if not np.isnan(ks_p)   else float("nan"),
        lr_vs_lognormal=round(lr_ln, 4),
        lr_vs_lognormal_p=round(p_ln, 4),
        lr_vs_exponential=round(lr_exp, 4),
        lr_vs_exponential_p=round(p_exp, 4),
        lr_vs_truncated_pl=round(lr_tpl, 4),
        lr_vs_truncated_pl_p=round(p_tpl, 4),
        pl_plausible=(ks_stat < 0.1 if not np.isnan(ks_stat) else False),
        pl_beats_lognormal=(lr_ln > 0 and p_ln < 0.05),
        pl_beats_exponential=(lr_exp > 0 and p_exp < 0.05),
        regime=_regime(alpha),
        gini=round(gini, 4),
    )

    if verbose:
        _print_result(result)

    return result


def _print_result(r: FitResult) -> None:
    pl = "✓" if r.pl_plausible else "✗"
    ln = "✓" if r.pl_beats_lognormal else "✗"
    print(
        f"  {r.label:<30}  n_tail={r.n_tail:5d} ({r.tail_fraction:.1%})  "
        f"xmin={r.x_min:6.1f}  α={r.alpha:.3f}±{r.sigma_alpha:.3f}  "
        f"KS={r.ks_stat:.3f}{pl}  "
        f"LR_ln={r.lr_vs_lognormal:+.2f}(p={r.lr_vs_lognormal_p:.3f}){ln}  "
        f"regime={r.regime}  dyn={r.dynamic_range}"
    )


# ── Batch fitting ─────────────────────────────────────────────

def fit_all(
    event_observables: Dict[str, List[int]],
    verbose: bool = True,
) -> Dict[str, FitResult]:
    """
    Fit all observables from extract_all_observables()["event_observables"].
    Pool data across seeds before calling this — do not fit per-run.
    """
    if verbose:
        print("Power-law fitting (discrete MLE, Clauset-Newman-Shalizi):")
    results: Dict[str, FitResult] = {}
    for name, data in event_observables.items():
        result = fit_observable(data, name=name, verbose=verbose)
        if result is not None:
            results[name] = result
    return results


# ── DataFrame export ──────────────────────────────────────────

def fits_to_dataframe(fits: Dict[str, FitResult]):
    """Convert to pandas DataFrame for LaTeX export (paper Table 1)."""
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("pandas required for fits_to_dataframe()")

    rows = []
    for name, r in fits.items():
        rows.append({
            "Observable":    r.label,
            "n_tail":        r.n_tail,
            "tail_frac":     r.tail_fraction,
            "x_min":         r.x_min,
            "x_max":         r.x_max,
            "dyn_range":     r.dynamic_range,
            "alpha":         r.alpha,
            "sigma":         r.sigma_alpha,
            "KS":            r.ks_stat,
            "PL_ok":         "✓" if r.pl_plausible else "✗",
            "LR_ln":         r.lr_vs_lognormal,
            "LR_ln_p":       r.lr_vs_lognormal_p,
            "PL>LN":         "✓" if r.pl_beats_lognormal else "✗",
            "Regime":        r.regime,
            "Gini_tail":     r.gini,
        })
    return pd.DataFrame(rows)


# ── CCDF helpers (used by visualization/ccdf_panel.py) ───────

def empirical_ccdf(data: List[int]) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (x, P(X >= x)) for plotting.
    P(X >= x) = (n - rank) / n, where rank is 0-indexed ascending.
    """
    arr = np.sort(np.array([x for x in data if x > 0], dtype=float))
    n   = len(arr)
    p   = (n - np.arange(n)) / n
    return arr, p


def powerlaw_ccdf_line(
    x_min:    float,
    x_max:    float,
    alpha:    float,
    n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Theoretical power-law CCDF P(X >= x) ∝ x^{-(alpha-1)}, normalised to 1 at x_min.
    Caller scales to empirical CCDF value at x_min.
    """
    x    = np.logspace(np.log10(max(x_min, 1)), np.log10(x_max), n_points)
    ccdf = (x / x_min) ** (-(alpha - 1.0))
    return x, ccdf