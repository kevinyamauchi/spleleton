"""Statistics of the benchmark (plan, Section 8).

The unit is the sample (a volume), never the voxel or the event.

- :func:`paired_difference`: the mean of per-sample differences from a
  baseline, with a BCa and a percentile bootstrap interval (10 000
  resamples), a Wilcoxon signed-rank p, and win / tie / loss counts;
- :func:`cluster_bootstrap`: a mean and a percentile interval that resample
  whole clusters (volumes for event rates; seeds when subsets are pooled,
  since subsets with one tree share it per seed, Section 4.2);
- :func:`noise_floor`: from jittered replicates of one variant, the
  within-sample standard deviation per metric and the floor, the standard
  deviation of a null paired difference (Section 8.3);
- :func:`null_flip_rates`: the share of events whose outcome differs between
  two replicates, per event type, subset and stage;
- :func:`minimum_floor`: what a floor measured as 0 is replaced by, one event
  per volume, for counts and rates (P4);
- :func:`flagged`: the flag rule, an interval that excludes 0 and a
  difference larger than the floor.
"""

import itertools
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

#: Bootstrap resamples of a paired difference.
RESAMPLES = 10_000
#: Bootstrap resamples of an event rate (cluster bootstrap).
CLUSTER_RESAMPLES = 2000
#: The subset label of comparisons pooled over subsets (not ``all``, a subset).
POOLED = "pooled"
#: Two-sided 5%, 80% power: the smallest effect ``(Z_ALPHA + Z_BETA) floor / √n``.
Z_ALPHA, Z_BETA = 1.96, 0.84


def paired_difference(
    differences: Sequence[float],
    higher_is_better: bool = True,
    resamples: int = RESAMPLES,
    seed: int = 0,
) -> dict[str, Any]:
    """Summarise per-sample differences ``other - baseline``.

    Parameters
    ----------
    differences : sequence of float
        One per sample both runs scored; NaNs are dropped.
    higher_is_better : bool
        Which sign counts as a win.
    resamples : int
        Bootstrap resamples.
    seed : int
        The percentile bootstrap's seed (BCa uses its own, fixed).

    Returns
    -------
    dict
        ``n``, ``mean``, ``low`` and ``high`` (BCa, or percentile where BCa
        fails; NaN for fewer than 2 samples), ``low_percentile``,
        ``high_percentile``, ``wilcoxon_p``, ``wins``, ``ties``, ``losses``.
    """
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    out: dict[str, Any] = {"n": len(d)}
    if not len(d):
        return out | {
            "mean": np.nan, "low": np.nan, "high": np.nan,
            "low_percentile": np.nan, "high_percentile": np.nan,
            "wilcoxon_p": np.nan, "wins": 0, "ties": 0, "losses": 0,
        }  # fmt: skip
    better = d > 0 if higher_is_better else d < 0
    worse = d < 0 if higher_is_better else d > 0
    low = high = low_p = high_p = np.nan  # one sample has no interval
    if len(d) >= 2:
        rng = np.random.default_rng(seed)
        means = d[rng.integers(0, len(d), size=(resamples, len(d)))].mean(axis=1)
        low_p, high_p = np.percentile(means, [2.5, 97.5])
        low, high = _bca(d, resamples)
        if not (np.isfinite(low) and np.isfinite(high)):
            low, high = low_p, high_p
    if np.all(d == 0):
        p = 1.0
    else:
        try:
            p = float(stats.wilcoxon(d).pvalue)
        except ValueError:
            p = np.nan
    return out | {
        "mean": float(d.mean()),
        "low": float(low),
        "high": float(high),
        "low_percentile": float(low_p),
        "high_percentile": float(high_p),
        "wilcoxon_p": p,
        "wins": int(better.sum()),
        "ties": int((d == 0).sum()),
        "losses": int(worse.sum()),
    }


def _bca(d: np.ndarray, resamples: int) -> tuple[float, float]:
    """BCa interval of the mean; the point when the data are constant."""
    if np.all(d == d[0]):
        return float(d[0]), float(d[0])
    try:
        result = stats.bootstrap(
            (d,), np.mean, n_resamples=resamples, method="BCa", random_state=0
        )
    except (ValueError, FloatingPointError):
        return np.nan, np.nan
    low, high = result.confidence_interval
    return float(low), float(high)


def cluster_bootstrap(
    values: Sequence[float],
    clusters: Sequence[Any],
    resamples: int = CLUSTER_RESAMPLES,
    seed: int = 0,
    level: float = 0.95,
    min_clusters: int = 1,
) -> tuple[float, float, float]:
    """The mean of ``values`` and a percentile interval resampling whole clusters.

    Each resample draws as many clusters as there are, with replacement, and
    takes the mean over all their values (so a cluster weighs by its size).

    Returns
    -------
    tuple of float
        The mean and the interval's two ends (NaN for no values; the ends are
        NaN for fewer than ``min_clusters`` clusters).
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    names, index = np.unique(np.asarray(clusters).astype(str), return_inverse=True)
    if len(names) < min_clusters:
        return float(values.mean()), np.nan, np.nan
    sums = np.bincount(index, weights=values, minlength=len(names))
    counts = np.bincount(index, minlength=len(names))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(names), size=(resamples, len(names)))
    means = sums[draws].sum(axis=1) / np.maximum(counts[draws].sum(axis=1), 1)
    alpha = (1 - level) / 2
    return (
        float(values.mean()),
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1 - alpha)),
    )


def noise_floor(
    replicates: Sequence[pd.DataFrame],
    metrics: Sequence[str],
    by: Sequence[str] = ("subset", "stage"),
) -> pd.DataFrame:
    """The noise floor per metric, from replicate runs of one variant.

    Every pair of replicates is a null comparison. With ``d`` the paired
    differences over samples and pairs, the within-sample standard deviation
    is ``sigma_w = sqrt(mean(d^2) / 2)`` and the floor, the standard deviation
    of a null paired difference, is ``sqrt(2) sigma_w``.

    Parameters
    ----------
    replicates : sequence of pandas.DataFrame
        Each replicate's samples table (``subset``, ``seed``, ``stage`` and
        the metrics).
    metrics : sequence of str
        The metrics.
    by : sequence of str
        Groups with a floor of their own; the pooled floor (``subset`` =
        :data:`POOLED`) is added when ``subset`` is among them.

    Returns
    -------
    pandas.DataFrame
        Per group and metric: ``n`` (differences), ``sigma_within``,
        ``floor``, and the smallest effects 8 and 20 samples resolve
        (``mde_8``, ``mde_20``).
    """
    keys = ["subset", "seed", "stage"]
    differences = []
    for a, b in itertools.combinations(replicates, 2):
        joined = a.merge(b, on=keys, suffixes=("_a", "_b"))
        for metric in metrics:
            if f"{metric}_a" not in joined:
                continue
            d = joined[f"{metric}_b"].astype(float) - joined[f"{metric}_a"].astype(
                float
            )
            differences.append(joined[keys].assign(metric=metric, d=d.to_numpy()))
    if not differences:
        return pd.DataFrame(
            columns=[*by, "metric", "n", "sigma_within", "floor", "mde_8", "mde_20"]
        )
    frame = pd.concat(differences, ignore_index=True).dropna(subset=["d"])
    groups = [frame]
    if "subset" in by:
        groups.append(frame.assign(subset=POOLED))
    rows = []
    for key, group in pd.concat(groups).groupby([*by, "metric"], sort=False):
        d = group["d"].to_numpy()
        sigma = float(np.sqrt(np.mean(d**2) / 2))
        floor = float(np.sqrt(2) * sigma)
        rows.append(
            {
                **dict(zip([*by, "metric"], key, strict=True)),
                "n": len(d),
                "sigma_within": sigma,
                "floor": floor,
                "mde_8": (Z_ALPHA + Z_BETA) * floor / np.sqrt(8),
                "mde_20": (Z_ALPHA + Z_BETA) * floor / np.sqrt(20),
            }
        )
    return pd.DataFrame(rows)


def null_flip_rates(
    replicates: Sequence[pd.DataFrame], outcomes: dict[str, str]
) -> pd.DataFrame:
    """The share of events whose outcome differs between two replicates.

    Parameters
    ----------
    replicates : sequence of pandas.DataFrame
        Each replicate's events table.
    outcomes : dict
        Event type -> the outcome column whose flips are counted.

    Returns
    -------
    pandas.DataFrame
        Per event type, subset (and :data:`POOLED`) and stage: ``events`` (event
        comparisons, over every pair of replicates), ``flips`` and
        ``flip_rate``.
    """
    keys = ["subset", "seed", "stage", "event_type", "event_id"]
    rows = []
    for a, b in itertools.combinations(replicates, 2):
        for kind, outcome in outcomes.items():
            pa = a[(a["event_type"] == kind)][[*keys, outcome]]
            pb = b[(b["event_type"] == kind)][[*keys, outcome]]
            joined = pa.merge(pb, on=keys, suffixes=("_a", "_b"))
            if joined.empty:
                continue
            flipped = joined[f"{outcome}_a"].astype(bool) != joined[
                f"{outcome}_b"
            ].astype(bool)
            rows.append(joined[["subset", "stage", "event_type"]].assign(flip=flipped))
    if not rows:
        return pd.DataFrame(
            columns=["event_type", "subset", "stage", "events", "flips", "flip_rate"]
        )
    frame = pd.concat(rows, ignore_index=True)
    frame = pd.concat([frame, frame.assign(subset=POOLED)])
    table = (
        frame.groupby(["event_type", "subset", "stage"], sort=False)["flip"]
        .agg(events="size", flips="sum")
        .reset_index()
    )
    table["flips"] = table["flips"].astype(int)
    table["flip_rate"] = table["flips"] / table["events"]
    return table


def minimum_floor(kind: str, events_per_sample: Sequence[float] = ()) -> float:
    """The floor a metric gets when its measured floor is 0 (one event per volume).

    Parameters
    ----------
    kind : {"count", "rate", "continuous"}
        The metric's kind.
    events_per_sample : sequence of float
        For a rate, the number of events it is taken over in each sample.

    Returns
    -------
    float
        1 for a count; for a rate, the mean over samples of ``1 / events``
        (the change of one event in every volume); 0 for a continuous
        metric.
    """
    if kind == "count":
        return 1.0
    if kind == "rate":
        n = np.asarray(events_per_sample, dtype=float)
        n = n[n > 0]
        return float(np.mean(1 / n)) if len(n) else 0.0
    return 0.0


def flagged(mean: float, low: float, high: float, floor: float) -> bool:
    """A difference is flagged when its interval excludes 0 and it beats the floor."""
    if not (np.isfinite(mean) and np.isfinite(low) and np.isfinite(high)):
        return False
    return bool((low > 0 or high < 0) and abs(mean) > floor)
