"""Reading a probe: loss-correct rates, the weight map, and the screen.

- :func:`loss_correct_rates`: per family, the share of events on which the
  loss prefers the correct configuration (every relaxed profile margin
  beyond ε = 0 above one voxel's worth of loss), with a cluster-bootstrap
  interval that resamples whole volumes (plan, Section 8.2);
- :func:`weight_map`: the same rate over a grid of two weights. The total is
  linear in the weights, ``ΔL = Σ_k w_k ΔL_k``, so any weights can be
  evaluated from the per-term profile margins without re-probing (the
  relaxation itself depends on the weights, so a chosen setting is
  confirmed by re-probing with it; Section 10.3);
- :func:`rate_floor`: from a probe run with ``floor``, the share of events
  whose loss-correctness changes when the margin threshold is raised to the
  event's floor: the rate difference that noise alone can make;
- :func:`screen`: the screen's pre-set rule (Section 10.7). A candidate
  passes when, against the current loss's probe on the same events, its
  loss-correct rate is higher in at least one target family by more than the
  floor with an interval that excludes 0, and lower in no family that both
  probe by more than the floor with an interval that excludes 0.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from spleleton.bench.probe._spec import TARGET_FAMILIES
from spleleton.bench.stats import cluster_bootstrap

#: What identifies a probed (event, family, opacity) across probes.
EVENT_KEYS = ("subset", "seed", "event_id", "family", "opacity")
#: Stands in for a missing opacity (families without a new primitive).
NO_OPACITY = -1.0
#: What a cluster is: one volume.
CLUSTER = ("subset", "seed")


def _clusters(frame: pd.DataFrame) -> np.ndarray:
    return frame[list(CLUSTER)].astype(str).agg("/".join, axis=1).to_numpy()


def loss_correct_rates(
    events: pd.DataFrame,
    column: str = "loss_correct",
    by: Sequence[str] = ("family",),
    n_boot: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Per group, the loss-correct rate with a cluster-bootstrap interval."""
    rows = []
    for key, group in events.groupby(list(by), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        rate, low, high = cluster_bootstrap(
            group[column].astype(float).to_numpy(), _clusters(group), n_boot, seed
        )
        rows.append(
            {
                **dict(zip(by, key, strict=True)),
                "n_events": len(group),
                "n_volumes": group[list(CLUSTER)].drop_duplicates().shape[0],
                "rate": rate,
                "low": low,
                "high": high,
            }
        )
    return pd.DataFrame(rows)


def profile_margins(margins: pd.DataFrame) -> pd.DataFrame:
    """Relaxed profile margins beyond ε = 0, one column per term."""
    usable = margins[
        (margins["eps"] > 0) & ~margins["collapsed"] & np.isfinite(margins["profile"])
    ]
    # most families have no opacity, and grouping drops missing keys
    usable = usable.assign(opacity=usable["opacity"].astype(float).fillna(NO_OPACITY))
    keys = [k for k in EVENT_KEYS if k in usable.columns]
    # (``dropna=False`` would expand the index to every combination of keys)
    return usable.pivot_table(
        index=[*keys, "eps"], columns="term", values="profile"
    ).reset_index()


def weight_map(
    margins: pd.DataFrame,
    weights: dict[str, float],
    term_a: str,
    term_b: str,
    values_a: Sequence[float],
    values_b: Sequence[float],
    tol: float,
) -> pd.DataFrame:
    """The loss-correct rate per family over a grid of two terms' weights.

    Parameters
    ----------
    margins : pandas.DataFrame
        The probe's margins table.
    weights : dict
        The probe's weights; the others stay at these.
    term_a, term_b : str
        The two terms whose weights vary.
    values_a, values_b : sequence of float
        Their values.
    tol : float
        The zero margin (one voxel's worth of loss).

    Returns
    -------
    pandas.DataFrame
        ``family``, ``weight_a``, ``weight_b``, ``rate`` and ``n_events``.
    """
    table = profile_margins(margins)
    terms = [t for t in weights if t in table.columns]
    keys = [k for k in EVENT_KEYS if k in table.columns]
    rows = []
    for a in values_a:
        for b in values_b:
            w = {**weights, term_a: a, term_b: b}
            total = sum(w[t] * table[t].fillna(0.0) for t in terms)
            correct = (total > tol).groupby([table[k] for k in keys]).all()
            per_family = correct.groupby(level="family").agg(["mean", "size"])
            for family, (rate, n) in per_family.iterrows():
                rows.append(
                    {
                        "family": family,
                        "weight_a": a,
                        "weight_b": b,
                        "rate": float(rate),
                        "n_events": int(n),
                    }
                )
    return pd.DataFrame(rows)


def rate_floor(events: pd.DataFrame) -> dict[str, float]:
    """Per family, the share of events whose loss-correctness the floor can flip."""
    if "loss_correct_floor" not in events.columns:
        return {}
    flips = events["loss_correct"].astype(bool) != events["loss_correct_floor"].astype(
        bool
    )
    return flips.groupby(events["family"]).mean().to_dict()


def screen(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    floor: dict[str, float] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[pd.DataFrame, bool]:
    """Compare a candidate's loss-correct rates with the baseline's (the screen).

    Parameters
    ----------
    candidate, baseline : pandas.DataFrame
        The two probes' events tables.
    floor : dict, optional
        Per family, the rate floor (:func:`rate_floor` of the baseline's
        floor run); 0 where missing.
    n_boot : int
        Bootstrap resamples.
    seed : int
        The bootstrap's seed.

    Returns
    -------
    table : pandas.DataFrame
        Per family both rates on the shared events, their difference and its
        cluster-bootstrap interval, and whether it is a gain or a loss.
    passes : bool
        Whether the candidate passes the screen.
    """
    floor = floor or {}
    keys = [k for k in EVENT_KEYS if k in candidate.columns and k in baseline.columns]
    joined = candidate[[*keys, "loss_correct"]].merge(
        baseline[[*keys, "loss_correct"]], on=keys, suffixes=("_candidate", "_baseline")
    )
    rows = []
    for family, group in joined.groupby("family", sort=False):
        difference = group["loss_correct_candidate"].astype(float) - group[
            "loss_correct_baseline"
        ].astype(float)
        mean, low, high = cluster_bootstrap(
            difference.to_numpy(), _clusters(group), n_boot, seed
        )
        threshold = floor.get(family, 0.0)
        rows.append(
            {
                "family": family,
                "n_events": len(group),
                "candidate": float(group["loss_correct_candidate"].mean()),
                "baseline": float(group["loss_correct_baseline"].mean()),
                "difference": mean,
                "low": low,
                "high": high,
                "floor": threshold,
                "gain": bool(mean > threshold and low > 0),
                "loss": bool(mean < -threshold and high < 0),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table, False
    gains = table[table["family"].isin(TARGET_FAMILIES)]["gain"].any()
    return table, bool(gains and not table["loss"].any())
