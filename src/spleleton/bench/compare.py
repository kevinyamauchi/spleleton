"""Comparing runs: paired statistics, flips, plots and the report (Sections 8, 9.2).

``bench compare A B C --baseline A -o DIR`` (:func:`compare_runs`) writes:

- ``report.md``: the scorecard, the paired comparisons that are flagged, the
  flips of events and errors with the null flip rate beside them, the error
  mix, the noise floor, and every plot;
- ``figures/``: the plots of Section 9.2 (and loss-correct against
  pipeline-correct, when a run has a probe of its variant);
- ``tables/``: the merged ``samples``, ``events`` and ``errors`` tables (with
  a ``run`` column) and the ``comparisons`` and ``flips`` tables, as parquet.

:func:`load` returns the merged tables, for analysis by hand.

**Comparisons** are paired over the samples every run completed. Per
(subset, stage, metric): the mean difference from the baseline with a BCa
interval, a Wilcoxon p, and win / tie / loss counts. Pooled over subsets
(``subset = "pooled"``), the interval resamples seeds, since the subsets on one
tree share it per seed (Section 4.2). A difference is **flagged** when its
interval excludes 0 and it is larger than the floor: the noise floor from the
baseline's jittered replicates when the store has them, and never less than
the minimum floor of one event per volume for counts and rates (P4).

**Flips** join events by ID: per (event type, subset, stage), how many events
the variant fixed and broke, the rate change with a cluster-bootstrap
interval over samples, and the null flip rate from the replicates. Errors
are matched by type and position (within :data:`ERROR_MATCH`): **fixed**
(only in the baseline), **new** (only in the variant) and **shared**.
"""

import dataclasses
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from spleleton._serialise import spec_hash
from spleleton.bench.stats import (
    POOLED,
    cluster_bootstrap,
    flagged,
    minimum_floor,
    noise_floor,
    null_flip_rates,
    paired_difference,
)
from spleleton.bench.store import Store

#: The keys of a sample's row.
SAMPLE_KEYS = ("subset", "seed", "stage")
#: Metrics compared: name -> (higher is better, kind). Error counts by type
#: and cause are added from the tables (lower is better, counts).
METRICS: dict[str, tuple[bool, str]] = {
    "centreline_f1": (True, "continuous"),
    "centreline_precision": (True, "continuous"),
    "centreline_recall": (True, "continuous"),
    "junction_f1": (True, "continuous"),
    "junction_precision": (True, "continuous"),
    "junction_recall": (True, "continuous"),
    "root_connected_fraction": (True, "continuous"),
    "radius_error": (False, "continuous"),
    "n_components": (False, "count"),
    "n_cycles": (False, "count"),
    "n_spur_ends": (False, "count"),
    "n_break_ends": (False, "count"),
    "n_algorithmic_breaks": (False, "count"),
    "junction_connected_rate": (True, "rate"),
    "thin_junction_connected_rate": (True, "rate"),
    "continuous_window_break_rate": (False, "rate"),
    "bump_spur_rate": (False, "rate"),
    "protrusion_spur_rate": (False, "rate"),
    "bump_deviation_mean": (False, "continuous"),
    "missed_length": (False, "continuous"),
    "missed_length_mask_lost": (False, "continuous"),
}
#: Sample columns that are not compared: fit descriptors, and the end counts
#: of the error table, which repeat ``compare_graphs``'s.
SKIPPED = (
    "n_segments", "n_alive", "n_rounds", "n_idle_rounds", "n_spur_end", "n_break_end",
)  # fmt: skip
#: The scorecard's metrics.
HEADLINE = (
    "centreline_f1",
    "junction_f1",
    "root_connected_fraction",
    "n_break_ends",
    "n_spur_ends",
    "thin_junction_connected_rate",
    "continuous_window_break_rate",
    "bump_spur_rate",
)
#: The events each rate is taken over: rate -> (event type, extra condition).
RATE_EVENTS: dict[str, tuple[str, str | None]] = {
    "junction_connected_rate": ("junction", None),
    "thin_junction_connected_rate": ("junction", "thin"),
    "continuous_window_break_rate": ("window", "continuous"),
    "bump_spur_rate": ("bump", None),
    "protrusion_spur_rate": ("protrusion", None),
}
#: The outcome whose flips are counted per event type, and its good value.
FLIP_OUTCOMES: dict[str, tuple[str, bool]] = {
    "junction": ("connected", True),
    "window": ("connected", True),
    "bump": ("spur", False),
    "protrusion": ("spur", False),
}
#: Errors of two runs match when of one type, sample and stage and this close.
ERROR_MATCH = 3.0


@dataclass
class RunTables:
    """A run's score tables and ``run.json``."""

    name: str
    directory: Path
    meta: dict[str, Any]
    samples: pd.DataFrame
    events: pd.DataFrame
    errors: pd.DataFrame


def read_tables(directory: Path | str) -> RunTables:
    """A run's three tables (empty frames where a table is missing)."""
    directory = Path(directory)
    meta = json.loads((directory / "run.json").read_text())
    tables = {}
    for table in ("samples", "events", "errors"):
        path = directory / f"{table}.parquet"
        tables[table] = pd.read_parquet(path) if path.exists() else pd.DataFrame()
    return RunTables(meta["name"], directory, meta, **tables)


def load(runs: Sequence[Path | str]) -> dict[str, pd.DataFrame]:
    """The runs' merged tables, each with a ``run`` column (the run's name).

    Parameters
    ----------
    runs : sequence of Path or str
        Run directories.

    Returns
    -------
    dict
        ``samples``, ``events`` and ``errors``.
    """
    loaded = [read_tables(run) for run in runs]
    out = {}
    for table in ("samples", "events", "errors"):
        frames = [getattr(r, table).assign(run=r.name) for r in loaded]
        frames = [f for f in frames if len(f)]
        out[table] = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out


def check_mixable(runs: Sequence[RunTables]) -> list[str]:
    """Refuse runs scored with different metrics; warn about code drift.

    Returns
    -------
    list of str
        Warnings (different ``src/`` hashes or commits).

    Raises
    ------
    ValueError
        When the runs' metrics versions differ.
    """
    versions = {r.name: r.meta["stage_versions"]["metrics"] for r in runs}
    if len(set(versions.values())) > 1:
        raise ValueError(f"runs scored with different metrics versions: {versions}")
    warnings = []
    hashes = {r.name: r.meta.get("provenance", {}).get("src_hash") for r in runs}
    if len(set(hashes.values())) > 1:
        warnings.append(
            "the runs come from different src/ hashes: "
            + ", ".join(f"{k} {str(v)[:12]}" for k, v in hashes.items())
        )
    return warnings


def metric_table(samples: pd.DataFrame) -> dict[str, tuple[bool, str]]:
    """The metrics of a samples table: :data:`METRICS` and the error counts."""
    out = {m: v for m, v in METRICS.items() if m in samples.columns}
    for column in samples.columns:
        if column.startswith("cause_") or (
            column.startswith("n_") and column not in out and column not in SKIPPED
        ):
            out[column] = (False, "count")
    return out


# -- floors ------------------------------------------------------------------------


def unjittered_hash(meta: dict[str, Any]) -> str:
    """The spec hash of a run's variant without its replicates' jitter.

    Replicate 0 and its jittered replicates share it, and a variant whose
    definition changed under the same name does not.
    """
    return spec_hash({**meta["spec"], "init_jitter": 0.0})


def replicate_runs(store: Store, run: RunTables) -> list[Path]:
    """The run and its jittered replicates in the store.

    A replicate has the run's variant name, unjittered spec
    (:func:`unjittered_hash`), suite, split, stage versions and numerics, and
    a replicate index above 0. The run itself is listed once, first.
    """
    found = [run.directory]
    meta = run.meta
    for path in sorted(store.root.glob(f"runs/{meta['suite']}/*/run.json")):
        other = json.loads(path.read_text())
        if (
            path.parent != run.directory
            and other.get("variant_name") == meta.get("variant_name")
            and other.get("replicate", 0) > 0
            and unjittered_hash(other) == unjittered_hash(meta)
            and other["split"] == meta["split"]
            and other["stage_versions"] == meta["stage_versions"]
            and other.get("numerics") == meta.get("numerics")
        ):
            found.append(path.parent)
    return found


@dataclass
class Floors:
    """The baseline's noise floor: per metric, and the null flip rates."""

    metrics: pd.DataFrame
    flips: pd.DataFrame
    replicates: int

    def floor(self, subset: str, stage: str, metric: str) -> float:
        """The measured floor (0 where none was measured)."""
        if self.metrics.empty:
            return 0.0
        rows = self.metrics[
            (self.metrics["subset"] == subset)
            & (self.metrics["stage"] == stage)
            & (self.metrics["metric"] == metric)
        ]
        return float(rows["floor"].iloc[0]) if len(rows) else 0.0

    def flip_rate(self, kind: str, subset: str, stage: str) -> float:
        """The null flip rate (NaN where none was measured)."""
        if self.flips.empty:
            return np.nan
        rows = self.flips[
            (self.flips["event_type"] == kind)
            & (self.flips["subset"] == subset)
            & (self.flips["stage"] == stage)
        ]
        return float(rows["flip_rate"].iloc[0]) if len(rows) else np.nan


def measure_floors(runs: Sequence[RunTables]) -> Floors:
    """The noise floor from replicate runs of one variant (empty for one run)."""
    if len(runs) < 2:
        return Floors(pd.DataFrame(), pd.DataFrame(), len(runs))
    metrics = list(metric_table(runs[0].samples))
    return Floors(
        noise_floor([r.samples for r in runs], metrics),
        null_flip_rates(
            [r.events for r in runs], {k: v[0] for k, v in FLIP_OUTCOMES.items()}
        ),
        len(runs),
    )


def _events_per_sample(events: pd.DataFrame, rate: str) -> pd.Series:
    """Per (subset, seed), the number of events a rate is taken over."""
    kind, condition = RATE_EVENTS[rate]
    rows = events[(events["event_type"] == kind) & (events["stage"] == "raw")]
    if condition is not None:
        if condition not in rows:  # no event of this type has it
            rows = rows.iloc[:0]
        else:
            rows = rows[rows[condition].astype("boolean").fillna(False).astype(bool)]
    return rows.groupby(["subset", "seed"]).size()


# -- comparisons -------------------------------------------------------------------


def compare_samples(
    base: RunTables, other: RunTables, floors: Floors | None = None
) -> pd.DataFrame:
    """Every (subset, stage, metric) of ``other`` against ``base``, flagged.

    Returns
    -------
    pandas.DataFrame
        Per subset (and pooled, :data:`POOLED`), stage and metric: the baseline's
        and the variant's means, the paired statistics (see
        :func:`~spleleton.bench.stats.paired_difference`), ``floor`` (the
        larger of the measured and the minimum floor) and ``flagged``.
    """
    floors = floors or Floors(pd.DataFrame(), pd.DataFrame(), 1)
    metrics = metric_table(base.samples)
    keys = list(SAMPLE_KEYS)
    joined = base.samples.merge(other.samples, on=keys, suffixes=("_a", "_b"))
    per_sample = {
        rate: _events_per_sample(base.events, rate)
        for rate in RATE_EVENTS
        if rate in metrics and len(base.events)
    }
    rows = []
    for (subset, stage), group in _with_pooled(joined).groupby(
        ["subset", "stage"], sort=False
    ):
        for metric, (higher, kind) in metrics.items():
            if f"{metric}_b" not in group:
                continue
            a = group[f"{metric}_a"].astype(float).to_numpy()
            b = group[f"{metric}_b"].astype(float).to_numpy()
            stats = paired_difference(b - a, higher)
            if stats["n"] == 0:
                continue
            if subset == POOLED:
                ok = np.isfinite(a) & np.isfinite(b)
                _, low, high = cluster_bootstrap(
                    (b - a)[ok], group["seed"].to_numpy()[ok], min_clusters=2
                )
                stats |= {"low": low, "high": high}
            counts = []
            if kind == "rate" and metric in per_sample:
                index = list(zip(group["subset_original"], group["seed"], strict=True))
                counts = [per_sample[metric].get(i, 0) for i in index]
            floor = max(
                floors.floor(subset, stage, metric), minimum_floor(kind, counts)
            )
            rows.append(
                {
                    "run": other.name,
                    "baseline": base.name,
                    "subset": subset,
                    "stage": stage,
                    "metric": metric,
                    "kind": kind,
                    "higher_is_better": higher,
                    "value_baseline": _mean(a),
                    "value": _mean(b),
                    **stats,
                    "floor": floor,
                    "flagged": flagged(
                        stats["mean"], stats["low"], stats["high"], floor
                    ),
                }
            )
    return pd.DataFrame(rows)


def _with_pooled(frame: pd.DataFrame) -> pd.DataFrame:
    """The frame, and again with every subset labelled :data:`POOLED`."""
    frame = frame.assign(subset_original=frame["subset"])
    return pd.concat([frame, frame.assign(subset=POOLED)], ignore_index=True)


def event_flips(
    base: RunTables, other: RunTables, floors: Floors | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Events joined by ID: flips per (event type, subset, stage), and the list.

    Returns
    -------
    table : pandas.DataFrame
        ``events``, ``flips``, ``fixed`` (to the good outcome), ``new`` (to
        the bad one), both rates (of the good outcome), the change's
        cluster-bootstrap interval over samples, and ``null_flip_rate``.
    flipped : pandas.DataFrame
        Every flipped event: its keys, ``outcome``, both values and whether
        it was ``fixed``.
    """
    floors = floors or Floors(pd.DataFrame(), pd.DataFrame(), 1)
    keys = ["subset", "seed", "stage", "event_type", "event_id"]
    rows, flipped = [], []
    for kind, (outcome, good) in FLIP_OUTCOMES.items():
        a = base.events[base.events["event_type"] == kind]
        b = other.events[other.events["event_type"] == kind]
        if a.empty or b.empty:
            continue
        joined = a[[*keys, outcome]].merge(
            b[[*keys, outcome]], on=keys, suffixes=("_a", "_b")
        )
        joined["good_a"] = joined[f"{outcome}_a"].astype(bool) == good
        joined["good_b"] = joined[f"{outcome}_b"].astype(bool) == good
        changed = joined[joined["good_a"] != joined["good_b"]]
        flipped.append(
            changed[keys].assign(
                outcome=outcome,
                value_baseline=changed[f"{outcome}_a"].astype(bool),
                value=changed[f"{outcome}_b"].astype(bool),
                fixed=changed["good_b"],
            )
        )
        joined = joined.assign(subset_original=joined["subset"])
        pooled = pd.concat([joined, joined.assign(subset=POOLED)], ignore_index=True)
        for (subset, stage), group in pooled.groupby(["subset", "stage"], sort=False):
            ga, gb = group["good_a"].to_numpy(), group["good_b"].to_numpy()
            clusters = (
                group["subset_original"].astype(str) + "/" + group["seed"].astype(str)
            )
            _, low, high = cluster_bootstrap(
                gb.astype(float) - ga.astype(float), clusters.to_numpy(), min_clusters=2
            )
            rows.append(
                {
                    "run": other.name,
                    "baseline": base.name,
                    "event_type": kind,
                    "outcome": outcome,
                    "subset": subset,
                    "stage": stage,
                    "events": len(group),
                    "flips": int((ga != gb).sum()),
                    "fixed": int((gb & ~ga).sum()),
                    "new": int((ga & ~gb).sum()),
                    "rate_baseline": float(ga.mean()),
                    "rate": float(gb.mean()),
                    "low": low,
                    "high": high,
                    "null_flip_rate": floors.flip_rate(kind, subset, stage),
                }
            )
    flipped = [f for f in flipped if len(f)]
    return (
        pd.DataFrame(rows),
        pd.concat(flipped, ignore_index=True) if flipped else pd.DataFrame(),
    )


def match_errors(base: pd.DataFrame, other: pd.DataFrame) -> pd.DataFrame:
    """Errors of two runs matched by sample, stage, type and position.

    Within each (subset, seed, stage, type), errors are matched one to one
    (by distance, within :data:`ERROR_MATCH`).

    Returns
    -------
    pandas.DataFrame
        Every error of either run: its keys, ``error_id`` in its run, its
        ``position``, ``cause`` and ``status`` (``fixed``: only in the
        baseline; ``new``: only in the variant; ``shared``), and the matched
        error's ID (``match``).
    """
    keys = ["subset", "seed", "stage", "type"]
    columns = [*keys, "error_id", "position", "cause"]
    rows = []
    groups_a = dict(list(base.groupby(keys))) if len(base) else {}
    groups_b = dict(list(other.groupby(keys))) if len(other) else {}
    for key in sorted(set(groups_a) | set(groups_b), key=str):
        a = groups_a.get(key, pd.DataFrame(columns=columns))
        b = groups_b.get(key, pd.DataFrame(columns=columns))
        pa, pb = _positions(a), _positions(b)
        pairs = []
        if len(a) and len(b):
            distance = np.linalg.norm(pa[:, None] - pb[None], axis=-1)
            cost = np.where(distance <= ERROR_MATCH, distance, 1e9)
            pairs = [
                (i, j)
                for i, j in zip(*linear_sum_assignment(cost), strict=True)
                if distance[i, j] <= ERROR_MATCH
            ]
        matched_a = dict(pairs)
        matched_b = {j: i for i, j in pairs}
        for side, frame, matched, other_frame in (
            ("baseline", a, matched_a, b),
            ("variant", b, matched_b, a),
        ):
            for k, row in enumerate(frame.itertuples(index=False)):
                if side == "variant" and k in matched:
                    continue  # listed once, from the baseline's side
                match = matched.get(k)
                rows.append(
                    {
                        **dict(zip(keys, key, strict=True)),
                        "run": side,
                        "error_id": row.error_id,
                        "position": list(row.position),
                        "cause": row.cause,
                        "status": "shared"
                        if match is not None
                        else ("fixed" if side == "baseline" else "new"),
                        "match": other_frame["error_id"].iloc[match]
                        if match is not None
                        else None,
                    }
                )
    return pd.DataFrame(rows)


def _positions(frame: pd.DataFrame) -> np.ndarray:
    """The ``position`` column as an ``(N, D)`` array."""
    if not len(frame):
        return np.zeros((0, 3))
    return np.array(frame["position"].tolist(), dtype=float)


def error_flips(matched: pd.DataFrame) -> pd.DataFrame:
    """Fixed, new and shared errors per (type, stage), from :func:`match_errors`."""
    if matched.empty:
        return pd.DataFrame(columns=["type", "stage", "fixed", "new", "shared"])
    table = matched.groupby(["type", "stage", "status"]).size().unstack(fill_value=0)
    for status in ("fixed", "new", "shared"):
        if status not in table:
            table[status] = 0
    return table[["fixed", "new", "shared"]].reset_index()


def _mean(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


# -- the probe beside the run ------------------------------------------------------

#: Per target family: the run's event outcome that is "pipeline-correct" and
#: the covariate the plot draws it against.
FAMILY_OUTCOMES: dict[str, tuple[str, str]] = {
    "junction_detach": ("connected", "radius_ratio"),
    "break_retract": ("connected", "curvature"),
    "bump_spur": ("no_spur", "height_over_r"),
    "bump_shift": ("centred", "height_over_r"),
}


def find_probe(store: Store, run: RunTables) -> Path | None:
    """The newest probe of the run's variant on the run's suite, if any."""
    name = run.meta.get("variant_name", run.name)
    found = [
        p.parent
        for p in store.root.glob(f"probes/{run.meta['suite']}/*/probe.json")
        if p.parent.name.split("__")[0] in (name, f"{name}-slim")
    ]
    return max(found, key=lambda p: p.stat().st_mtime) if found else None


def loss_and_pipeline(
    probe_events: pd.DataFrame, run_events: pd.DataFrame, stage: str = "raw"
) -> pd.DataFrame:
    """Probed events joined to the run's outcomes for the same events.

    Returns
    -------
    pandas.DataFrame
        Per probed event of a target family: ``family``, ``covariate``,
        ``x``, ``loss_correct``, ``pipeline_correct`` and ``cluster``.
    """
    from spleleton.bench.events import OFF_CENTRE

    events = run_events[run_events["stage"] == stage].assign(
        no_spur=lambda f: (
            ~f["spur"].astype("boolean").fillna(False) if "spur" in f else np.nan
        ),
        centred=lambda f: (
            f["deviation"].astype(float) <= OFF_CENTRE if "deviation" in f else np.nan
        ),
    )
    rows = []
    for family, (outcome, covariate) in FAMILY_OUTCOMES.items():
        probed = probe_events[probe_events["family"] == family]
        if probed.empty:
            continue
        probed = probed.groupby(["subset", "seed", "event_id"], as_index=False)[
            "loss_correct"
        ].min()  # an event probed at two opacities is correct only at both
        joined = probed.merge(
            events[["subset", "seed", "event_id", outcome, covariate]],
            on=["subset", "seed", "event_id"],
        )
        rows.append(
            pd.DataFrame(
                {
                    "family": family,
                    "covariate": covariate,
                    "x": joined[covariate].astype(float),
                    "loss_correct": joined["loss_correct"].astype(float),
                    "pipeline_correct": joined[outcome].astype(float),
                    "cluster": joined["subset"].astype(str)
                    + "/"
                    + joined["seed"].astype(str),
                }
            )
        )
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def loss_histories(store: Store, run: RunTables, every: int = 10) -> pd.DataFrame:
    """Each fitted sample's loss history (every ``every`` iterations)."""
    from spleleton.bench.runner import load_fit

    rows = []
    for label, entry in run.meta["samples"].items():
        if entry.get("status") != "done" or not entry.get("fit_key"):
            continue
        path = store.fit_path(entry["fit_key"])
        if not path.exists():
            continue
        history = load_fit(path).history
        for iteration in range(0, len(history), every):
            rows.append(
                {"run": run.name, "sample": label, "iteration": iteration,
                 **history[iteration]}
            )  # fmt: skip
    return pd.DataFrame(rows)


# -- the comparison ----------------------------------------------------------------


@dataclass
class Comparison:
    """Everything ``bench compare`` computes, before it is written."""

    runs: list[RunTables]
    baseline: RunTables
    floors: Floors
    comparisons: pd.DataFrame
    flips: pd.DataFrame
    flipped: pd.DataFrame
    errors: pd.DataFrame
    error_flips: pd.DataFrame
    warnings: list[str]


def compare(
    store: Store, runs: Sequence[Path | str], baseline: Path | str | None = None
) -> Comparison:
    """Compare runs with a baseline (the first run by default).

    The baseline's floor comes from its jittered replicates in the store.
    """
    tables = [read_tables(r) for r in runs]
    base = read_tables(baseline) if baseline is not None else tables[0]
    others = [t for t in tables if t.directory != base.directory]
    if not others:  # a run compared with itself, under a label of its own
        others = [dataclasses.replace(base, name=f"{base.name} (again)")]
    warnings = check_mixable([base, *others])
    floors = measure_floors([read_tables(d) for d in replicate_runs(store, base)])
    comparisons, flips, flipped, errors, error_table = [], [], [], [], []
    for other in others:
        comparisons.append(compare_samples(base, other, floors))
        table, changed = event_flips(base, other, floors)
        flips.append(table)
        flipped.append(changed.assign(run=other.name) if len(changed) else changed)
        matched = match_errors(base.errors, other.errors)
        errors.append(matched.assign(variant=other.name))
        error_table.append(error_flips(matched).assign(run=other.name))

    def concat(frames):
        frames = [f for f in frames if len(f)]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    return Comparison(
        [base, *others], base, floors, concat(comparisons), concat(flips),
        concat(flipped), concat(errors), concat(error_table), warnings,
    )  # fmt: skip


def compare_runs(
    store: Store,
    runs: Sequence[Path | str],
    output: Path | str,
    baseline: Path | str | None = None,
    report_name: str = "report.md",
    result: Comparison | None = None,
) -> Path:
    """``bench compare``: the tables, figures and ``report.md`` in ``output``.

    Returns
    -------
    Path
        The report.
    """
    from spleleton.bench import plots

    output = Path(output)
    (output / "tables").mkdir(parents=True, exist_ok=True)
    result = result or compare(store, runs, baseline)
    merged = {
        table: pd.concat(
            [getattr(r, table).assign(run=r.name) for r in result.runs],
            ignore_index=True,
        )
        for table in ("samples", "events", "errors")
    }
    for name, frame in {
        **merged,
        "comparisons": result.comparisons,
        "flips": result.flips,
        "flipped": result.flipped,
        "error_matches": result.errors,
    }.items():
        if len(frame):
            frame.to_parquet(output / "tables" / f"{name}.parquet", index=False)
    figures = output / "figures"
    order = [r.name for r in result.runs]
    drawn: dict[str, Path] = {}
    if len(result.comparisons):
        for stage in ("clean", "raw"):
            drawn[f"scorecard_{stage}"] = plots.plot_scorecard(
                result.comparisons, HEADLINE, stage, figures / f"scorecard_{stage}.png"
            )
            drawn[f"forest_{stage}"] = plots.plot_difference_forest(
                result.comparisons, HEADLINE[:4], stage, figures / f"forest_{stage}.png"
            )
    if len(merged["events"]):
        for stage in ("raw", "clean"):
            drawn[f"error_modes_{stage}"] = plots.plot_error_mode_curves(
                merged["events"], stage, figures / f"error_modes_{stage}.png"
            )
    drawn["paired_scatter"] = plots.plot_paired_scatter(
        merged["samples"], result.baseline.name, HEADLINE[:4], "clean",
        figures / "paired_scatter.png",
    )  # fmt: skip
    sweeps = _sweeps(result.runs)
    for stage in ("raw", "clean"):
        drawn[f"tradeoff_{stage}"] = plots.plot_tradeoff(
            merged["samples"], order, {k: [n for n, _ in v] for k, v in sweeps.items()},
            stage, figures / f"tradeoff_{stage}.png",
        )  # fmt: skip
    for (base, path), members in sweeps.items():
        slug = _slug(f"sweep_{base}_{path}")
        drawn[slug] = plots.plot_sweep(
            merged["samples"], members, path, HEADLINE[:4], "clean",
            figures / f"{slug}.png",
        )  # fmt: skip
    if len(result.flips) or len(result.error_flips):
        for stage in ("raw", "clean"):
            drawn[f"flips_{stage}"] = plots.plot_flips(
                result.flips, result.error_flips, stage, figures / f"flips_{stage}.png"
            )
    histories = [loss_histories(store, r) for r in result.runs]
    histories = [h for h in histories if len(h)]
    if histories:
        drawn["loss_curves"] = plots.plot_loss_curves(
            pd.concat(histories, ignore_index=True), figures / "loss_curves.png"
        )
    probes = {}
    for run in result.runs:
        probe = find_probe(store, run)
        if probe is None:
            continue
        table = loss_and_pipeline(pd.read_parquet(probe / "events.parquet"), run.events)
        if len(table):
            slug = f"loss_vs_pipeline_{_slug(run.name)}"
            drawn[slug] = plots.plot_loss_vs_pipeline(table, figures / f"{slug}.png")
            probes[run.name] = (probe, table)
    report = output / report_name
    report.write_text(write_report(result, drawn, probes, output))
    return report


def _sweeps(runs: Sequence[RunTables]) -> dict[tuple[str, str], list[tuple[str, Any]]]:
    """Runs that belong to a sweep, by (base, path), in the sweep's order."""
    out: dict[tuple[str, str], list[tuple[str, Any]]] = {}
    for run in runs:
        tag = run.meta.get("sweep")
        if tag:
            out.setdefault((tag["base"], tag["path"]), []).append(
                (run.name, tag["value"])
            )
    return {k: sorted(v, key=lambda m: m[1]) for k, v in out.items() if len(v) > 1}


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text)


# -- the report --------------------------------------------------------------------


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(value)
    return f"{value:.{digits}f}"


def _signed(value: Any, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value:+.{digits}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return [*lines, ""]


def _figure(drawn: dict[str, Path], key: str, output: Path, alt: str) -> list[str]:
    if key not in drawn:
        return []
    return [f"![{alt}]({drawn[key].relative_to(output).as_posix()})", ""]


def write_report(
    result: Comparison,
    drawn: dict[str, Path],
    probes: dict[str, tuple[Path, pd.DataFrame]],
    output: Path,
) -> str:
    """The comparison as markdown, in the plans' table format."""
    import time

    base = result.baseline
    comparisons = result.comparisons
    others = [r.name for r in result.runs[1:]]
    n_samples = base.samples[["subset", "seed"]].drop_duplicates().shape[0]
    src = str(base.meta.get("provenance", {}).get("src_hash"))[:12]
    lines = [
        f"# Comparison: {', '.join(others)} against {base.name}",
        "",
        (
            f"*{time.strftime('%d %B %Y')}. Suite `{base.meta['suite']}`, split "
            f"`{base.meta['split']}`, {n_samples} samples. Metrics version "
            f"{base.meta['stage_versions']['metrics']}. Code: src hash `{src}`.*"
        ),
        "",
    ]
    lines += [f"> **Warning:** {w}" for w in result.warnings]
    if result.warnings:
        lines.append("")
    lines += [
        "A difference is **flagged** (bold) when its interval excludes 0 and it is "
        "larger than the floor. Floors: "
        + (
            f"measured from {result.floors.replicates} replicates of the baseline, "
            "and never below one event per volume for counts and rates."
            if result.floors.replicates > 1
            else "no replicates of the baseline in this store, so only the minimum "
            "floor (one event per volume for counts and rates); continuous metrics "
            "have no floor yet."
        ),
        "",
        "## Scorecard",
        "",
        (
            "Pooled over subsets: the baseline's mean, and each run's "
            "paired difference from it."
        ),
        "",
    ]
    if len(comparisons):
        for stage in ("clean", "raw"):
            pooled = comparisons[
                (comparisons["subset"] == POOLED) & (comparisons["stage"] == stage)
            ]
            rows = []
            for metric in HEADLINE:
                part = pooled[pooled["metric"] == metric]
                if part.empty:
                    continue
                cells = [metric, _fmt(part["value_baseline"].iloc[0])]
                for name in others:
                    row = part[part["run"] == name]
                    if row.empty:
                        cells.append("-")
                        continue
                    row = row.iloc[0]
                    text = _signed(row["mean"])
                    cells.append(f"**{text}**" if row["flagged"] else text)
                rows.append(cells)
            lines += [f"**{stage} graph**", ""]
            lines += _table(["metric", base.name, *others], rows)
            lines += _figure(drawn, f"scorecard_{stage}", output, f"scorecard {stage}")
        lines += [
            "## Paired comparisons",
            "",
            (
                "Every flagged difference, per subset and pooled (the full table "
                "is `tables/comparisons.parquet`)."
            ),
            "",
        ]
        flagged_rows = comparisons[comparisons["flagged"]]
        lines += _table(
            ["run", "subset", "stage", "metric", "n", "mean difference", "95% interval",
             "Wilcoxon p", "win / tie / loss", "floor"],
            [
                [r.run, r.subset, r.stage, r.metric, r.n, _signed(r.mean),
                 f"[{_signed(r.low)}, {_signed(r.high)}]", _fmt(r.wilcoxon_p),
                 f"{r.wins} / {r.ties} / {r.losses}", _fmt(r.floor)]
                for r in flagged_rows.itertuples()
            ],
        )  # fmt: skip
        if flagged_rows.empty:
            lines += ["No difference is flagged.", ""]
        for stage in ("clean", "raw"):
            lines += _figure(drawn, f"forest_{stage}", output, f"forest {stage}")
    lines += ["## Flips", "", "Events joined by ID, pooled over subsets.", ""]
    if len(result.flips):
        pooled = result.flips[result.flips["subset"] == POOLED]
        lines += _table(
            ["run", "event", "outcome", "stage", "events", "fixed / new",
             "rate (good outcome)", "95% cluster interval of the change",
             "null flip rate"],
            [
                [r.run, r.event_type, r.outcome, r.stage, r.events,
                 f"{r.fixed} / {r.new}", f"{r.rate_baseline:.3f} -> {r.rate:.3f}",
                 f"[{_signed(r.low)}, {_signed(r.high)}]", _fmt(r.null_flip_rate)]
                for r in pooled.itertuples()
            ],
        )  # fmt: skip
    if len(result.error_flips):
        lines += [
            (
                f"Errors matched by type and position (within {ERROR_MATCH:g}): "
                "fixed (only in the baseline), new (only in the run), shared."
            ),
            "",
        ]
        lines += _table(
            ["run", "type", "stage", "fixed", "new", "shared"],
            [
                [r.run, r.type, r.stage, r.fixed, r.new, r.shared]
                for r in result.error_flips.itertuples()
            ],
        )
    for stage in ("raw", "clean"):
        lines += _figure(drawn, f"flips_{stage}", output, f"flips {stage}")
    lines += ["## Errors", "", "Errors per volume, by type and by cause.", ""]
    for stage in ("clean", "raw"):
        by_type, by_cause = {}, {}
        for run in result.runs:
            errors = (
                run.errors[run.errors["stage"] == stage]
                if len(run.errors)
                else run.errors
            )
            volumes = max(run.samples[["subset", "seed"]].drop_duplicates().shape[0], 1)
            if len(errors):
                by_type[run.name] = errors["type"].value_counts() / volumes
                by_cause[run.name] = (
                    errors["cause"].str.split("#").str[0].value_counts() / volumes
                )
        for title, table in (("type", by_type), ("cause", by_cause)):
            if not table:
                continue
            frame = pd.DataFrame(table).fillna(0)
            lines += [f"**{stage} graph, by {title}**", ""]
            lines += _table(
                [title, *frame.columns],
                [[i, *(f"{v:.1f}" for v in row)] for i, row in frame.iterrows()],
            )
    lines += ["## Noise floor", ""]
    if result.floors.replicates > 1 and len(result.floors.metrics):
        pooled = result.floors.metrics[result.floors.metrics["subset"] == POOLED]
        lines += [
            (
                f"From {result.floors.replicates} replicates of `{base.name}`: "
                "sigma_w = sqrt(mean d^2 / 2) over every replicate pair; floor = "
                "sqrt(2) sigma_w; smallest resolvable effect = (1.96 + 0.84) "
                "floor / sqrt(n) for n volumes."
            ),
            "",
        ]
        lines += _table(
            ["stage", "metric", "sigma_w", "floor", "8 volumes", "20 volumes"],
            [
                [r.stage, r.metric, _fmt(r.sigma_within), _fmt(r.floor),
                 _fmt(r.mde_8), _fmt(r.mde_20)]
                for r in pooled.itertuples()
                if r.floor > 0
            ],
        )  # fmt: skip
        zero = pooled[pooled["floor"] == 0].groupby("metric", sort=True)["stage"]
        named = [
            f"`{metric}`" + ("" if len(set(stages)) > 1 else f" ({stages.iloc[0]})")
            for metric, stages in zero
        ]
        if named:
            lines += [
                "Measured as 0, so the minimum floor applies: "
                + ", ".join(named)
                + ".",
                "",
            ]
        flips = result.floors.flips
        if len(flips):
            pooled_flips = flips[flips["subset"] == POOLED]
            lines += [
                (
                    "**Null flip rates**: the share of events whose outcome "
                    "differs between two replicates, over every pair of them."
                ),
                "",
            ]
            lines += _table(
                ["event", "outcome", "stage", "events", "flips", "null flip rate"],
                [
                    [r.event_type, FLIP_OUTCOMES[r.event_type][0], r.stage,
                     str(r.events), str(r.flips), _fmt(r.flip_rate)]
                    for r in pooled_flips.itertuples()
                ],
            )  # fmt: skip
    else:
        lines += [
            "No jittered replicates of the baseline yet (stage S4 makes them).",
            "",
        ]
    lines += ["## Error-mode curves", ""]
    for stage in ("raw", "clean"):
        lines += _figure(drawn, f"error_modes_{stage}", output, f"error modes {stage}")
    lines += ["## Paired samples and trade-offs", ""]
    lines += _figure(drawn, "paired_scatter", output, "paired scatter")
    for stage in ("raw", "clean"):
        lines += _figure(drawn, f"tradeoff_{stage}", output, f"trade-off {stage}")
    for key in drawn:
        if key.startswith("sweep_"):
            lines += _figure(drawn, key, output, key)
    lines += ["## Loss curves", ""]
    lines += _figure(drawn, "loss_curves", output, "loss curves") or [
        "No run has stored fits.",
        "",
    ]
    lines += ["## Loss-correct against pipeline-correct", ""]
    if probes:
        for name, (probe, table) in probes.items():
            rates = table.groupby("family")[["loss_correct", "pipeline_correct"]].mean()
            lines += [f"`{name}` with the probe `{probe.name}`:", ""]
            lines += _table(
                ["family", "events", "loss-correct", "pipeline-correct"],
                [
                    [f, int((table["family"] == f).sum()), _fmt(r.loss_correct),
                     _fmt(r.pipeline_correct)]
                    for f, r in rates.iterrows()
                ],
            )  # fmt: skip
            lines += _figure(
                drawn, f"loss_vs_pipeline_{_slug(name)}", output, "loss vs pipeline"
            )
    else:
        lines += ["No run has a probe of its variant on this suite.", ""]
    return "\n".join(lines)
