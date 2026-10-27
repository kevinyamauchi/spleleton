"""Plots of the benchmark (matplotlib, PNG).

So far the probe's (plan, Section 10.8):

- :func:`plot_eps_profiles`: per family, the profile margin against ε (the
  median of each weighted term, and the total's median with its
  interquartile band), and the force the fit feels;
- :func:`plot_weight_map`: per family, the loss-correct rate over a grid of
  two weights, with the current setting marked;
- :func:`plot_screening`: per family, each candidate's loss-correct rate
  with its cluster-bootstrap interval, the current loss's rate marked;
- :func:`plot_repair_classes`: the share of search, loss, assembly and
  undetermined errors, per error type and per cause;
- :func:`plot_threshold_curves`, :func:`plot_tornado`,
  :func:`plot_flip_distances` and :func:`plot_validation`: the sensitivity
  report's (Section 11.7).

Colours follow one fixed categorical order (by term, never by rank), a
single blue ramp for magnitudes, and recessive axes; each panel has one
y-axis.
"""

import itertools
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from spleleton.bench.stats import POOLED, cluster_bootstrap

#: The chart surface and ink.
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
#: Categorical slots, in their fixed order.
CATEGORICAL = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
#: The sequential blue ramp, light to dark.
BLUES = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
NEUTRAL = "#8a8984"


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT_SECONDARY,
            "axes.titlecolor": TEXT,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": TEXT_SECONDARY,
            "ytick.color": TEXT_SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "lines.linewidth": 2,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def term_colours(terms: Sequence[str]) -> dict[str, str]:
    """A fixed colour per term (``total`` in ink), in the order given."""
    colours = {"total": TEXT}
    others = [t for t in terms if t != "total"]
    for index, term in enumerate(others):
        colours[term] = CATEGORICAL[index % len(CATEGORICAL)]
    return colours


def plot_eps_profiles(
    margins: pd.DataFrame,
    weights: dict[str, float],
    path: Path,
    families: Sequence[str] | None = None,
) -> Path:
    """Profile margins and forces against ε, per family.

    Parameters
    ----------
    margins : pandas.DataFrame
        A probe's margins table.
    weights : dict
        The probe's weights (terms are drawn weighted, so they sum to the total).
    path : Path
        The PNG to write.
    families : sequence of str, optional
        The families to draw, in order (all, by default).

    Returns
    -------
    Path
        ``path``.
    """
    plt = _pyplot()
    families = list(families or margins["family"].drop_duplicates())
    terms = [t for t in weights if t in set(margins["term"])]
    colours = term_colours(terms)
    usable = margins[~margins["collapsed"]]
    fig, axes = plt.subplots(
        len(families), 2, figsize=(9, 2.4 * len(families)), squeeze=False
    )
    for row, family in enumerate(families):
        data = usable[usable["family"] == family]
        margin_ax, force_ax = axes[row]
        for term in [*terms, "total"]:
            part = data[data["term"] == term]
            scale = 1.0 if term == "total" else weights[term]
            grouped = (part.groupby("eps")["profile"].median() * scale).dropna()
            if grouped.empty:
                continue
            width = 2 if term == "total" else 1.2
            margin_ax.plot(
                grouped.index, grouped.values, color=colours[term], lw=width,
                marker="o", ms=3, label=term,
            )  # fmt: skip
            if term == "total":
                q = part.groupby("eps")["profile"].quantile([0.25, 0.75]).unstack()
                q = q.dropna()
                margin_ax.fill_between(
                    q.index, q[0.25], q[0.75], color=NEUTRAL, alpha=0.18, lw=0
                )
        total = data[data["term"] == "total"]
        force = total.groupby("eps")["force_rigid"]
        median = force.median()
        q = force.quantile([0.25, 0.75]).unstack()
        force_ax.plot(median.index, median.values, color=TEXT, marker="o", ms=3)
        force_ax.fill_between(
            q.index, q[0.25], q[0.75], color=NEUTRAL, alpha=0.18, lw=0
        )
        for ax in (margin_ax, force_ax):
            ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
            ax.set_xlabel("ε")
        n = data[["subset", "seed", "event_id", "opacity"]].drop_duplicates().shape[0]
        margin_ax.set_title(f"{family} ({n} events): profile margin")
        margin_ax.set_ylabel("weighted ΔL")
        force_ax.set_title(f"{family}: force along ε (rigid, fit's convention)")
        force_ax.set_ylabel("dL/dε")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_weight_map(
    table: pd.DataFrame,
    current: tuple[float, float],
    names: tuple[str, str],
    path: Path,
) -> Path:
    """The loss-correct rate over two weights, per family (from ``weight_map``)."""
    plt = _pyplot()
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    families = list(table["family"].drop_duplicates())
    columns = min(len(families), 4)
    rows = int(np.ceil(len(families) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(3.2 * columns + 1, 3.0 * rows),
        squeeze=False,
        layout="constrained",
    )
    image = None
    for ax, family in zip(axes.flat, families, strict=False):
        part = table[table["family"] == family]
        grid = part.pivot_table(index="weight_b", columns="weight_a", values="rate")
        a, b = grid.columns.to_numpy(), grid.index.to_numpy()
        image = ax.imshow(
            grid.to_numpy(), origin="lower", cmap=cmap, vmin=0, vmax=1,
            aspect="auto", interpolation="nearest",
        )  # fmt: skip
        ax.set_xticks(range(len(a))[::2], [f"{v:.2g}" for v in a][::2])
        ax.set_yticks(range(len(b))[::2], [f"{v:.2g}" for v in b][::2])
        ax.grid(False)
        x = np.interp(current[0], a, np.arange(len(a)))
        y = np.interp(current[1], b, np.arange(len(b)))
        ax.plot(x, y, marker="x", color=TEXT, ms=8, mew=2)
        ax.set_xlabel(f"{names[0]} weight")
        ax.set_ylabel(f"{names[1]} weight")
        ax.set_title(f"{family} (n = {int(part['n_events'].max())})")
    for ax in list(axes.flat)[len(families) :]:
        ax.set_visible(False)
    if image is not None:
        bar = fig.colorbar(image, ax=axes, shrink=0.6)
        bar.set_label("loss-correct rate", color=TEXT_SECONDARY)
        bar.outline.set_visible(False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_screening(rates: pd.DataFrame, baseline: str, path: Path) -> Path:
    """Loss-correct rates per candidate, one panel per family.

    Parameters
    ----------
    rates : pandas.DataFrame
        ``candidate``, ``family``, ``rate``, ``low``, ``high`` (from
        ``loss_correct_rates`` of each probe).
    baseline : str
        The current loss's candidate name; its rate is drawn as a line.
    path : Path
        The PNG to write.
    """
    plt = _pyplot()
    families = list(rates["family"].drop_duplicates())
    candidates = list(rates["candidate"].drop_duplicates())
    fig, axes = plt.subplots(
        1, len(families),
        figsize=(2.6 * len(families) + 1.5, 0.35 * len(candidates) + 1.4),
        sharey=True, squeeze=False,
    )  # fmt: skip
    for ax, family in zip(axes[0], families, strict=True):
        part = rates[rates["family"] == family].set_index("candidate")
        for y, candidate in enumerate(candidates):
            if candidate not in part.index:
                continue
            row = part.loc[candidate]
            colour = TEXT if candidate == baseline else CATEGORICAL[0]
            ax.plot([row["low"], row["high"]], [y, y], color=colour, lw=2)
            ax.plot(row["rate"], y, "o", color=colour, ms=7, mec=SURFACE, mew=1.5)
        if baseline in part.index:
            ax.axvline(
                part.loc[baseline, "rate"], color=TEXT_SECONDARY, lw=0.8, ls="--"
            )
        ax.set_xlim(-0.03, 1.03)
        ax.set_title(family)
        ax.set_xlabel("loss-correct rate")
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(range(len(candidates)), candidates)
    axes[0][0].invert_yaxis()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# -- runs (plan, Section 9.2) ----------------------------------------------------------

#: The colour that marks a better and a worse flagged difference.
BETTER, WORSE = CATEGORICAL[2], CATEGORICAL[1]


def run_colours(runs: Sequence[str]) -> dict[str, str]:
    """A fixed colour per run, in the order given (the baseline first, in ink)."""
    return {
        run: TEXT if index == 0 else CATEGORICAL[(index - 1) % len(CATEGORICAL)]
        for index, run in enumerate(runs)
    }


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return path


def _value(value: float, digits: int = 2) -> str:
    return "-" if not np.isfinite(value) else f"{value:.{digits}f}"


def plot_scorecard(
    comparisons: pd.DataFrame, metrics: Sequence[str], stage: str, path: Path
) -> Path:
    """Variants by headline metrics, per subset (one panel per variant).

    Each cell holds the baseline's mean and the variant's difference from it,
    coloured better or worse when flagged, grey otherwise.
    """
    plt = _pyplot()
    rows = comparisons[
        (comparisons["stage"] == stage) & comparisons["metric"].isin(metrics)
    ]
    runs = list(rows["run"].drop_duplicates())
    subsets = list(rows["subset"].drop_duplicates())
    metrics = [m for m in metrics if m in set(rows["metric"])]
    n = max(len(runs), 1)
    fig, axes = plt.subplots(
        n, 1, figsize=(1.05 * len(metrics) + 2.2, (0.42 * len(subsets) + 1.9) * n),
        squeeze=False,
    )  # fmt: skip
    for ax, run in zip(axes[:, 0], runs, strict=False):
        part = rows[rows["run"] == run].set_index(["subset", "metric"])
        for y, subset in enumerate(subsets):
            for x, metric in enumerate(metrics):
                if (subset, metric) not in part.index:
                    continue
                row = part.loc[(subset, metric)]
                colour = NEUTRAL
                if row["flagged"]:
                    good = (row["mean"] > 0) == bool(row["higher_is_better"])
                    colour = BETTER if good else WORSE
                ax.add_patch(
                    plt.Rectangle(
                        (x - 0.48, y - 0.45), 0.96, 0.9, color=colour,
                        alpha=0.85 if row["flagged"] else 0.18, lw=0,
                    )
                )  # fmt: skip
                text = SURFACE if row["flagged"] else TEXT
                ax.text(
                    x, y, f"{_value(row['value_baseline'])}\n{row['mean']:+.3f}",
                    ha="center", va="center", fontsize=7, color=text,
                )  # fmt: skip
        ax.set_xlim(-0.5, len(metrics) - 0.5)
        ax.set_ylim(len(subsets) - 0.5, -0.5)
        ax.set_xticks(range(len(metrics)), metrics, rotation=40, ha="right")
        ax.set_yticks(range(len(subsets)), subsets)
        ax.grid(False)
        ax.set_title(f"{run} - baseline ({stage})")
    fig.tight_layout()
    return _save(fig, path)


def plot_difference_forest(
    comparisons: pd.DataFrame, metrics: Sequence[str], stage: str, path: Path
) -> Path:
    """Per metric (rows) and subset (columns): each variant's difference, interval."""
    plt = _pyplot()
    rows = comparisons[
        (comparisons["stage"] == stage) & comparisons["metric"].isin(metrics)
    ]
    runs = list(rows["run"].drop_duplicates())
    subsets = list(rows["subset"].drop_duplicates())
    metrics = [m for m in metrics if m in set(rows["metric"])]
    colours = run_colours(["baseline", *runs])
    # room for the row labels: the metric and the run names
    labels = max(map(len, metrics), default=0) + max(map(len, runs), default=0)
    width = 1.9 * len(subsets) + 0.075 * labels + 0.8
    fig, axes = plt.subplots(
        len(metrics), len(subsets),
        figsize=(width, (0.28 * len(runs) + 1.2) * len(metrics)),
        squeeze=False, sharey=True,
    )  # fmt: skip
    for i, metric in enumerate(metrics):
        for j, subset in enumerate(subsets):
            ax = axes[i][j]
            part = rows[(rows["metric"] == metric) & (rows["subset"] == subset)]
            floor = part["floor"].max() if len(part) else 0
            if floor > 0:
                ax.axvspan(-floor, floor, color=NEUTRAL, alpha=0.15, lw=0)
            ax.axvline(0, color=TEXT_SECONDARY, lw=0.8)
            for y, run in enumerate(runs):
                row = part[part["run"] == run]
                if row.empty:
                    continue
                row = row.iloc[0]
                ax.plot([row["low"], row["high"]], [y, y], color=colours[run], lw=2)
                ax.plot(row["mean"], y, "o", color=colours[run], ms=5)
            ax.set_yticks(range(len(runs)), runs)
            ax.set_ylim(len(runs) - 0.5, -0.5)
            ax.grid(axis="y", visible=False)
            if i == 0:
                ax.set_title(subset)
            if j == 0:
                ax.set_ylabel(metric, rotation=0, ha="right", va="center")
    fig.tight_layout()
    return _save(fig, path)


def binned(
    frame: pd.DataFrame,
    covariate: str,
    outcome: str,
    edges: Sequence[float],
    clusters: str = "cluster",
) -> pd.DataFrame:
    """An outcome's mean per covariate bin, with a cluster-bootstrap band."""
    rows = []
    values = frame[covariate].astype(float)
    for low, high in itertools.pairwise(edges):
        part = frame[(values > low) & (values <= high)]
        y = part[outcome].astype(float)
        part, y = part[np.isfinite(y)], y[np.isfinite(y)]
        if part.empty:
            continue
        mean, lo, hi = cluster_bootstrap(y.to_numpy(), part[clusters].to_numpy())
        rows.append(
            {"centre": float(part[covariate].astype(float).median()), "n": len(part),
             "mean": mean, "low": lo, "high": hi}
        )  # fmt: skip
    return pd.DataFrame(rows)


def _edges(values: pd.Series, n: int = 5) -> np.ndarray:
    """Quantile bin edges (unique), open below the smallest value.

    A covariate with at most ``n`` distinct values gets one bin per value.
    """
    values = values.astype(float)
    values = values[np.isfinite(values)]
    if values.empty:
        return np.array([0.0, 1.0])
    distinct = np.unique(values)
    if len(distinct) <= n:  # one bin per value
        middle = (distinct[:-1] + distinct[1:]) / 2
        return np.concatenate([[distinct[0] - 1e-9], middle, [distinct[-1]]])
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n + 1)))
    edges[0] = edges[0] - 1e-9
    return edges if len(edges) > 1 else np.array([edges[0], edges[0] + 1])


#: Error-mode panels: (title, event type, condition, covariate, outcome, label).
ERROR_MODE_PANELS = (
    (
        "continuous windows broken",
        "window",
        "continuous",
        "curvature",
        "broken",
        "curvature (rad / voxel)",
    ),
    (
        "continuous windows broken",
        "window",
        "continuous",
        "r_local",
        "broken",
        "local radius",
    ),
    (
        "continuous windows broken",
        "window",
        "continuous",
        "junction_distance_over_r",
        "broken",
        "distance to junction / r",
    ),
    (
        "junction branches connected",
        "junction",
        None,
        "radius_ratio",
        "connected",
        "r_child / r_max",
    ),
    ("bumps with a spur", "bump", None, "height_over_r", "spur", "height / r"),
    ("bump deviation (r)", "bump", None, "height_over_r", "deviation", "height / r"),
    ("window deviation (r)", "window", None, "roughness", "deviation", "roughness"),
)
#: Radius-ratio bins of the junction curve (P4).
RATIO_EDGES = (0.0, 0.5, 0.7, 0.85, 1.0)


def plot_error_mode_curves(events: pd.DataFrame, stage: str, path: Path) -> Path:
    """The error-mode curves of every run in ``events`` (with a ``run`` column).

    Binned rates with cluster-bootstrap bands (clusters are samples). The
    junction panel splits its runs' curves by ``fit_class`` where there is
    one (``reaches`` solid, ``short`` dashed); the bump-spur panel draws the
    protrusions of the noise subsets dotted, on the same axis.
    """
    plt = _pyplot()
    runs = list(events["run"].drop_duplicates())
    colours = run_colours(runs)
    data = events[events["stage"] == stage].assign(
        cluster=lambda f: f["subset"].astype(str) + "/" + f["seed"].astype(str)
    )
    if "connected" in data:
        data = data.assign(broken=~data["connected"].astype("boolean").fillna(True))
    columns = 4
    rows = int(np.ceil(len(ERROR_MODE_PANELS) / columns))
    fig, axes = plt.subplots(
        rows, columns, figsize=(3.3 * columns, 2.8 * rows), squeeze=False
    )
    for ax, (title, kind, condition, covariate, outcome, label) in zip(
        axes.flat, ERROR_MODE_PANELS, strict=False
    ):
        part = data[data["event_type"] == kind]
        if condition is not None and condition in part:
            part = part[part[condition].astype("boolean").fillna(False).astype(bool)]
        if part.empty or covariate not in part or outcome not in part:
            ax.set_title(f"{title}: no events")
            continue
        edges = RATIO_EDGES if covariate == "radius_ratio" else _edges(part[covariate])
        for run in runs:
            mine = part[part["run"] == run]
            groups = [("", mine, "-")]
            if (
                kind == "junction"
                and "fit_class" in mine
                and mine["fit_class"].notna().any()
            ):
                groups = [
                    (f" {c}", mine[mine["fit_class"] == c], style)
                    for c, style in (("reaches", "-"), ("short", "--"))
                ]
            for suffix, group, style in groups:
                curve = binned(group, covariate, outcome, edges)
                if curve.empty:
                    continue
                ax.plot(curve["centre"], curve["mean"], style, color=colours[run],
                        marker="o", ms=3, label=f"{run}{suffix}")  # fmt: skip
                ax.fill_between(curve["centre"], curve["low"], curve["high"],
                                color=colours[run], alpha=0.12, lw=0)  # fmt: skip
        if kind == "bump" and outcome == "spur":
            noise = data[
                (data["event_type"] == "protrusion")
                & data["subset"].astype(str).str.startswith("noise")
            ]
            for run in runs:
                curve = binned(
                    noise[noise["run"] == run], "height_over_r", "spur", edges
                )
                if not curve.empty:
                    ax.plot(curve["centre"], curve["mean"], ":", color=colours[run],
                            marker="s", ms=3, label=f"{run} protrusions (noise)")  # fmt: skip  # noqa: E501
        ax.set_title(title)
        ax.set_xlabel(label)
    for ax in list(axes.flat)[len(ERROR_MODE_PANELS) :]:
        ax.set_visible(False)
    handles, labels = [], []
    for ax in axes.flat:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / fig.get_figheight()))
    return _save(fig, path)


def plot_paired_scatter(
    samples: pd.DataFrame,
    baseline: str,
    metrics: Sequence[str],
    stage: str,
    path: Path,
) -> Path:
    """Per sample, each variant against the baseline, one panel per metric."""
    plt = _pyplot()
    data = samples[samples["stage"] == stage]
    runs = [r for r in data["run"].drop_duplicates() if r != baseline]
    colours = run_colours([baseline, *runs])
    keys = ["subset", "seed"]
    base = data[data["run"] == baseline]
    metrics = [m for m in metrics if m in data]
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(2.8 * len(metrics), 3.0), squeeze=False
    )
    for ax, metric in zip(axes[0], metrics, strict=True):
        values = []
        for run in runs:
            joined = base[[*keys, metric]].merge(
                data[data["run"] == run][[*keys, metric]], on=keys
            )
            x, y = (
                joined[f"{metric}_x"].astype(float),
                joined[f"{metric}_y"].astype(float),
            )
            ax.plot(x, y, "o", color=colours[run], ms=4, alpha=0.8, label=run)
            values += [*x, *y]
        values = np.asarray(values, dtype=float)
        values = values[np.isfinite(values)]
        if len(values):
            low, high = values.min(), values.max()
            pad = 0.05 * (high - low or 1)
            ax.plot([low - pad, high + pad], [low - pad, high + pad],
                    color=TEXT_SECONDARY, lw=0.8)  # fmt: skip
        ax.set_title(metric)
        ax.set_xlabel(baseline)
    axes[0][0].set_ylabel("variant")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return _save(fig, path)


#: Trade-off panels: (x, y).
TRADEOFFS = (("n_break_ends", "n_spur_ends"), ("n_components", "junction_f1"))


def plot_tradeoff(
    samples: pd.DataFrame,
    order: Sequence[str],
    sweeps: dict[str, list[str]],
    stage: str,
    path: Path,
) -> Path:
    """Run means of two trade-offs; a sweep's runs are joined in order."""
    plt = _pyplot()
    data = samples[samples["stage"] == stage]
    colours = run_colours(order)
    means = data.groupby("run")[[c for pair in TRADEOFFS for c in pair]].mean()
    fig, axes = plt.subplots(1, len(TRADEOFFS), figsize=(8, 3.4), squeeze=False)
    for ax, (x, y) in zip(axes[0], TRADEOFFS, strict=True):
        for run in order:
            if run in means.index:
                ax.plot(means.loc[run, x], means.loc[run, y], "o",
                        color=colours[run], ms=7, label=run)  # fmt: skip
        for members in sweeps.values():
            members = [m for m in members if m in means.index]
            ax.plot(means.loc[members, x], means.loc[members, y], "-",
                    color=NEUTRAL, lw=1)  # fmt: skip
        ax.set_xlabel(f"{x} (mean per volume)")
        ax.set_ylabel(y)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4))
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    return _save(fig, path)


def plot_sweep(
    samples: pd.DataFrame,
    sweep: Sequence[tuple[str, float]],
    parameter: str,
    metrics: Sequence[str],
    stage: str,
    path: Path,
) -> Path:
    """Metrics against a swept parameter, one line per subset."""
    plt = _pyplot()
    data = samples[samples["stage"] == stage]
    value_of = dict(sweep)
    data = data[data["run"].isin(value_of)].assign(
        value=lambda f: f["run"].map(value_of).astype(float)
    )
    subsets = list(data["subset"].drop_duplicates())
    colours = {s: CATEGORICAL[i % len(CATEGORICAL)] for i, s in enumerate(subsets)}
    metrics = [m for m in metrics if m in data]
    fig, axes = plt.subplots(
        1, len(metrics), figsize=(2.8 * len(metrics), 3.0), squeeze=False
    )
    for ax, metric in zip(axes[0], metrics, strict=True):
        for subset in subsets:
            curve = data[data["subset"] == subset].groupby("value")[metric].mean()
            ax.plot(curve.index, curve.values, "-o", color=colours[subset], ms=3,
                    label=subset)  # fmt: skip
        ax.set_title(metric)
        ax.set_xlabel(parameter)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6))
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    return _save(fig, path)


def plot_flips(
    events: pd.DataFrame, errors: pd.DataFrame, stage: str, path: Path
) -> Path:
    """Fixed and new events (per type) and errors (per type), per variant."""
    plt = _pyplot()
    panels = []
    if len(events):
        part = events[(events["stage"] == stage) & (events["subset"] == POOLED)]
        panels.append(("events", part.rename(columns={"event_type": "kind"})))
    if len(errors):
        part = errors[errors["stage"] == stage]
        panels.append(("errors", part.rename(columns={"type": "kind"})))
    fig, axes = plt.subplots(
        1, max(len(panels), 1), figsize=(5.5 * max(len(panels), 1), 3.2), squeeze=False
    )
    for ax, (title, part) in zip(axes[0], panels, strict=False):
        runs = list(part["run"].drop_duplicates())
        kinds = list(part["kind"].drop_duplicates())
        width = 0.8 / max(len(runs), 1)
        for i, run in enumerate(runs):
            mine = part[part["run"] == run].groupby("kind")[["fixed", "new"]].sum()
            mine = mine.reindex(kinds, fill_value=0)
            x = np.arange(len(kinds)) + (i - (len(runs) - 1) / 2) * width
            ax.bar(
                x, mine["fixed"], width, color=BETTER, label="fixed" if i == 0 else None
            )
            ax.bar(x, -mine["new"], width, color=WORSE, label="new" if i == 0 else None)
        ax.axhline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_xticks(range(len(kinds)), kinds, rotation=30, ha="right")
        ax.set_ylabel("fixed (up) / new (down)")
        ax.set_title(f"{title} ({stage}): {', '.join(runs)}")
        ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def plot_loss_curves(histories: pd.DataFrame, path: Path) -> Path:
    """Median and interquartile range of each loss term over samples, per run.

    ``histories`` has ``run``, ``sample``, ``iteration`` and one column per
    term.
    """
    plt = _pyplot()
    runs = list(histories["run"].drop_duplicates())
    terms = [c for c in histories.columns if c not in ("run", "sample", "iteration")]
    colours = term_colours(terms)
    fig, axes = plt.subplots(
        1, len(runs), figsize=(3.6 * len(runs), 3.0), squeeze=False, sharey=True
    )
    for ax, run in zip(axes[0], runs, strict=True):
        part = histories[histories["run"] == run]
        for term in terms:
            grouped = part.groupby("iteration")[term]
            median = grouped.median()
            q = grouped.quantile([0.25, 0.75]).unstack()
            ax.plot(median.index, median.values, color=colours[term], lw=1.2,
                    label=term)  # fmt: skip
            ax.fill_between(q.index, q[0.25], q[0.75], color=colours[term],
                            alpha=0.15, lw=0)  # fmt: skip
        ax.set_yscale("log")
        ax.set_title(run)
        ax.set_xlabel("iteration")
    axes[0][0].set_ylabel("loss")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6))
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return _save(fig, path)


def plot_loss_vs_pipeline(table: pd.DataFrame, path: Path) -> Path:
    """Loss-correct against pipeline-correct, per family, against a covariate.

    ``table`` has ``family``, ``covariate`` (its name), ``x``, ``loss_correct``,
    ``pipeline_correct`` and ``cluster``, one row per probed event (see
    :func:`spleleton.bench.compare.loss_and_pipeline`). The gap between the
    two curves is what search and assembly lose.
    """
    plt = _pyplot()
    families = list(table["family"].drop_duplicates())
    fig, axes = plt.subplots(
        1, max(len(families), 1), figsize=(3.2 * max(len(families), 1), 3.0),
        squeeze=False,
    )  # fmt: skip
    for ax, family in zip(axes[0], families, strict=False):
        part = table[table["family"] == family]
        edges = _edges(part["x"], 4)
        for column, colour, label in (
            ("loss_correct", CATEGORICAL[0], "loss-correct"),
            ("pipeline_correct", TEXT, "pipeline-correct"),
        ):
            curve = binned(part, "x", column, edges)
            if curve.empty:
                continue
            ax.plot(curve["centre"], curve["mean"], "-o", color=colour, ms=3,
                    label=label)  # fmt: skip
            ax.fill_between(curve["centre"], curve["low"], curve["high"],
                            color=colour, alpha=0.12, lw=0)  # fmt: skip
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(f"{family} ({len(part)} events)")
        ax.set_xlabel(part["covariate"].iloc[0])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    return _save(fig, path)


# -- repairs (plan, Section 10.8) ------------------------------------------------------

#: The repair classes, in the order they are stacked, and their colours.
REPAIR_CLASSES = {
    "search": CATEGORICAL[0],
    "loss": CATEGORICAL[1],
    "assembly": CATEGORICAL[3],
    "undetermined": NEUTRAL,
    "none": GRID,
}


def plot_repair_classes(repairs: pd.DataFrame, path: Path) -> Path:
    """The share of each repair class, per error type and per cause.

    Parameters
    ----------
    repairs : pandas.DataFrame
        A run's ``repairs.parquet`` (several runs' rows may be stacked). Each
        error of a repairable type counts once, with its chosen repair's
        class (``none`` when it has none).
    path : Path
        The PNG to write.
    """
    from spleleton.bench.repair import REPAIRS

    plt = _pyplot()
    keys = [k for k in ("run", "subset", "seed", "error_id") if k in repairs]
    errors = repairs.drop_duplicates(keys)
    errors = errors[errors["error_type"].isin(list(REPAIRS))].assign(
        label=lambda f: f["error_class"].fillna("none"),
        cause_kind=lambda f: f["cause"].astype(str).str.split("#").str[0],
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), squeeze=False)
    for ax, column, title in zip(
        axes[0],
        ("error_type", "cause_kind"),
        ("by error type", "by cause"),
        strict=True,
    ):
        counts = (
            errors.groupby([column, "label"]).size().unstack(fill_value=0)
            if len(errors)
            else pd.DataFrame()
        )
        groups = list(counts.index)
        left = np.zeros(len(groups))
        totals = counts.sum(axis=1).to_numpy() if len(counts) else left
        for label, colour in REPAIR_CLASSES.items():
            if label not in counts:
                continue
            share = counts[label].to_numpy() / np.maximum(totals, 1)
            ax.barh(groups, share, left=left, color=colour, label=label, height=0.6)
            left = left + share
        for y, total in enumerate(totals):
            ax.text(1.01, y, f"n={int(total)}", va="center", fontsize=8,
                    color=TEXT_SECONDARY)  # fmt: skip
        ax.set_xlim(0, 1.12)
        ax.set_xlabel("share of errors")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.invert_yaxis()
    handles, labels = axes[0][0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[0][1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=len(labels))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(fig, path)


# -- sensitivity (plan, Section 11.7) --------------------------------------------------


def plot_threshold_curves(curves: pd.DataFrame, path: Path) -> Path:
    """Each threshold's step function (line) with the grid's rates (dots).

    Parameters
    ----------
    curves : pandas.DataFrame
        ``threshold``, ``outcome``, ``stage``, ``value``, ``rate`` and
        ``kind`` (``step`` for the step function, ``grid`` for the actual
        graph stage); the default is marked from ``default``.
    path : Path
        The PNG to write.
    """
    plt = _pyplot()
    panels = list(curves[["threshold"]].drop_duplicates()["threshold"])
    columns = min(3, len(panels)) or 1
    rows = math.ceil(len(panels) / columns) or 1
    fig, axes = plt.subplots(rows, columns, figsize=(3.6 * columns, 2.8 * rows),
                             squeeze=False)  # fmt: skip
    series = list(
        curves[["outcome", "stage"]].drop_duplicates().itertuples(index=False)
    )
    colours = {
        tuple(s): CATEGORICAL[i % len(CATEGORICAL)] for i, s in enumerate(series)
    }
    for ax, threshold in zip(axes.ravel(), panels, strict=False):
        part = curves[curves["threshold"] == threshold]
        for (outcome, stage), group in part.groupby(["outcome", "stage"]):
            colour = colours[(outcome, stage)]
            step = group[group["kind"] == "step"].sort_values("value")
            grid = group[group["kind"] == "grid"].sort_values("value")
            if len(step):
                ax.step(step["value"], step["rate"], where="post", color=colour,
                        lw=1.5, label=f"{outcome} ({stage})")  # fmt: skip
            if len(grid):
                ax.plot(grid["value"], grid["rate"], "o", color=colour, ms=4,
                        mec=SURFACE, mew=1, label=None if len(step) else
                        f"{outcome} ({stage})")  # fmt: skip
        default = part["default"].iloc[0]
        ax.axvline(default, color=TEXT_SECONDARY, lw=0.8, ls="--")
        ax.set_title(threshold)
        ax.set_ylabel("good rate")
        ax.set_ylim(-0.03, 1.03)
        ax.legend(fontsize=7)
    for ax in axes.ravel()[len(panels) :]:
        ax.set_visible(False)
    fig.tight_layout()
    return _save(fig, path)


def plot_tornado(elasticity: pd.DataFrame, path: Path) -> Path:
    """Pooled elasticities per parameter, one panel per outcome rate."""
    plt = _pyplot()
    part = elasticity[elasticity["subset"] == POOLED]
    outcomes = list(part["outcome"].drop_duplicates())
    fig, axes = plt.subplots(1, max(len(outcomes), 1),
                             figsize=(3.2 * max(len(outcomes), 1) + 1.5, 2.8),
                             sharey=True, squeeze=False)  # fmt: skip
    params = list(part["parameter"].drop_duplicates())
    labels = [p.removeprefix("fit.").removeprefix("fit.") for p in params]
    for ax, outcome in zip(axes[0], outcomes, strict=False):
        mine = part[part["outcome"] == outcome].set_index("parameter")
        for y, param in enumerate(params):
            if param not in mine.index:
                continue
            row = mine.loc[param]
            colour = BETTER if row["elasticity"] >= 0 else WORSE
            ax.barh(y, row["elasticity"], color=colour, height=0.6)
            if np.isfinite(row["low"]) and np.isfinite(row["high"]):
                ax.plot([row["low"], row["high"]], [y, y], color=TEXT, lw=1)
        ax.axvline(0, color=TEXT_SECONDARY, lw=0.8)
        ax.set_title(outcome)
        ax.set_xlabel("net good flips per +10%")
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(range(len(params)), labels)
    axes[0][0].invert_yaxis()
    fig.tight_layout()
    return _save(fig, path)


def plot_flip_distances(implicit: pd.DataFrame, path: Path) -> Path:
    """Histograms of |flip distance| (in log lambda) per parameter."""
    plt = _pyplot()
    params = list(implicit["parameter"].drop_duplicates())
    fig, axes = plt.subplots(1, max(len(params), 1),
                             figsize=(3.0 * max(len(params), 1), 2.6),
                             sharey=True, squeeze=False)  # fmt: skip
    bins = np.logspace(-2, 3, 21)
    for ax, param in zip(axes[0], params, strict=False):
        values = np.abs(implicit[implicit["parameter"] == param]["flip_distance"])
        values = values[np.isfinite(values)]
        ax.hist(np.clip(values, bins[0], bins[-1]), bins=bins, color=CATEGORICAL[0])
        ax.set_xscale("log")
        ax.axvline(math.log(1.1), color=TEXT_SECONDARY, lw=0.8, ls="--")
        ax.set_title(param.removeprefix("fit.").removeprefix("fit."), fontsize=8)
        ax.set_xlabel("|flip distance| (log units)")
    axes[0][0].set_ylabel("events")
    fig.tight_layout()
    return _save(fig, path)


def plot_validation(validation: pd.DataFrame, path: Path) -> Path:
    """Predicted against observed rate changes of the re-fits (pooled)."""
    plt = _pyplot()
    part = validation[validation["subset"] == POOLED]
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    params = list(part["parameter"].drop_duplicates())
    for i, param in enumerate(params):
        mine = part[part["parameter"] == param]
        ax.plot(mine["predicted"], mine["observed"], "o",
                color=CATEGORICAL[i % len(CATEGORICAL)], ms=6, mec=SURFACE,
                label=param.removeprefix("fit.").removeprefix("fit."))  # fmt: skip
    values = np.concatenate([part["predicted"], part["observed"]]) if len(part) else [0]
    lim = max(float(np.nanmax(np.abs(values))), 1e-3) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color=TEXT_SECONDARY, lw=0.8, ls="--")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("predicted change of the good rate")
    ax.set_ylabel("observed change")
    ax.legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, path)
