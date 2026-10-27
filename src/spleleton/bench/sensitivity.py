"""Sensitivity: which settings matter, how much, and for which error mode.

(Plan, Section 11.) Every discrete outcome is a continuous decision quantity
``q`` crossing a threshold ``tau`` (:mod:`spleleton.bench.decisions`). This
module turns ``q`` into sensitivities:

- **Graph-stage thresholds** (Section 11.3). Where a rule is one comparison,
  each event's **critical value** follows from its ``q`` in the cached fit
  (:func:`critical_rows`), so the outcome rates are exact step functions of
  the threshold. The thresholds interact, so the step functions are checked
  against the actual graph stage on a grid (:func:`grid_settings`: 9 values
  per threshold, and 2D grids for bridge reach x protrusion and join x
  prune), scoring only the outcomes a threshold can change (**restricted**).
- **Continuous fit parameters** (Section 11.4): implicit derivatives at the
  stored fit. With ``Phi`` the fit's own loss (the sparsity volumes held, the
  connection prior's final pair table), ``H`` its Hessian over the alive
  segments' 9 parameters each and ``b = d(grad Phi)/d log lambda``,
  ``dq/d log lambda = -grad q . (H + delta I)^-1 b``. ``H`` is explicit (one
  ``jvp`` of the gradient per parameter) up to :data:`EXPLICIT_MAX` alive
  segments, and MINRES with a group-wise Jacobi preconditioner above.
  :func:`flip_distance` and :func:`elasticities` turn derivatives into
  outcome sensitivities.
- **Validation** (Section 11.4): paired re-fits at ``lambda (1 +- f)``
  (:func:`validation_variant`, :func:`validation_rows`).
- **Data margins** (Section 11.6): per-bump critical heights within a sweep,
  and per-variant robustness margins (:func:`bump_critical_levels`,
  :func:`robustness_margins`).
- **Morris screening** (Section 11.5), optional: :func:`morris_design` and
  :func:`morris_effects`.
"""

import dataclasses
import itertools
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import minimum_spanning_tree

from spleleton.bench.stats import POOLED, cluster_bootstrap

# -- graph-stage thresholds ------------------------------------------------------------

#: Multiples of a default, for a threshold's grid.
FACTORS = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
#: The graph stage's thresholds (Section 11.1): where each lives in a
#: ``GraphConfig`` (``top``, ``params`` or ``clean``) and its grid of 9 values
#: (``None``: the default times :data:`FACTORS`).
THRESHOLDS: dict[str, tuple[str, tuple[float, ...] | None]] = {
    "join": ("top", (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)),
    "prune_threshold": ("top", (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 0.95)),
    "contained": ("params", None),
    "min_edge": ("params", None),
    "min_radius": ("params", (0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0, 1.25, 1.5)),
    "min_protrusion": ("clean", None),
    "bridge_reach": ("clean", None),
    "bridge_angle": ("clean", (10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 90.0)),
    "min_island": ("clean", None),
}
#: Interacting pairs, checked on 2D grids (Section 11.3).
PAIRS = (("bridge_reach", "min_protrusion"), ("join", "prune_threshold"))
#: The event types whose outcomes a threshold can change, and those outcomes.
RESTRICTED = {
    "junction": "connected",
    "window": "connected",
    "bump": "spur",
    "protrusion": "spur",
}
#: Thresholds with a step function (a critical value from ``q``).
STEPPED = ("join", "prune_threshold", "bridge_reach", "bridge_angle", "min_protrusion")
#: The step functions must agree with the grid on this share of the points.
GRID_AGREEMENT = 0.95


def threshold_value(config, name: str) -> float:
    """A threshold's value in a ``GraphConfig`` (its method's default if unset)."""
    import inspect

    from spleleton.assemble import assemble_graph

    where, _ = THRESHOLDS[name]
    if where == "top":
        return float(getattr(config, name))
    if where == "clean":
        return float(getattr(config.clean, name))
    params = dict(config.params)
    if name in params:
        return float(params[name])
    return float(inspect.signature(assemble_graph).parameters[name].default)


def graph_with(config, values: dict[str, float]):
    """A ``GraphConfig`` with some thresholds changed."""
    top = {k: v for k, v in values.items() if THRESHOLDS[k][0] == "top"}
    clean = {k: v for k, v in values.items() if THRESHOLDS[k][0] == "clean"}
    params = dict(config.params)
    params |= {k: v for k, v in values.items() if THRESHOLDS[k][0] == "params"}
    return dataclasses.replace(
        config,
        **top,
        params=tuple(sorted(params.items())),
        clean=dataclasses.replace(config.clean, **clean),
    )


def grid_values(config, name: str) -> tuple[float, ...]:
    """A threshold's 9 grid values (its default among them)."""
    _, grid = THRESHOLDS[name]
    if grid is not None:
        return grid
    default = threshold_value(config, name)
    return tuple(default * f for f in FACTORS)


def grid_settings(config, full: bool = False) -> list[dict[str, float]]:
    """The grid's settings: each threshold's line and the pairs' 2D grids.

    Each setting holds the thresholds that differ from the defaults (the
    defaults themselves once, as ``{}``), in a fixed order, without repeats.
    ``full`` changes nothing here; it is the scoring that differs.
    """
    defaults = {name: threshold_value(config, name) for name in THRESHOLDS}
    settings: list[dict[str, float]] = [{}]
    seen = {()}

    def add(values: dict[str, float]) -> None:
        changed = {
            k: float(v) for k, v in values.items() if not np.isclose(v, defaults[k])
        }
        key = tuple(sorted(changed.items()))
        if key not in seen:
            seen.add(key)
            settings.append(changed)

    for name in THRESHOLDS:
        for value in grid_values(config, name):
            add({name: value})
    for a, b in PAIRS:
        for va, vb in itertools.product(grid_values(config, a), grid_values(config, b)):
            add({a: va, b: vb})
    return settings


def critical_rows(events: pd.DataFrame) -> pd.DataFrame:
    """Each event's critical value per stepped threshold (Section 11.3).

    From the fit-stage decision quantities of a run's events table (raw-stage
    rows; they are repeated in both stages). ``direction`` says on which side
    of the critical value the outcome holds, and ``stage`` which graph the
    threshold acts on.
    """
    raw = events[events["stage"] == "raw"]
    rows = []
    keys = ["subset", "seed", "event_type", "event_id"]
    for row in raw.itertuples():
        base = {k: getattr(row, k) for k in keys}
        if row.event_type in ("junction", "window") and _finite(row, "q_join"):
            rows += [
                {
                    **base,
                    "threshold": "join",
                    "critical": row.q_join,
                    "outcome": "connected",
                    "direction": "above",
                    "stage": "raw",
                },
                {
                    **base,
                    "threshold": "prune_threshold",
                    "critical": row.q_prune_connect,
                    "outcome": "connected",
                    "direction": "at or below",
                    "stage": "raw",
                },
            ]
            if _finite(row, "q_bridge"):
                rows += [
                    {
                        **base,
                        "threshold": "bridge_reach",
                        "critical": row.q_bridge,
                        "outcome": "connected",
                        "direction": "at or above",
                        "stage": "clean",
                    },
                    {
                        **base,
                        "threshold": "bridge_angle",
                        "critical": row.bridge_angle,
                        "outcome": "connected",
                        "direction": "at or above",
                        "stage": "clean",
                    },
                ]
        if row.event_type in ("bump", "protrusion") and _finite(row, "q_prune_spur"):
            rows.append(
                {
                    **base,
                    "threshold": "prune_threshold",
                    "critical": row.q_prune_spur,
                    "outcome": "spur",
                    "direction": "at or below",
                    "stage": "raw",
                }
            )
            if _finite(row, "q_protrusion"):
                rows.append(
                    {
                        **base,
                        "threshold": "min_protrusion",
                        "critical": row.q_protrusion,
                        "outcome": "spur",
                        "direction": "at or below",
                        "stage": "clean",
                    }
                )
    return pd.DataFrame(rows)


def _finite(row, name: str) -> bool:
    value = getattr(row, name, np.nan)
    return value is not None and bool(np.isfinite(value))


def predict(events: pd.DataFrame, values: dict[str, float], config) -> pd.DataFrame:
    """The step functions' outcomes at a setting, per event and stage.

    Raw ``connected``: ``q_join < join`` and ``prune <= q_prune_connect``.
    Clean ``connected``: raw connected, or bridged (``q_bridge <= reach``
    and ``bridge_angle <= angle``). Raw ``spur``: ``prune <= q_prune_spur``.
    Clean ``spur``: raw spur and ``q_protrusion >= min_protrusion``.
    """
    setting = {name: threshold_value(config, name) for name in THRESHOLDS} | values
    raw = events[events["stage"] == "raw"].set_index(
        ["subset", "seed", "event_type", "event_id"]
    )
    out = []
    connection = raw[
        raw.index.get_level_values("event_type").isin(["junction", "window"])
    ]
    if len(connection):
        q = connection["q_join"].astype(float)
        linked = (q < setting["join"]) & (
            setting["prune_threshold"] <= connection["q_prune_connect"].astype(float)
        )
        bridge = connection["q_bridge"].astype(float)
        angle = connection["bridge_angle"].astype(float)
        bridged = (bridge <= setting["bridge_reach"]) & (
            angle <= setting["bridge_angle"]
        )
        out.append(
            pd.DataFrame({"raw": linked, "clean": linked | bridged.fillna(False)})
        )
    spurs = raw[raw.index.get_level_values("event_type").isin(["bump", "protrusion"])]
    if len(spurs):
        spur = setting["prune_threshold"] <= spurs["q_prune_spur"].astype(float)
        kept = spurs["q_protrusion"].astype(float) >= setting["min_protrusion"]
        out.append(
            pd.DataFrame(
                {"raw": spur.fillna(False), "clean": (spur & kept).fillna(False)}
            )
        )
    if not out:
        return pd.DataFrame()
    frame = pd.concat(out).reset_index()
    return frame.melt(
        id_vars=["subset", "seed", "event_type", "event_id"],
        value_vars=["raw", "clean"],
        var_name="stage",
        value_name="predicted",
    )


def restricted_outcomes(sample, segments, config, true=None, events=None):
    """The graph stage at a setting, scored only for the outcomes it can change.

    Returns one row per (event, stage) of the junctions, windows, bumps and
    protrusions: ``connected`` and ``anchors_matched``, or ``spur``.
    """
    from spleleton.bench.events import (
        TrueCentreline,
        extract_events,
        stage_outcomes,
        stage_view,
    )
    from spleleton.pipeline import graph_stage

    true = true or TrueCentreline(sample.graph)
    events = events if events is not None else extract_events(sample, true)
    events = [e for e in events if e.event_type in RESTRICTED]
    raw, clean = graph_stage(segments, None, sample.mask, sample.spacing, config)
    rows = []
    for stage, graph in (("raw", raw), ("clean", clean)):
        view = stage_view(graph, true)
        for event in events:
            outcome = stage_outcomes(event, view, true)
            rows.append(
                {
                    "event_type": event.event_type,
                    "event_id": event.event_id,
                    "stage": stage,
                    "actual": bool(outcome[RESTRICTED[event.event_type]]),
                    "anchors_matched": bool(outcome.get("anchors_matched", True)),
                }
            )
    return rows


def grid_rows(sample, segments, config, settings, progress=None) -> pd.DataFrame:
    """The restricted outcomes at every grid setting, for one sample."""
    from spleleton.bench.events import TrueCentreline, extract_events

    true = TrueCentreline(sample.graph)
    events = extract_events(sample, true)
    rows = []
    for index, values in enumerate(settings):
        started = time.perf_counter()
        part = restricted_outcomes(
            sample, segments, graph_with(config, values), true, events
        )
        elapsed = time.perf_counter() - started
        label = _setting_label(values)
        for row in part:
            rows.append(
                {
                    **row,
                    "setting": index,
                    "label": label,
                    **{f"v_{k}": v for k, v in values.items()},
                    "time_s": elapsed,
                }
            )
        if progress and index % 50 == 0:
            progress(f"    grid {index + 1}/{len(settings)} ({elapsed:.3f} s)")
    return pd.DataFrame(rows)


def full_grid_rows(sample, segments, config, settings) -> pd.DataFrame:
    """Every setting scored in full: the samples table's rows (``--full-grid``).

    For the threshold curves of the guard metrics (components, cycles, F1);
    optional, since it scores everything at every setting (Section 11.3).
    """
    from spleleton.bench.events import score_sample
    from spleleton.pipeline import graph_stage

    rows = []
    for index, values in enumerate(settings):
        changed = graph_with(config, values)
        raw, clean = graph_stage(segments, None, sample.mask, sample.spacing, changed)
        scores = score_sample(sample, raw, clean, segments, changed)
        label = _setting_label(values)
        for row in scores.samples:
            plain = {k: v for k, v in row.items() if np.isscalar(v) or v is None}
            rows.append({**plain, "setting": index, "label": label,
                         **{f"v_{k}": v for k, v in values.items()}})  # fmt: skip
    return pd.DataFrame(rows)


def _setting_label(values: dict[str, float]) -> str:
    return ",".join(f"{k}={v:g}" for k, v in sorted(values.items())) or "default"


def grid_agreement(grid: pd.DataFrame, events: pd.DataFrame, settings, config):
    """The step functions against the grid, per threshold line and pair grid.

    Only settings whose changed thresholds all have step functions are
    compared; connection outcomes only where both anchors matched (as P3).

    Returns
    -------
    pandas.DataFrame
        Per ``line`` (a threshold, or ``a x b``) and stage: the points
        compared and the share that agree.
    """
    rows = []
    keys = ["subset", "seed", "event_type", "event_id", "stage"]
    for index, values in enumerate(settings):
        if not set(values) <= set(STEPPED):
            continue
        line = (
            " x ".join(sorted(values))
            if len(values) > 1
            else (next(iter(values)) if values else "default")
        )
        predicted = predict(events, values, config)
        if predicted.empty:
            continue
        actual = grid[grid["setting"] == index]
        joined = actual.merge(predicted, on=keys)
        joined = joined[joined["anchors_matched"]]
        for stage, part in joined.groupby("stage"):
            rows.append(
                {
                    "line": line,
                    "setting": index,
                    "stage": stage,
                    "points": len(part),
                    "agree": int((part["actual"] == part["predicted"]).sum()),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    table = frame.groupby(["line", "stage"])[["points", "agree"]].sum().reset_index()
    table["share"] = table["agree"] / table["points"].clip(lower=1)
    return table


# -- the fit's loss at a stored fit (JAX, float64) -------------------------------------

#: Parameter groups per segment (for the preconditioner).
GROUPS = ("position", "radius", "opacity")
#: Damping relative to the largest eigenvalue; the first is the result.
DELTAS = (1e-5, 1e-4, 1e-6)
#: Above this many alive segments, MINRES replaces the explicit Hessian.
EXPLICIT_MAX = 95
#: MINRES's tolerance and iteration cap.
MINRES_TOL, MINRES_MAX = 1e-9, 600
#: A ``|dq/d log lambda|`` below this is no steering (an unsteerable event).
DQ_FLOOR = 1e-5
#: The +10% of an elasticity, in log units.
ELASTICITY_STEP = math.log(1.1)
#: Products per ``lax.map`` batch of the explicit Hessian (one at a time: P9
#: found batches of 8 under ``vmap`` about 4 times slower in all).
HESSIAN_BATCH = 1


@dataclass
class Problem:
    """A stored fit's loss as a function of its alive segments' parameters.

    Attributes
    ----------
    x : jax.Array
        The alive segments' parameters, flattened (``2 D + 3`` per segment).
    free : np.ndarray
        The alive segments' indices.
    base : JaxLineSegments
        Every segment (float64); the dead ones are held.
    states, target, temperature, static
        The fit's last step: prior states (the pair table), its target, the
        render's temperature and the static settings of ``_loss``.
    weights : dict
        The fit's loss weights by term, and ``connection``.
    width : float
        The render kernel's width (``wall_width`` or ``sigma``).
    """

    x: Any
    free: np.ndarray
    base: Any
    states: tuple
    target: Any
    temperature: Any
    static: dict[str, Any]
    weights: dict[str, float]
    width: float
    width_name: str

    @property
    def per_segment(self) -> int:
        """Parameters per segment."""
        return 2 * self.base.start.shape[1] + 3

    def unflatten(self, x):
        """Every segment, the alive ones taken from ``x``."""
        from spleleton.primitives._primitives_jax import JaxLineSegments

        d = self.base.start.shape[1]
        x = x.reshape(-1, self.per_segment)
        free = self.free
        base = self.base
        return JaxLineSegments(
            start=base.start.at[free].set(x[:, 0:d]),
            end=base.end.at[free].set(x[:, d : 2 * d]),
            log_radius=base.log_radius.at[free].set(x[:, 2 * d : 2 * d + 2]),
            opacity_logit=base.opacity_logit.at[free].set(x[:, 2 * d + 2]),
        )

    def loss(self, x, width=None):
        """``Phi`` and the terms (the fit's ``_loss``), with an optional width."""
        from spleleton.fit._fit_jax import _loss

        value, (terms, bins) = _loss(
            self.unflatten(x),
            self.states,
            self.target,
            self.temperature,
            width=width,
            **self.static,
        )
        return value, (terms, bins.overflow)

    def groups(self) -> np.ndarray:
        """Per parameter, its group index in :data:`GROUPS`."""
        d = self.base.start.shape[1]
        one = np.array([0] * (2 * d) + [1, 1] + [2])
        return np.tile(one, len(self.free))


def make_problem(variant, sample, record, alive: np.ndarray) -> Problem:
    """The loss of a stored fit (a :class:`~spleleton.bench.runner.FitRecord`)."""
    import jax.numpy as jnp

    from spleleton.bench.runner import _field, _regularisers
    from spleleton.fit._fit_jax import _check, _settings
    from spleleton.primitives._primitives_jax import JaxLineSegments
    from spleleton.render._render_jax import BinCapacity, _static

    fit = variant.fit.fit
    segments = record.segments
    base = JaxLineSegments(
        **{
            k: jnp.asarray(np.asarray(getattr(segments, k), dtype=np.float64))
            for k in ("start", "end", "log_radius", "opacity_logit")
        }
    )
    target = jnp.asarray(_field(variant, sample.mask, sample.spacing), jnp.float64)
    shape, spacing = _check(base, target.shape, sample.spacing)
    regularisers = tuple(_regularisers(variant, sample.mask.size, record.layout))
    states = tuple(
        tuple(jnp.asarray(np.asarray(a)) for a in state)
        if isinstance(state, (tuple, list))
        else state
        for state in record.prior_states
    )
    render = record.render_config
    static = {
        "shape": shape,
        "spacing": spacing,
        "render_config": _static(render),
        "capacity": BinCapacity(*record.capacity),
        "settings": _settings(fit, spacing),
        "regularisers": regularisers,
    }
    free = np.flatnonzero(alive)
    d = base.start.shape[1]
    x = jnp.concatenate(
        [
            base.start[free],
            base.end[free],
            base.log_radius[free],
            base.opacity_logit[free][:, None],
        ],
        axis=1,
    ).ravel()
    weights = {term.name: float(term.weight) for term in fit.losses}
    if variant.pipeline == "scratch_fit":
        weights["connection"] = float(variant.fit.connection_weight)
    width_name = "wall_width" if render.mode == "tube" else "sigma"
    assert x.shape[0] == len(free) * (2 * d + 3)
    return Problem(
        x,
        free,
        base,
        states,
        target,
        jnp.asarray(render.temperature, jnp.float64),
        static,
        weights,
        float(getattr(render, width_name)),
        width_name,
    )


def gradient(problem: Problem, x, width=None):
    """The field the fit follows (the gradient of ``Phi``) and the overflow."""
    import jax

    grad, (_, overflow) = jax.grad(lambda y: problem.loss(y, width), has_aux=True)(x)
    return grad, overflow


def hessian_explicit(problem: Problem, x, batch: int = HESSIAN_BATCH):
    """``H``, one ``jvp`` of the gradient per parameter (``lax.map``)."""
    import jax
    import jax.numpy as jnp

    def field(y):
        return gradient(problem, y)[0]

    def column(v):
        return jax.jvp(field, (x,), (v,))[1]

    run = jax.jit(
        lambda: jax.lax.map(
            column, jnp.eye(x.shape[0], dtype=x.dtype), batch_size=batch
        )
    )
    return np.asarray(run())


def hvp_function(problem: Problem, x) -> Callable[[np.ndarray], np.ndarray]:
    """``v -> H v`` (numpy in and out), compiled once."""
    import jax
    import jax.numpy as jnp

    def field(y):
        return gradient(problem, y)[0]

    product = jax.jit(lambda v: jax.jvp(field, (x,), (v,))[1])
    return lambda v: np.asarray(product(jnp.asarray(v, dtype=x.dtype)))


def mixed_terms(problem: Problem, x, params: Sequence[str]) -> dict[str, np.ndarray]:
    """``b = d(grad Phi) / d log lambda`` per parameter.

    A loss weight's is ``w_k grad L_k`` (its term unweighted); the connection
    weight's is the prior's term's gradient (the prior returns it weighted);
    the kernel width's is ``width * jvp`` of the gradient along the width.
    """
    import jax
    import jax.numpy as jnp

    out = {}
    names = [_param_term(p, problem) for p in params]
    terms_wanted = [n for n in names if n not in (None, "width")]
    if terms_wanted:

        def terms(y):
            _, (values, _) = problem.loss(y)
            return jnp.stack([values[name] for name in terms_wanted])

        jacobian = np.asarray(jax.jit(jax.jacrev(terms))(x))
        for name, row in zip(terms_wanted, jacobian, strict=True):
            weight = 1.0 if name == "connection" else problem.weights[name]
            out[name] = weight * row
    result = {}
    for param, name in zip(params, names, strict=True):
        if name == "width":
            width = jnp.asarray(problem.width, dtype=x.dtype)

            def along(w):
                return gradient(problem, x, w)[0]

            def tangent(w):
                return jax.jvp(along, (w,), (jnp.ones_like(w),))[1]

            result[param] = problem.width * np.asarray(jax.jit(tangent)(width))
        elif name is None:
            raise ValueError(f"no mixed term for {param!r}")
        else:
            result[param] = out[name]
    return result


def _param_term(param: str, problem: Problem) -> str | None:
    """The loss term (or ``width``) a parameter path names."""
    if param == "fit.connection_weight":
        return "connection"
    if param in ("fit.fit.render.wall_width", "fit.fit.render.sigma"):
        if not param.endswith(problem.width_name):
            raise ValueError(f"{param} does not act in {problem.width_name}'s mode")
        return "width"
    parts = param.split(".")
    if parts[:3] == ["fit", "fit", "losses"] and parts[-1] == "weight":
        name = parts[3]
        if name not in problem.weights:
            raise ValueError(f"the fit has no loss term {name!r}")
        return name
    return None


def width_central_difference(problem: Problem, x, step: float = 1e-4) -> np.ndarray:
    """Central differences of the gradient along the width (``d/d log w``)."""
    import jax.numpy as jnp

    w = problem.width
    plus = np.asarray(gradient(problem, x, jnp.asarray(w * (1 + step)))[0])
    minus = np.asarray(gradient(problem, x, jnp.asarray(w * (1 - step)))[0])
    return (plus - minus) / (2 * step)


def spectrum(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Eigendecomposition of the symmetric part, and a summary of it."""
    asymmetry = float(np.linalg.norm(matrix - matrix.T) / np.linalg.norm(matrix))
    eigenvalues, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    top = float(eigenvalues[-1])
    return (
        eigenvalues,
        vectors,
        {
            "asymmetry": asymmetry,
            "largest": top,
            "smallest": float(eigenvalues[0]),
            "negative": int((eigenvalues < 0).sum()),
            "below_1e-4": int((eigenvalues < 1e-4 * top).sum()),
            "below_1e-6": int((eigenvalues < 1e-6 * top).sum()),
        },
    )


def explicit_solve(eigenvalues, vectors, b, delta) -> np.ndarray:
    """``(H + delta I)^-1 b`` from the eigendecomposition of ``H``."""
    return vectors @ ((vectors.T @ b) / (eigenvalues + delta))


def group_diagonal(apply, groups: np.ndarray, seed: int = 0) -> np.ndarray:
    """A Jacobi preconditioner per parameter group, from one product per group.

    With a Rademacher vector ``v`` on a group, ``v . H v / |v|^2`` is the
    group's mean diagonal plus noise from the off-diagonal entries.
    """
    rng = np.random.default_rng(seed)
    diagonal = np.ones(len(groups))
    for group in np.unique(groups):
        mask = groups == group
        v = np.where(mask, rng.choice([-1.0, 1.0], size=len(groups)), 0.0)
        diagonal[mask] = max(abs(float(v @ apply(v) / (v @ v))), 1e-12)
    return diagonal


def largest_eigenvalue(apply, n: int, iterations: int = 30, seed: int = 0) -> float:
    """The largest eigenvalue of ``H`` by power iteration (for MINRES's damping)."""
    v = np.random.default_rng(seed).normal(size=n)
    v /= np.linalg.norm(v)
    value = 0.0
    for _ in range(iterations):
        w = apply(v)
        value = float(v @ w)
        v = w / max(np.linalg.norm(w), 1e-300)
    return abs(value)


def solve_minres(apply, b, delta, precondition) -> dict[str, Any]:
    """MINRES on ``(H + delta I) u = b`` (symmetric, not definite)."""
    from scipy.sparse.linalg import LinearOperator, minres

    n = len(b)
    operator = LinearOperator((n, n), matvec=apply, dtype=np.float64)
    inverse = LinearOperator((n, n), matvec=lambda v: v / precondition)
    iterations = [0]

    def count(_):
        iterations[0] += 1

    u, info = minres(
        operator,
        b,
        rtol=MINRES_TOL,
        shift=-delta,
        maxiter=MINRES_MAX,
        M=inverse,
        callback=count,
    )
    residual = np.linalg.norm(apply(u) + delta * u - b) / max(np.linalg.norm(b), 1e-300)
    return {
        "u": u,
        "iterations": iterations[0],
        "converged": info == 0,
        "residual": float(residual),
    }


# -- decision quantities in JAX (Section 11.2) -----------------------------------------


def link_form(links, u: int, v: int) -> dict[str, Any]:
    """The form of the link between candidates ``u`` and ``v`` that sets its ratio.

    ``ee``: endpoint ``ka`` of ``a`` to endpoint ``kb`` of ``b``; ``te``: end
    ``ka`` of ``a`` to the interior of ``b`` (as ``segment_links`` computes
    them). Indices are segment indices.
    """
    arrays = links.segments
    seg = links.candidate
    ends = np.stack([arrays.start, arrays.end], axis=1)
    radius = arrays.radius
    forms = []
    for a, b in ((u, v), (v, u)):
        sa, sb = seg[a], seg[b]
        for ka in range(2):
            for kb in range(2):
                d = np.linalg.norm(ends[sa, ka] - ends[sb, kb])
                forms.append(
                    (d / max(radius[sa, ka], radius[sb, kb]), "ee", sa, ka, sb, kb)
                )
            p = ends[sa, ka]
            s, vector = arrays.start[sb], arrays.end[sb] - arrays.start[sb]
            length = np.linalg.norm(vector)
            t = np.clip((p - s) @ vector / max(length**2, 1e-9), 0, 1)
            rj = radius[sb, 0] + t * (radius[sb, 1] - radius[sb, 0])
            along = t * length
            if rj < along < length - rj:
                d = np.linalg.norm(p - (s + t * vector))
                forms.append((d / max(rj, radius[sa, ka]), "te", sa, ka, sb, 0))
    value, kind, sa, ka, sb, kb = min(forms, key=lambda f: f[0])
    if not math.isclose(value, links.ratio[u, v], rel_tol=1e-9, abs_tol=1e-12):
        raise RuntimeError(f"link ({u}, {v}): no form gives its ratio")
    return {"kind": kind, "a": int(sa), "ka": int(ka), "b": int(sb), "kb": int(kb)}


def _spanning_tree(links) -> np.ndarray:
    """The minimax spanning tree behind ``Links.minimax`` (shifted weights)."""
    alive = links.alive
    weight = np.where(alive[:, None] & alive[None], links.ratio, np.inf)
    dense = np.where(np.isfinite(weight), weight + 1.0, 0.0)
    tree = minimum_spanning_tree(dense).toarray()
    return np.maximum(tree, tree.T)


def _tree_path(tree: np.ndarray, source: int, target: int) -> list[tuple[int, int]]:
    parent = {source: -1}
    stack = [source]
    while stack:
        u = stack.pop()
        for v in np.flatnonzero(tree[u]):
            if int(v) not in parent:
                parent[int(v)] = u
                stack.append(int(v))
    path, node = [], target
    while parent.get(node, -1) != -1:
        path.append((parent[node], node))
        node = parent[node]
    return path


def critical_links(sample, fit, events: list) -> list[dict[str, Any]]:
    """Each event's critical quantity at the stored fit: its selection and ``q``.

    Connection events (junction branches, windows): the anchors' covering
    segments, the minimax chain between them in the spanning tree, and its
    weakest link, with ``tau = join`` (Section 11.2); events whose
    ``q_join`` is 0 (one segment covers both anchors) or infinite have none.
    Bumps and protrusions: the spur segment of largest opacity
    (``q_prune_spur``, ``tau = prune_threshold``). In both, ``q < tau`` is the
    good outcome (connected; no spur).
    """
    from spleleton.bench.decisions import _first_covering, connection_quantities
    from spleleton.bench.events import (
        TrueCentreline,
        _anchor_reach,
        _positions,
        _spur_place,
        spur_segments,
        tolerance,
    )

    true = TrueCentreline(sample.graph)
    links = fit.links
    tree = _spanning_tree(links)
    out = []
    for event in events:
        base = {"event_id": event.event_id, "event_type": event.event_type}
        if event.event_type in ("bump", "protrusion"):
            centre, _, reach = _spur_place(event, true)
            spur = _positions(fit, spur_segments(fit, centre, reach))
            if not len(spur):
                continue
            k = int(spur[np.argmax(links.opacity[spur])])
            out.append(
                {
                    **base,
                    "q_numpy": float(links.opacity[k]),
                    "kind": "opacity",
                    "tau_name": "prune_threshold",
                    "a": int(links.candidate[k]),
                    "ka": 0,
                    "b": 0,
                    "kb": 0,
                }
            )
            continue
        if event.event_type not in ("junction", "window"):
            continue
        a_index, b_index = event.anchors
        if event.event_type == "window":
            reach = _anchor_reach(event)
            a_list, b_list = event.region[:reach], event.region[::-1][:reach]
        else:
            a_list, b_list = [a_index], [b_index]
        anchors_a = [(true.points[i], float(tolerance(true.radii[i]))) for i in a_list]
        anchors_b = [(true.points[i], float(tolerance(true.radii[i]))) for i in b_list]
        q = connection_quantities(links, anchors_a, anchors_b)["q_join"]
        if not (np.isfinite(q) and q > 0):
            continue
        a = _first_covering(links, anchors_a, True)
        b = _first_covering(links, anchors_b, True)
        block = links.minimax[np.ix_(a, b)]
        i, j = np.unravel_index(np.argmin(block), block.shape)
        path = _tree_path(tree, int(a[i]), int(b[j]))
        if not path:
            continue
        u, v = max(path, key=lambda edge: tree[edge])
        if not math.isclose(tree[u, v] - 1.0, q, rel_tol=1e-9, abs_tol=1e-12):
            raise RuntimeError(f"{event.event_id}: the path's bottleneck is not q")
        out.append(
            {**base, "q_numpy": float(q), "tau_name": "join", **link_form(links, u, v)}
        )
    return out


def q_jax(problem: Problem, x, selected: list[dict[str, Any]]):
    """The selected quantities at ``x`` (the selection fixed): links and opacities."""
    import jax
    import jax.numpy as jnp

    seg = problem.unflatten(x)
    ends = jnp.stack([seg.start, seg.end], axis=1)
    radius = jnp.exp(seg.log_radius)
    a = jnp.asarray([r["a"] for r in selected])
    ka = jnp.asarray([r["ka"] for r in selected])
    b = jnp.asarray([r["b"] for r in selected])
    kb = jnp.asarray([r["kb"] for r in selected])
    te = jnp.asarray([r["kind"] == "te" for r in selected])
    opacity = jnp.asarray([r["kind"] == "opacity" for r in selected])
    p, rp = ends[a, ka], radius[a, ka]
    ee = jnp.linalg.norm(p - ends[b, kb], axis=-1) / jnp.maximum(rp, radius[b, kb])
    s, vector = seg.start[b], seg.end[b] - seg.start[b]
    squared = jnp.maximum((vector**2).sum(-1), 1e-30)
    t = jnp.clip(((p - s) * vector).sum(-1) / squared, 0.0, 1.0)
    rj = radius[b, 0] + t * (radius[b, 1] - radius[b, 0])
    tee = jnp.linalg.norm(p - (s + t[:, None] * vector), axis=-1) / jnp.maximum(rj, rp)
    link = jnp.where(te, tee, ee)
    return jnp.where(opacity, jax.nn.sigmoid(seg.opacity_logit[a]), link)


# -- one sample's implicit derivatives -------------------------------------------------


def implicit_sample(
    variant,
    sample,
    record,
    params: Sequence[str],
    method: str | None = None,
    check_width: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Implicit derivatives of a stored fit's decision quantities.

    Parameters
    ----------
    variant : Variant
        The run's variant (resolved).
    sample : Sample
        The volume.
    record : FitRecord
        The stored fit.
    params : sequence of str
        Parameter paths: ``fit.connection_weight``,
        ``fit.fit.losses.<term>.weight``, ``fit.fit.render.wall_width``.
    method : {"explicit", "minres", "both"}, optional
        By default explicit up to :data:`EXPLICIT_MAX` alive segments.
    check_width : bool
        Also compare the width's mixed term with central differences.
    progress : callable, optional
        Called with a summary line.

    Returns
    -------
    dict
        ``rows`` (per event and parameter: ``q``, ``tau``, ``dq`` at each
        damping, and with ``both`` the MINRES ``dq`` too), ``solves`` (per
        parameter and damping: the solver's diagnostics) and ``info`` (the
        field norm, the spectrum, the JAX against numpy ``q``, timings).
    """
    import jax
    import jax.numpy as jnp

    from spleleton.bench.events import TrueCentreline, extract_events, fit_view

    started = time.perf_counter()
    with jax.enable_x64(True):
        true = TrueCentreline(sample.graph)
        fit = fit_view(true, record.segments, variant.graph)
        problem = make_problem(variant, sample, record, fit.alive)
        x = problem.x
        n = int(x.shape[0])
        grad, overflow = jax.jit(lambda y: gradient(problem, y))(x)
        if bool(overflow):
            raise RuntimeError("the fit's bin capacity overflowed in float64")
        field_norm = float(jnp.linalg.norm(grad))
        selected = critical_links(sample, fit, extract_events(sample, true))
        info: dict[str, Any] = {
            "alive": len(problem.free),
            "parameters": n,
            "field_norm": field_norm,
            "events": len(selected),
        }
        if not selected:
            info["time_s"] = time.perf_counter() - started
            return {"rows": [], "solves": [], "info": info}
        q = np.asarray(q_jax(problem, x, selected))
        q_numpy = np.array([r["q_numpy"] for r in selected])
        info["q_error"] = float(np.abs(q - q_numpy).max())
        grad_q = np.asarray(
            jax.jit(jax.jacrev(lambda y: q_jax(problem, y, selected)))(x)
        )
        b = mixed_terms(problem, x, params)
        if check_width and any(_param_term(p, problem) == "width" for p in params):
            param = next(p for p in params if _param_term(p, problem) == "width")
            fd = width_central_difference(problem, x)
            info["width_error"] = float(
                np.linalg.norm(b[param] - fd) / max(np.linalg.norm(fd), 1e-300)
            )
        method = method or (
            "explicit" if len(problem.free) <= EXPLICIT_MAX else "minres"
        )
        dq: dict[tuple[str, str], np.ndarray] = {}
        solves = []
        if method in ("explicit", "both"):
            t0 = time.perf_counter()
            matrix = hessian_explicit(problem, x)
            eigenvalues, vectors, summary = spectrum(matrix)
            info["spectrum"] = summary
            info["hessian_s"] = time.perf_counter() - t0
            top = summary["largest"]
            for param, rhs in b.items():
                for delta in DELTAS:
                    u = explicit_solve(eigenvalues, vectors, rhs, delta * top)
                    dq[param, f"explicit/{delta:g}"] = -grad_q @ u
                    solves.append(
                        {
                            "parameter": param,
                            "solver": "explicit",
                            "delta": delta,
                            "residual": float(
                                np.linalg.norm(matrix @ u + delta * top * u - rhs)
                                / max(np.linalg.norm(rhs), 1e-300)
                            ),
                        }
                    )
        if method in ("minres", "both"):
            t0 = time.perf_counter()
            apply = hvp_function(problem, x)
            top = info.get("spectrum", {}).get("largest") or largest_eigenvalue(
                apply, n
            )
            precondition = group_diagonal(apply, problem.groups())
            for param, rhs in b.items():
                for delta in DELTAS[:1] if method == "both" else DELTAS:
                    result = solve_minres(
                        apply, rhs, delta * top, precondition + delta * top
                    )
                    dq[param, f"minres/{delta:g}"] = -grad_q @ result["u"]
                    solves.append(
                        {
                            "parameter": param,
                            "solver": "minres",
                            "delta": delta,
                            "iterations": result["iterations"],
                            "converged": result["converged"],
                            "residual": result["residual"],
                        }
                    )
            info["minres_s"] = time.perf_counter() - t0
        default = f"{method if method != 'both' else 'explicit'}/{DELTAS[0]:g}"
        taus = {
            "join": float(variant.graph.join),
            "prune_threshold": float(variant.graph.prune_threshold),
        }
        rows = []
        for k, link in enumerate(selected):
            tau = taus[link["tau_name"]]
            for param in params:
                row = {
                    "event_id": link["event_id"],
                    "event_type": link["event_type"],
                    "parameter": param,
                    "q": float(q[k]),
                    "tau": tau,
                    "tau_name": link["tau_name"],
                    "link": link["kind"],
                }
                for (p, key), values in dq.items():
                    if p == param:
                        row[f"dq/{key}"] = float(values[k])
                row["dq"] = row[f"dq/{default}"]
                spread = [
                    row[f"dq/{default.split('/')[0]}/{d:g}"]
                    for d in DELTAS
                    if f"dq/{default.split('/')[0]}/{d:g}" in row
                ]
                row["damping_spread"] = float(np.ptp(spread)) if spread else np.nan
                row["flip_distance"] = flip_distance(row["q"], tau, row["dq"])
                rows.append(row)
    info["time_s"] = time.perf_counter() - started
    if progress:
        progress(
            f"    {info['alive']} alive, {len(selected)} links, field "
            f"{field_norm:.1e}, q error {info['q_error']:.1e}, "
            f"{info['time_s']:.0f} s"
        )
    return {"rows": rows, "solves": solves, "info": info}


def flip_distance(q: float, tau: float, dq: float) -> float:
    """``Delta log lambda`` at which ``q`` reaches ``tau`` (``inf`` if unsteerable)."""
    if not np.isfinite(dq) or abs(dq) < DQ_FLOOR:
        return np.inf
    return float((tau - q) / dq)


#: The outcome rates elasticities are reported for: name -> (event type, the
#: events' flag column or ``None``).
OUTCOMES = {
    "thin_junction": ("junction", "thin"),
    "junction": ("junction", None),
    "continuous_window": ("window", "continuous"),
    "window": ("window", None),
    "bump_spur": ("bump", None),
    "protrusion_spur": ("protrusion", None),
}
#: The target outcome rates (Section 11.4), for S7's choice of weight.
TARGET_OUTCOMES = ("thin_junction", "bump_spur", "continuous_window")


def with_outcomes(implicit: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """The implicit rows with each outcome rate they count in (``outcome``).

    The ``thin`` and ``continuous`` flags come from the run's events table.
    """
    if implicit.empty:
        return implicit.assign(outcome=pd.Series(dtype=str))
    keys = ["subset", "seed", "event_type", "event_id"]
    flags = events[events["stage"] == "raw"][
        keys + [c for c in ("thin", "continuous") if c in events]
    ].drop_duplicates(keys)
    frame = implicit.merge(flags, on=keys, how="left")
    parts = []
    for name, (kind, flag) in OUTCOMES.items():
        mine = frame[frame["event_type"] == kind]
        if flag is not None:
            mine = mine[mine[flag].astype("boolean").fillna(False).astype(bool)]
        parts.append(mine.assign(outcome=name))
    return pd.concat(parts, ignore_index=True)


def elasticities(implicit: pd.DataFrame, step: float = ELASTICITY_STEP) -> pd.DataFrame:
    """Net share of events that flip the good way per +10% of each parameter.

    ``q < tau`` is the good outcome (connected; no spur). At ``log lambda +
    step``, an event flips when its flip distance lies in ``(0, step]``: to
    ``q < tau`` is good, the other way bad. Per (parameter, subset and
    pooled, outcome rate of :func:`with_outcomes`): the elasticity (good
    minus bad flips over events) with a cluster-bootstrap interval over
    samples, the flips both ways, and the unsteerable share.
    """
    if implicit.empty:
        return pd.DataFrame()
    frame = implicit.copy()
    if "outcome" not in frame:
        frame["outcome"] = frame["event_type"]
    distance = frame["flip_distance"].astype(float)
    flips = (distance > 0) & (distance <= step)
    good_now = frame["q"] < frame["tau"]
    frame["good"] = (flips & ~good_now).astype(float)
    frame["bad"] = (flips & good_now).astype(float)
    frame["net"] = frame["good"] - frame["bad"]
    frame["unsteerable"] = ~np.isfinite(distance)
    frame = frame.assign(
        cluster=frame["subset"].astype(str) + "/" + frame["seed"].astype(str)
    )
    frame = pd.concat(
        [frame, frame.assign(subset=POOLED, cluster=frame["seed"].astype(str))],
        ignore_index=True,
    )
    rows = []
    for (param, subset, outcome), group in frame.groupby(
        ["parameter", "subset", "outcome"], sort=False
    ):
        _, low, high = cluster_bootstrap(
            group["net"].to_numpy(), group["cluster"].to_numpy(), min_clusters=2
        )
        rows.append(
            {
                "parameter": param,
                "subset": subset,
                "outcome": outcome,
                "events": len(group),
                "elasticity": float(group["net"].mean()),
                "low": low,
                "high": high,
                "good": int(group["good"].sum()),
                "bad": int(group["bad"].sum()),
                "unsteerable": float(group["unsteerable"].mean()),
                "median_flip_distance": float(
                    np.median(np.abs(group["flip_distance"].astype(float)))
                ),
            }
        )
    return pd.DataFrame(rows)


def second_weight(elasticity: pd.DataFrame, candidates=("bce", "dice")) -> dict:
    """S7's second weight: the BCE or Dice weight with the larger elasticity.

    The larger absolute elasticity on any target outcome rate
    (:data:`TARGET_OUTCOMES`), in any subset (Section 11.4). Ties go to the
    first candidate.
    """
    best: dict[str, Any] = {"parameter": None, "elasticity": 0.0}
    if elasticity.empty:
        return best
    targets = elasticity[elasticity["outcome"].isin(TARGET_OUTCOMES)]
    for name in candidates:
        param = f"fit.fit.losses.{name}.weight"
        part = targets[targets["parameter"] == param]
        if part.empty:
            continue
        row = part.loc[part["elasticity"].abs().idxmax()]
        if best["parameter"] is None or abs(row["elasticity"]) > abs(
            best["elasticity"]
        ):
            best = {
                "parameter": param,
                "elasticity": float(row["elasticity"]),
                "subset": row["subset"],
                "outcome": row["outcome"],
            }
    return best


# -- validation (Section 11.4) ---------------------------------------------------------


def validation_variant(named, param: str, factor: float):
    """The variant with one parameter scaled by ``factor`` (a named override)."""
    from spleleton.bench.variants import get_variant

    base = named.variant
    value = _get_path(base, param) * factor
    name = named.name.split("[")[0]
    return get_variant(name, [f"{param}={value:.12g}"])


def _get_path(variant, path: str) -> float:
    obj: Any = variant
    parts = path.split(".")
    i = 0
    while i < len(parts):
        part = parts[i]
        if part == "losses":
            name = parts[i + 1]
            term = next(t for t in obj.losses if t.name == name)
            obj = term
            i += 2
            continue
        obj = getattr(obj, part)
        i += 1
    return float(obj)


def _good(events: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """The raw-stage events of an outcome rate, with ``good`` (connected; no spur)."""
    kind, flag = OUTCOMES[outcome]
    part = events[(events["stage"] == "raw") & (events["event_type"] == kind)]
    if flag is not None and flag in part:
        part = part[part[flag].astype("boolean").fillna(False).astype(bool)]
    column = "connected" if kind in ("junction", "window") else "spur"
    if part.empty or column not in part:
        return part.iloc[:0].assign(good=pd.Series(dtype=bool))
    if kind in ("junction", "window"):
        good = part["connected"].astype("boolean").fillna(False).astype(bool)
    else:
        good = ~part["spur"].astype("boolean").fillna(False).astype(bool)
    return part.assign(good=good)


def validation_rows(
    base_events: pd.DataFrame,
    other_events: pd.DataFrame,
    implicit: pd.DataFrame,
    param: str,
    factor: float,
) -> list[dict]:
    """Observed against predicted changes of the good rates, per outcome and subset.

    Observed: the raw-stage good rate (connected; no spur) of the re-fit
    minus the baseline's, over the events both have. Predicted, from the
    implicit derivatives (with :func:`with_outcomes`' ``outcome``): the net
    share of those events whose flip distance lies between 0 and ``log
    factor``, the good way minus the bad.
    """
    rows = []
    step = math.log(factor)
    keys = ["subset", "seed", "event_type", "event_id"]
    mine = implicit[implicit["parameter"] == param]
    for outcome in OUTCOMES:
        a, b = _good(base_events, outcome), _good(other_events, outcome)
        joined = a[[*keys, "good"]].merge(
            b[[*keys, "good"]], on=keys, suffixes=("_a", "_b")
        )
        if joined.empty:
            continue
        predicted_rows = mine[mine["outcome"] == outcome] if "outcome" in mine else mine
        for subset in [*sorted(joined["subset"].unique()), POOLED]:
            part = joined if subset == POOLED else joined[joined["subset"] == subset]
            pred = (
                predicted_rows
                if subset == POOLED
                else predicted_rows[predicted_rows["subset"] == subset]
            )
            if part.empty:
                continue
            distance = pred["flip_distance"].astype(float)
            if step > 0:
                flips = (distance > 0) & (distance <= step)
            else:
                flips = (distance < 0) & (distance >= step)
            good_now = pred["q"] < pred["tau"]
            net = float((flips & ~good_now).sum() - (flips & good_now).sum())
            rows.append(
                {
                    "parameter": param,
                    "factor": factor,
                    "outcome": outcome,
                    "subset": subset,
                    "events": len(part),
                    "observed": float(part["good_b"].mean() - part["good_a"].mean()),
                    "predicted": net / len(part),
                }
            )
    return rows


# -- data margins (Section 11.6) -------------------------------------------------------


def bump_critical_levels(events: pd.DataFrame, stage: str = "clean") -> pd.DataFrame:
    """Per bump of a height sweep, the lowest height at which it grows a spur.

    Bumps of ``bumps-h*`` sit in the same places at every height, so a bump
    is one event ID per seed across the sweep's subsets.
    """
    part = events[
        (events["event_type"] == "bump")
        & (events["stage"] == stage)
        & events["subset"].astype(str).str.startswith("bumps-h")
    ]
    if part.empty:
        return pd.DataFrame()
    part = part.assign(level=part["subset"].str.removeprefix("bumps-h").astype(float))
    rows = []
    for (seed, event_id), group in part.groupby(["seed", "event_id"]):
        spurred = group[group["spur"].astype(bool)]["level"]
        rows.append(
            {
                "seed": seed,
                "event_id": event_id,
                "levels": len(group),
                "critical_level": float(spurred.min()) if len(spurred) else np.inf,
            }
        )
    return pd.DataFrame(rows)


#: Robustness margins: (event type, outcome that is good, covariate, the good
#: value, and whether the covariate's low end is the hard one).
MARGINS = (
    ("junction", "connected", "radius_ratio", True, True),
    ("bump", "spur", "height_over_r", False, False),
)


def robustness_margins(
    events: pd.DataFrame,
    levels=(0.9, 0.5),
    bins: int = 5,
    resamples: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """The covariate value at which an outcome rate drops below 90% and 50%.

    Per (stage, event type): the good rate in bins of the covariate (on the
    hard side), and the first crossing of each level, linearly
    interpolated between bin centres; with a cluster-bootstrap interval
    (resampling seeds). NaN when the rate never drops below the level.
    """
    rows = []
    rng = np.random.default_rng(seed)
    for kind, outcome, covariate, good, low_is_hard in MARGINS:
        part = events[(events["event_type"] == kind)]
        if part.empty or covariate not in part:
            continue
        for stage, group in part.groupby("stage"):
            group = group.dropna(subset=[covariate])
            x = group[covariate].astype(float).to_numpy()
            y = (group[outcome].astype(bool) == good).astype(float).to_numpy()
            seeds = (
                group["subset"].astype(str) + "/" + group["seed"].astype(str)
            ).to_numpy()
            if len(np.unique(x)) < 2:
                continue
            edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
            for level in levels:
                value = _crossing(x, y, edges, level, low_is_hard)
                boot = []
                clusters = np.unique(seeds)
                for _ in range(resamples if len(clusters) > 1 else 0):
                    pick = rng.choice(clusters, len(clusters))
                    index = np.concatenate([np.flatnonzero(seeds == c) for c in pick])
                    boot.append(
                        _crossing(x[index], y[index], edges, level, low_is_hard)
                    )
                boot = np.array(boot, dtype=float)
                # no interval for a margin the data never reach
                finite = boot[np.isfinite(boot)] if np.isfinite(value) else boot[:0]
                rows.append(
                    {
                        "stage": stage,
                        "event_type": kind,
                        "outcome": outcome,
                        "covariate": covariate,
                        "level": level,
                        "margin": value,
                        "low": float(np.quantile(finite, 0.025))
                        if len(finite)
                        else np.nan,
                        "high": float(np.quantile(finite, 0.975))
                        if len(finite)
                        else np.nan,
                        "events": len(group),
                    }
                )
    return pd.DataFrame(rows)


def _crossing(x, y, edges, level, low_is_hard) -> float:
    centres, rates = [], []
    for lo, hi in itertools.pairwise(edges):
        inside = (x >= lo) & (x <= hi)
        if inside.any():
            centres.append(0.5 * (lo + hi))
            rates.append(float(y[inside].mean()))
    order = np.argsort(centres)[::-1] if low_is_hard else np.argsort(centres)
    centres, rates = np.array(centres)[order], np.array(rates)[order]
    for k in range(1, len(rates)):
        if rates[k - 1] >= level > rates[k]:
            t = (rates[k - 1] - level) / max(rates[k - 1] - rates[k], 1e-12)
            return float(centres[k - 1] + t * (centres[k] - centres[k - 1]))
    return np.nan


# -- Morris screening (Section 11.5; written, not run) ---------------------------------


def morris_design(k: int, trajectories: int, levels: int = 4, seed: int = 0):
    """Morris's elementary-effects design on ``[0, 1]^k``.

    Each trajectory starts at a random point of the ``levels`` grid and moves
    one coordinate at a time, in random order, by ``Delta = levels / (2
    (levels - 1))``. Returns an array ``(trajectories, k + 1, k)`` and the
    order the coordinates moved in, ``(trajectories, k)``.
    """
    rng = np.random.default_rng(seed)
    delta = levels / (2 * (levels - 1))
    grid = np.arange(levels) / (levels - 1)
    starts = grid[grid <= 1 - delta + 1e-12]
    points = np.zeros((trajectories, k + 1, k))
    orders = np.zeros((trajectories, k), dtype=int)
    for t in range(trajectories):
        x = rng.choice(starts, size=k)
        order = rng.permutation(k)
        points[t, 0] = x
        for step, i in enumerate(order):
            x = x.copy()
            x[i] += delta
            points[t, step + 1] = x
        orders[t] = order
    return points, orders


def morris_effects(points: np.ndarray, orders: np.ndarray, outputs: np.ndarray):
    """``mu*`` (mean absolute effect) and ``sigma`` per coordinate.

    ``outputs`` has shape ``(trajectories, k + 1)``: the outcome at each
    point of :func:`morris_design`.
    """
    trajectories, _, k = points.shape
    effects = np.zeros((trajectories, k))
    for t in range(trajectories):
        for step, i in enumerate(orders[t]):
            delta = points[t, step + 1, i] - points[t, step, i]
            effects[t, i] = (outputs[t, step + 1] - outputs[t, step]) / delta
    return {
        "mu_star": np.abs(effects).mean(axis=0),
        "mu": effects.mean(axis=0),
        "sigma": effects.std(axis=0, ddof=1) if trajectories > 1 else np.zeros(k),
    }


def morris_values(point: np.ndarray, ranges: dict[str, tuple[float, float]]):
    """A design point's parameter values (log-spaced within each range)."""
    return {
        name: float(np.exp(np.log(lo) + p * (np.log(hi) - np.log(lo))))
        for p, (name, (lo, hi)) in zip(point, ranges.items(), strict=True)
    }


# -- runs ------------------------------------------------------------------------------


def _done_samples(meta: dict, subsets=None):
    for label, entry in meta["samples"].items():
        subset, seed = label.rsplit("_", 1)
        if entry["status"] == "done" and (not subsets or subset in subsets):
            yield label, subset, int(seed), entry


def thresholds_run(
    store,
    directory: Path,
    grid: bool = True,
    subsets=None,
    progress: Callable[[str], None] | None = print,
    full_grid: bool = False,
) -> dict[str, Path]:
    """``bench sensitivity thresholds``: critical values and the restricted grid.

    Writes ``sensitivity/thresholds.parquet`` (event, threshold, critical
    value, stage), ``sensitivity/grid.parquet`` (per setting, event and
    stage: the actual outcome) and ``sensitivity/agreement.parquet``; with
    ``full_grid``, also ``sensitivity/grid_samples.parquet`` (every setting
    scored in full, :func:`full_grid_rows`).
    """
    from spleleton.bench.compare import read_tables
    from spleleton.bench.runner import load_fit, read_run
    from spleleton.bench.suites import load_split
    from spleleton.bench.variants import Variant

    meta = read_run(directory)
    variant = Variant.from_dict(meta["spec"])
    if variant.graph.method != "assemble":
        raise ValueError("the decision quantities are built for 'assemble' only")
    out_dir = directory / "sensitivity"
    out_dir.mkdir(exist_ok=True)
    events = read_tables(directory).events
    if subsets:
        events = events[events["subset"].isin(subsets)]
    critical = critical_rows(events)
    paths = {"thresholds": out_dir / "thresholds.parquet"}
    critical.to_parquet(paths["thresholds"], index=False)
    if not grid:
        return paths
    settings = grid_settings(variant.graph)
    frames = []
    for label, subset, seed, entry in _done_samples(meta, subsets):
        part_path = out_dir / "parts" / f"{label}.grid.parquet"
        if part_path.exists():
            frame = pd.read_parquet(part_path)
            if frame["setting"].nunique() == len(settings):
                frames.append(frame)
                continue
        (loaded,) = load_split(store, meta["suite"], subset, meta["split"], [seed])
        segments = load_fit(store.fit_path(entry["fit_key"])).segments
        started = time.perf_counter()
        frame = grid_rows(loaded.sample, segments, variant.graph, settings).assign(
            subset=subset, seed=seed
        )
        if progress:
            progress(
                f"  {label}: {len(settings)} settings, "
                f"{time.perf_counter() - started:.1f} s"
            )
        part_path.parent.mkdir(exist_ok=True)
        frame.to_parquet(part_path, index=False)
        frames.append(frame)
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    paths["grid"] = out_dir / "grid.parquet"
    table.to_parquet(paths["grid"], index=False)
    agreement = grid_agreement(table, events, settings, variant.graph)
    paths["agreement"] = out_dir / "agreement.parquet"
    agreement.to_parquet(paths["agreement"], index=False)
    if full_grid:
        frames = []
        for label, subset, seed, entry in _done_samples(meta, subsets):
            part_path = out_dir / "parts" / f"{label}.grid_samples.parquet"
            if part_path.exists():
                frame = pd.read_parquet(part_path)
                if frame["setting"].nunique() == len(settings):
                    frames.append(frame)
                    continue
            (loaded,) = load_split(store, meta["suite"], subset, meta["split"], [seed])
            segments = load_fit(store.fit_path(entry["fit_key"])).segments
            frame = full_grid_rows(
                loaded.sample, segments, variant.graph, settings
            ).assign(subset=subset, seed=seed)
            frame.to_parquet(part_path, index=False)
            frames.append(frame)
        paths["grid_samples"] = out_dir / "grid_samples.parquet"
        pd.concat(frames, ignore_index=True).to_parquet(
            paths["grid_samples"], index=False
        )
    return paths


#: The implicit derivatives' parameters by default (Section 11.4).
DEFAULT_PARAMS = (
    "fit.fit.losses.bce.weight",
    "fit.fit.losses.dice.weight",
    "fit.connection_weight",
    "fit.fit.render.wall_width",
)


def implicit_run(
    store,
    directory: Path,
    params: Sequence[str] = DEFAULT_PARAMS,
    subsets=None,
    method: str | None = None,
    check_width: bool = False,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Path]:
    """``bench sensitivity implicit``: per sample, resumable.

    Writes ``sensitivity/implicit.parquet`` (per event and parameter),
    ``sensitivity/solves.parquet`` (per sample, parameter and damping: the
    solver's diagnostics) and ``sensitivity/implicit_info.parquet`` (per
    sample: the field norm, the spectrum, the ``q`` check, timings).
    """
    import json

    from spleleton.bench.runner import load_fit, read_run
    from spleleton.bench.suites import load_split
    from spleleton.bench.variants import Variant

    meta = read_run(directory)
    variant = Variant.from_dict(meta["spec"])
    out_dir = directory / "sensitivity" / "parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    key = "|".join([*params, str(method), str(check_width)])
    rows, solves, infos = [], [], []
    for label, subset, seed, entry in _done_samples(meta, subsets):
        path = out_dir / f"{label}.implicit.json"
        if path.exists() and json.loads(path.read_text()).get("key") == key:
            result = json.loads(path.read_text())
        else:
            (loaded,) = load_split(store, meta["suite"], subset, meta["split"], [seed])
            record = load_fit(store.fit_path(entry["fit_key"]))
            if progress:
                progress(f"  {label}:")
            result = implicit_sample(
                variant, loaded.sample, record, params, method, check_width, progress
            )
            result["key"] = key
            path.write_text(json.dumps(_plain(result)))
        keys = {"subset": subset, "seed": seed}
        rows += [{**keys, **r} for r in result["rows"]]
        solves += [{**keys, **r} for r in result["solves"]]
        infos.append({**keys, **_flat(result["info"])})
    base = directory / "sensitivity"
    paths = {}
    for name, data in (
        ("implicit", rows),
        ("solves", solves),
        ("implicit_info", infos),
    ):
        paths[name] = base / f"{name}.parquet"
        pd.DataFrame(data).to_parquet(paths[name], index=False)
    return paths


def _flat(info: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in info.items():
        if isinstance(v, dict):
            out |= _flat(v, f"{prefix}{k}.")
        else:
            out[f"{prefix}{k}"] = v
    return out


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# -- the report (Section 11.7) ---------------------------------------------------------

#: The outcome groups of the threshold curves: event types -> label.
CURVE_OUTCOMES = {
    "junction": "connected",
    "window": "connected",
    "bump": "no spur",
    "protrusion": "no spur",
}


def _good_of(frame: pd.DataFrame, column: str) -> pd.Series:
    """Good outcomes: connected for connection events, no spur for spurs."""
    spur = frame["event_type"].isin(["bump", "protrusion"])
    value = frame[column].astype(bool)
    return value.where(~spur, ~value)


def threshold_curves(events, grid, settings, config, points: int = 200) -> pd.DataFrame:
    """Each threshold's step function and the grid's actual rates.

    Rates of good outcomes (connected; no spur), pooled over the run's
    events, per outcome group and stage. Step functions only for the
    thresholds with one (:data:`STEPPED`), on ``points`` values across the
    grid's range; grid rates at the threshold's 1D line.
    """
    rows = []
    for name in THRESHOLDS:
        values = grid_values(config, name)
        default = threshold_value(config, name)
        if name in STEPPED:
            for value in np.linspace(min(values), max(values), points):
                predicted = predict(events, {name: float(value)}, config)
                if predicted.empty:
                    continue
                predicted = predicted.assign(
                    good=_good_of(predicted, "predicted"),
                    group=predicted["event_type"].map(CURVE_OUTCOMES),
                )
                for (group, stage), part in predicted.groupby(["group", "stage"]):
                    rows.append(
                        {
                            "threshold": name,
                            "outcome": group,
                            "stage": stage,
                            "value": float(value),
                            "rate": float(part["good"].mean()),
                            "kind": "step",
                            "default": default,
                        }
                    )
        if grid is None or grid.empty:
            continue
        for index, setting in enumerate(settings):
            if set(setting) - {name} or (setting and name not in setting):
                continue
            value = setting.get(name, default)
            part = grid[grid["setting"] == index]
            part = part.assign(
                good=_good_of(part, "actual"),
                group=part["event_type"].map(CURVE_OUTCOMES),
            )
            for (group, stage), mine in part.groupby(["group", "stage"]):
                rows.append(
                    {
                        "threshold": name,
                        "outcome": group,
                        "stage": stage,
                        "value": float(value),
                        "rate": float(mine["good"].mean()),
                        "kind": "grid",
                        "default": default,
                    }
                )
    return pd.DataFrame(rows)


def threshold_effects(curves: pd.DataFrame) -> pd.DataFrame:
    """Per threshold, outcome and stage: how much its grid moves the good rate.

    ``range`` is the largest minus the smallest grid rate; ``near_default``
    the largest change between the default and its neighbours on the grid.
    """
    rows = []
    grid = curves[curves["kind"] == "grid"]
    for (name, outcome, stage), part in grid.groupby(["threshold", "outcome", "stage"]):
        part = part.sort_values("value")
        rates, values = part["rate"].to_numpy(), part["value"].to_numpy()
        k = int(np.argmin(np.abs(values - part["default"].iloc[0])))
        near = [abs(rates[j] - rates[k]) for j in (k - 1, k + 1) if 0 <= j < len(rates)]
        rows.append(
            {
                "threshold": name,
                "outcome": outcome,
                "stage": stage,
                "range": float(rates.max() - rates.min()),
                "near_default": float(max(near, default=0.0)),
                "rate_at_default": float(rates[k]),
            }
        )
    return pd.DataFrame(rows)


def prune_inert(curves: pd.DataFrame, lo: float = 0.1, hi: float = 0.9) -> bool | None:
    """Whether ``prune_threshold`` changes no connection rate on ``[lo, hi]``."""
    part = curves[
        (curves["threshold"] == "prune_threshold")
        & (curves["kind"] == "grid")
        & (curves["outcome"] == "connected")
    ]
    part = part[(part["value"] >= lo - 1e-9) & (part["value"] <= hi + 1e-9)]
    if part.empty:
        return None
    return bool((part.groupby("stage")["rate"].agg(np.ptp) == 0).all())


def validation_agreement(validation: pd.DataFrame, floor: float = 0.0) -> dict:
    """F11 and F12 from the validation rows (pooled, per parameter and factor).

    Sign agreement where both changes exceed ``floor``; magnitude within
    2x; ranking of the parameters by mean absolute observed and predicted
    change; linearity (the f = 0.2 change about twice the f = 0.1 one); and
    F12's rule: Morris screening is needed when the observed changes rank
    two validated weights opposite to the predictions, or contradict a
    predicted sign on a target outcome beyond the floor.
    """
    if validation.empty:
        return {"pending": "no validation rows"}
    part = validation[validation["subset"] == POOLED]
    signs = part[(part["observed"].abs() > floor) & (part["predicted"].abs() > floor)]
    same = np.sign(signs["observed"]) == np.sign(signs["predicted"])
    ratio = part["observed"].abs() / part["predicted"].abs().replace(0, np.nan)
    within = ((ratio >= 0.5) & (ratio <= 2)).mean() if ratio.notna().any() else np.nan
    rank = part.groupby("parameter")[["observed", "predicted"]].apply(
        lambda f: f.abs().mean()
    )
    params = list(rank.index)
    ranking_agrees = None
    if len(params) >= 2:
        ranking_agrees = bool(
            list(rank["observed"].sort_values().index)
            == list(rank["predicted"].sort_values().index)
        )
    linear = {}
    for (param, outcome), group in part.assign(
        f=(part["factor"] - 1).abs().round(6)
    ).groupby(["parameter", "outcome"]):
        by_f = group.groupby("f")["observed"].apply(lambda s: s.abs().mean())
        if {0.1, 0.2} <= set(by_f.index) and by_f[0.1] > 0:
            linear[f"{param}/{outcome}"] = float(by_f[0.2] / by_f[0.1])
    targets = part[part["outcome"].isin(TARGET_OUTCOMES)]
    contradicted = targets[
        (np.sign(targets["observed"]) == -np.sign(targets["predicted"]))
        & (targets["observed"].abs() > floor)
        & (targets["predicted"].abs() > floor)
    ]
    morris = None
    if ranking_agrees is not None:
        morris = bool((not ranking_agrees) or len(contradicted))
    return {
        "sign_agreement": float(same.mean()) if len(same) else None,
        "sign_compared": len(same),
        "within_2x": None if not np.isfinite(within) else float(within),
        "ranking_agrees": ranking_agrees,
        "linearity": linear,
        "contradicted_signs": contradicted[["parameter", "factor", "outcome"]].to_dict(
            "records"
        ),
        "morris_needed": morris,
        "rule": "Morris is needed when the re-fits rank the two weights "
        "opposite to the elasticities or contradict a predicted sign "
        "on a target outcome rate beyond the floor (Section 11.5)",
    }


def sensitivity_report(
    directory: Path, output: Path | None = None
) -> tuple[Path, dict]:
    """A run's sensitivity report and its digest (F9, F10).

    Reads the run's ``sensitivity/`` tables and writes ``report.md`` with the
    threshold curves, the grid's agreement, the elasticities, the flip
    distances and the robustness margins, and their figures.
    """
    from spleleton.bench import plots
    from spleleton.bench.compare import _fmt, _table, read_tables
    from spleleton.bench.runner import read_run
    from spleleton.bench.variants import Variant

    base = directory / "sensitivity"
    output = output or base / "report"
    (output / "figures").mkdir(parents=True, exist_ok=True)
    meta = read_run(directory)
    config = Variant.from_dict(meta["spec"]).graph
    events = read_tables(directory).events
    lines = [f"# Sensitivity of `{meta['name']}`", ""]
    digest: dict[str, Any] = {}
    grid = (
        pd.read_parquet(base / "grid.parquet")
        if (base / "grid.parquet").exists()
        else None
    )
    curves = pd.DataFrame()
    if (base / "thresholds.parquet").exists():
        settings = grid_settings(config)
        if grid is not None and len(grid):
            events = events[events["subset"].isin(grid["subset"].unique())]
        curves = threshold_curves(events, grid, settings, config)
        curves.to_parquet(output / "threshold_curves.parquet", index=False)
        figure = plots.plot_threshold_curves(
            curves, output / "figures" / "thresholds.png"
        )
        effects = threshold_effects(curves) if len(curves) else pd.DataFrame()
        lines += ["## Graph-stage thresholds (F9)", ""]
        if len(effects):
            lines += _table(
                [
                    "threshold",
                    "outcome",
                    "stage",
                    "rate at default",
                    "grid range",
                    "next to default",
                ],
                [
                    [
                        r.threshold,
                        r.outcome,
                        r.stage,
                        _fmt(r.rate_at_default),
                        _fmt(r.range),
                        _fmt(r.near_default),
                    ]
                    for r in effects.itertuples()
                ],
            )
        lines += [f"![threshold curves]({figure.relative_to(output).as_posix()})", ""]
        agreement = (
            pd.read_parquet(base / "agreement.parquet")
            if (base / "agreement.parquet").exists()
            else pd.DataFrame()
        )
        if len(agreement):
            lines += [
                "The step functions against the restricted grid (anchors matched):",
                "",
            ]
            lines += _table(
                ["line", "stage", "points", "agree"],
                [
                    [r.line, r.stage, r.points, _fmt(r.share)]
                    for r in agreement.itertuples()
                ],
            )
        digest["F9"] = {
            "effects": effects.to_dict("records") if len(effects) else [],
            "agreement": agreement.to_dict("records") if len(agreement) else [],
            "prune_inert": prune_inert(curves) if len(curves) else None,
            "settings": len(grid_settings(config)),
        }
    if (base / "implicit.parquet").exists():
        implicit = pd.read_parquet(base / "implicit.parquet")
        info = pd.read_parquet(base / "implicit_info.parquet")
        implicit = with_outcomes(implicit, events)
        elasticity = elasticities(implicit)
        elasticity.to_parquet(output / "elasticities.parquet", index=False)
        lines += ["## Fit parameters: implicit derivatives (F10)", ""]
        lines += [
            (
                f"{len(info)} volumes, {int(info['alive'].sum())} alive segments in "
                f"all; the field's norm at the fits "
                f"{_fmt(info['field_norm'].median(), 5)} (median); JAX q against "
                f"numpy q to {info['q_error'].max():.1e}."
            ),
            "",
        ]
        pooled = (
            elasticity[elasticity["subset"] == POOLED]
            if len(elasticity)
            else elasticity
        )
        if len(pooled):
            lines += _table(
                [
                    "parameter",
                    "outcome",
                    "events",
                    "elasticity",
                    "95% interval",
                    "good",
                    "bad",
                    "unsteerable",
                    "median |flip|",
                ],
                [
                    [
                        r.parameter,
                        r.outcome,
                        r.events,
                        _fmt(r.elasticity),
                        f"[{_fmt(r.low)}, {_fmt(r.high)}]",
                        r.good,
                        r.bad,
                        _fmt(r.unsteerable),
                        _fmt(r.median_flip_distance),
                    ]
                    for r in pooled.itertuples()
                ],
            )
            tornado = plots.plot_tornado(pooled, output / "figures" / "tornado.png")
            flips = plots.plot_flip_distances(
                implicit.drop_duplicates(["subset", "seed", "event_id", "parameter"]),
                output / "figures" / "flip_distances.png",
            )
            for figure, alt in ((tornado, "tornado"), (flips, "flip distances")):
                lines += [f"![{alt}]({figure.relative_to(output).as_posix()})", ""]
        spread = implicit.drop_duplicates(["subset", "seed", "event_id", "parameter"])
        digest["F10"] = {
            "elasticities": pooled.to_dict("records") if len(pooled) else [],
            "unsteerable_events": int(
                spread.groupby(["subset", "seed", "event_id"])["flip_distance"]
                .apply(lambda s: bool(np.isinf(s.astype(float)).all()))
                .sum()
            ),
            "events": int(
                spread[["subset", "seed", "event_id"]].drop_duplicates().shape[0]
            ),
            "damping_spread_median": float(spread["damping_spread"].abs().median()),
            "q_error_max": float(info["q_error"].max()),
            "second_weight": second_weight(elasticity),
        }
    margins = robustness_margins(read_tables(directory).events)
    if len(margins):
        margins.to_parquet(output / "robustness.parquet", index=False)
        lines += ["## Robustness margins", ""]
        lines += _table(
            ["stage", "event", "covariate", "level", "margin", "95% interval"],
            [
                [
                    r.stage,
                    r.event_type,
                    r.covariate,
                    _fmt(r.level),
                    _fmt(r.margin),
                    f"[{_fmt(r.low)}, {_fmt(r.high)}]",
                ]
                for r in margins.itertuples()
            ],
        )
        digest.setdefault("F9", {})["robustness"] = margins.to_dict("records")
    levels = bump_critical_levels(read_tables(directory).events)
    if len(levels):
        digest.setdefault("F9", {})["bump_critical_levels"] = (
            levels["critical_level"].replace(np.inf, np.nan).describe().to_dict()
        )
    report = output / "report.md"
    report.write_text("\n".join(lines) + "\n")
    return report, digest


def morris_run(
    store,
    named,
    params: Sequence[str],
    scope: dict[str, Any],
    trajectories: int = 8,
    span: float = 2.0,
    seed: int = 0,
    progress: Callable[[str], None] | None = print,
) -> pd.DataFrame:
    """Morris screening of fit parameters by re-fits (optional, Section 11.5).

    Each parameter ranges over ``[default / span, default * span]`` on a log
    scale; every design point is a run (``k + 1`` per trajectory). Returns
    per (parameter, outcome rate): ``mu_star``, ``mu`` and ``sigma`` of the
    pooled raw-stage good rate. Not part of any stage: S6 and S7 decide
    whether it is needed (F12).
    """
    from spleleton.bench.compare import read_tables
    from spleleton.bench.runner import run_variant
    from spleleton.bench.variants import get_variant

    base = named.variant
    ranges = {p: (_get_path(base, p) / span, _get_path(base, p) * span) for p in params}
    points, orders = morris_design(len(params), trajectories, seed=seed)
    name = named.name.split("[")[0]
    outputs: dict[str, np.ndarray] = {}
    for t in range(trajectories):
        for step in range(len(params) + 1):
            values = morris_values(points[t, step], ranges)
            variant = get_variant(name, [f"{k}={v!r}" for k, v in values.items()])
            directory = run_variant(store, variant, progress=progress, **scope)
            events = read_tables(directory).events
            for outcome in OUTCOMES:
                part = _good(events, outcome)
                rate = float(part["good"].mean()) if len(part) else np.nan
                outputs.setdefault(
                    outcome, np.full((trajectories, len(params) + 1), np.nan)
                )[t, step] = rate
    rows = []
    for outcome, values in outputs.items():
        effects = morris_effects(points, orders, values)
        for i, param in enumerate(params):
            rows.append(
                {
                    "parameter": param,
                    "outcome": outcome,
                    "mu_star": float(effects["mu_star"][i]),
                    "mu": float(effects["mu"][i]),
                    "sigma": float(effects["sigma"][i]),
                }
            )
    return pd.DataFrame(rows)
