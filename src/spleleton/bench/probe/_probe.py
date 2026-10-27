"""Probing one set-up: margins, forces, classes; and the checks behind them.

For one (event, family, opacity), four batched calls on its crop:

1. **terms** of the reference, the ε grid, and ε ± h on and off the grid
   (for central finite differences);
2. **profile relaxations**: the reference and each profile ε, the error's
   segments held, every other free parameter re-fitted for
   ``spec.profile_iterations`` Adam steps (and, for the floor, the reference
   from two jittered starts);
3. **forces** by ``jax.jvp`` along the error path: on the grid, off the grid
   (the event's points moved by 0.3 voxel, because the closest-point
   parameter clamps exactly at a segment's end) and at the relaxed profiles;
   each as the force the fit feels and as the margin's own derivative.

Each margin is per term, unweighted; the total is weighted. A margin under
one voxel's worth of loss (1 / n_voxels) counts as zero.
"""

import math
import time
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from spleleton.bench.probe._crop import (
    SLOT_BUCKETS,
    Background,
    background,
    bucket,
    crop_blocks,
    crop_capacity,
    crop_static,
    inside_crop,
    make_crop,
    place,
)
from spleleton.bench.probe._setup import (
    Reference,
    Setup,
    capacity_for,
    config,
    full_static,
    jittered,
    padded,
    relax_full,
    tangent,
)
from spleleton.bench.probe._spec import ProbeSpec
from spleleton.bench.probe._terms import (
    FIELDS,
    Static,
    _masked,
    crop_terms,
    forces_batch,
    full_eval,
    mask_tree,
    relax_batch,
    segments_of,
    stack,
    terms_batch,
)
from spleleton.bench.timing import stopwatch
from spleleton.primitives._primitives_jax import JaxLineSegments
from spleleton.render._render_jax import BinCapacity

#: Step of the central finite differences, in units of ε.
FD_STEP = 1e-5
FD_RTOL = 1e-3
FD_ATOL = 1e-10
#: A margin below this many voxels' worth of loss counts as zero.
MARGIN_VOXELS = 1.0
#: A force below this many times the reference's median endpoint gradient is flat.
FLAT_SNR = 1.0
#: Full-volume profile steps compared with the crop's.
CHECK_ITERS = 10
#: How often a crop may grow to cover a relaxation that left it.
MAX_REGROW = 2
#: The two jitter seeds of the floor's relaxations.
FLOOR_SEEDS = (1, 2)


def _np(tree) -> Any:
    return jax.tree.map(np.asarray, jax.device_get(tree))


def _floats(terms: dict) -> dict:
    return {k: v for k, v in terms.items() if k != "overflow"}


@dataclass
class PlacedCrop:
    """A crop covering some configurations, with its data and settings."""

    origin: np.ndarray
    shape: tuple[int, ...]
    exact_shape: tuple[int, ...]
    capacity: BinCapacity
    crop: Any
    region: tuple[int, ...] | None

    def static(self, reference: Reference) -> Static:
        """The crop's evaluation settings."""
        return crop_static(reference, self.shape, self.capacity, self.region)


def place_crop(
    reference: Reference,
    back: Background,
    spec: ProbeSpec,
    configs: list[dict],
    slots: int,
    bucketed: bool = True,
    extent: list[dict] = (),
) -> PlacedCrop:
    """The crop (bucketed or exact) covering every configuration's blocks.

    ``extent`` adds configurations that only widen the crop (a relaxation's
    trajectory), not the bin capacity.
    """
    shape, spacing = reference.shape, reference.spacing
    render = reference.static.render
    lo, hi = crop_blocks([*configs, *extent], shape, spacing, render)
    origin, crop_shape = place(lo, hi, shape, render.block_size, bucketed)
    _, exact_shape = place(lo, hi, shape, render.block_size, False)
    origin_physical = np.asarray(origin, float) * np.asarray(spacing)
    capacity = crop_capacity(
        configs, origin_physical, crop_shape, spacing, render, slots, bucketed
    )
    crop, region = make_crop(
        reference, back, origin, crop_shape, spec.connection_reach, bucketed
    )
    return PlacedCrop(origin, crop_shape, exact_shape, capacity, crop, region)


def probe_setup(
    reference: Reference,
    setup: Setup,
    back: Background,
    spec: ProbeSpec,
    bucketed: bool = True,
    floor: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """Rigid and profile margins and forces of one set-up on its crop.

    Parameters
    ----------
    reference : Reference
        The relaxed reference.
    setup : Setup
        The (event, family, opacity).
    back : Background
        Its frozen segments' render (:func:`background`).
    spec : ProbeSpec
        The probe.
    bucketed : bool
        Pad the crop, the slots, the frozen endpoints and the capacity to
        their buckets (as the probe runs); otherwise use the exact sizes
        (to check that padding changes nothing).
    floor : bool
        Also relax the reference configuration from two jittered starts.
    seed : int
        Seeds the floor's jitter (with :data:`FLOOR_SEEDS`).

    Returns
    -------
    dict
        Terms, histories, forces, crop and timing (see the code).
    """
    started = time.time()
    slots = bucket(setup.n_exact, SLOT_BUCKETS, 10**9) if bucketed else setup.n_exact
    arrays = padded(setup, slots)
    eps = setup.eps
    profile_eps = spec.profile_eps(eps, setup.collapsed)
    groups = {
        "ref": [config(arrays, setup, 0.0, new=False)],
        "grid": [config(arrays, setup, e) for e in eps],
        "plus": [config(arrays, setup, e + FD_STEP) for e in eps],
        "minus": [config(arrays, setup, e - FD_STEP) for e in eps],
        "offgrid": [config(arrays, setup, e, offset=True) for e in eps],
        "plus_offgrid": [config(arrays, setup, e + FD_STEP, offset=True) for e in eps],
        "minus_offgrid": [config(arrays, setup, e - FD_STEP, offset=True) for e in eps],
    }
    relax_rows = groups["ref"] + [groups["grid"][eps.index(e)] for e in profile_eps]
    if floor:
        for jitter_seed in FLOOR_SEEDS:
            rng = np.random.default_rng([seed, jitter_seed])
            relax_rows.append(
                jittered(groups["ref"][0], arrays.frozen, spec.jitter, rng)
            )
    every = [c for group in groups.values() for c in group] + relax_rows
    render = reference.static.render
    frozen = mask_tree(arrays.frozen)
    term_rows = [
        c
        for name in ("ref", "grid", "plus", "minus", "plus_offgrid", "minus_offgrid")
        for c in groups[name]
    ]

    # bucketed batches have fixed row counts (the last row repeated), so that
    # every family and every collapse compiles the same programs
    n_grid = len(spec.eps)
    n_profile_rows = len(spec.profile_eps(spec.eps)) + (2 if floor else 0)
    rows_of = {
        "terms": 1 + 5 * n_grid,
        "relax": 1 + n_profile_rows,
        "forces": 2 * n_grid + len(spec.profile_eps(spec.eps)),
    }

    def call(function, configs, kind, *args, direction=None, **kwargs):
        if bucketed:
            n = len(configs)
            padded_rows = configs + [configs[-1]] * (rows_of[kind] - n)
            extra = ()
            if direction is not None:
                # padded in NumPy: an eager JAX op compiles for every new size
                extra = (
                    jax.tree.map(
                        lambda x: jnp.asarray(
                            np.concatenate(
                                [x, np.repeat(x[-1:], len(padded_rows) - n, axis=0)]
                            )
                        ),
                        direction,
                    ),
                )
            out = function(*stack(padded_rows), *extra, *args, **kwargs)
            return jax.tree.map(lambda x: np.asarray(x)[:n], jax.device_get(out))
        outs = []
        for i, c in enumerate(configs):
            extra = ()
            if direction is not None:
                extra = (
                    jax.tree.map(lambda x, i=i: jnp.asarray(x[i : i + 1]), direction),
                )
            outs.append(function(*stack([c]), *extra, *args, **kwargs))
        return jax.tree.map(lambda *xs: jnp.concatenate(xs), *outs)

    def run(st: Static, crop):
        timing: dict[str, Any] = {}
        with stopwatch() as watch:
            with stopwatch() as part:
                terms = _np(call(terms_batch, term_rows, "terms", crop, st=st))
            timing["terms_s"] = part["wall_s"]
            with stopwatch() as part:
                relaxed = _np(
                    call(
                        relax_batch,
                        relax_rows,
                        "relax",
                        frozen,
                        crop,
                        st=st,
                        n_iterations=spec.profile_iterations,
                    )
                )
            timing["relax_s"] = part["wall_s"]
            history, path, final = relaxed
            n_profile = len(profile_eps)
            relaxed_configs = [
                {
                    **{k: getattr(path, k)[1 + i, -1] for k in FIELDS},
                    "live": relax_rows[1 + i]["live"],
                }
                for i in range(n_profile)
            ]
            force_rows = groups["grid"] + groups["offgrid"] + relaxed_configs
            force_eps = [*eps, *eps, *profile_eps]
            tangents = [tangent(arrays, e) for e in force_eps]
            direction = JaxLineSegments(
                **{k: np.stack([t[k] for t in tangents]) for k in FIELDS}
            )
            with stopwatch() as part:
                forces = _np(
                    call(
                        forces_batch, force_rows, "forces", crop,
                        direction=direction, st=st,
                    )
                )  # fmt: skip
            timing["forces_s"] = part["wall_s"]
        timing.update(
            total_s=watch["wall_s"],
            compile_s=watch["compile_s"],
            programs=watch["programs"],
        )
        overflow = bool(
            np.any(terms["overflow"])
            or np.any(history["overflow"])
            or np.any(final["overflow"])
        )
        return terms, history, path, final, forces, tangents, overflow, timing

    # a free segment that drifts out of its crop in a relaxation would no
    # longer be rendered whole, so the crop grows to cover the trajectory and
    # the batch runs again (at most MAX_REGROW times)
    trajectory: list[dict] = []
    regrown = -1
    while regrown < MAX_REGROW:
        regrown += 1
        placed = place_crop(
            reference, back, spec, every, slots, bucketed, extent=trajectory
        )
        origin, crop_shape = placed.origin, placed.shape
        exact_shape, region, crop = placed.exact_shape, placed.region, placed.crop
        st = placed.static(reference)
        terms, history, path, final, forces, tangents, overflow, timing = run(st, crop)
        if overflow:  # grow to the largest capacity once
            blocks = math.prod(math.ceil(n / render.block_size) for n in crop_shape)
            st = st._replace(capacity=BinCapacity(blocks, slots))
            timing_first = timing
            terms, history, path, final, forces, tangents, overflow, timing = run(
                st, crop
            )
            timing["overflowed"] = timing_first
        if overflow:
            raise RuntimeError("the crop's capacity overflowed")
        trajectory = _trajectories(path, relax_rows)
        outside = [
            r
            for r, row in enumerate(trajectory)
            if not inside_crop(row, row["live"], origin, crop_shape, reference)
        ]
        if not outside:
            break
    timing["regrown"] = regrown
    n = len(eps)
    terms = _floats(terms)
    pick = lambda j: {k: v[1 + j * n : 1 + (j + 1) * n] for k, v in terms.items()}  # noqa: E731
    _, force, derivative = (_floats(f) for f in forces)
    n_profile = len(profile_eps)
    # escapes (rows still outside after regrowing) and how far the free
    # segments moved, per profile ε
    names = ["ref", *map(str, profile_eps), *(["floor"] * 2 if floor else [])]
    escaped = sorted({names[r] for r in outside})
    moves = {}
    movable = ~arrays.frozen["start"].all(axis=1)
    for i, e in enumerate(profile_eps):
        initial = relax_rows[1 + i]
        params = {k: getattr(path, k)[1 + i, -1] for k in FIELDS}
        keep = initial["live"] & movable
        shift = np.concatenate(
            [
                np.linalg.norm(params[f] - initial[f], axis=1)[keep]
                for f in ("start", "end")
            ]
        )
        moves[e] = float(shift.max(initial=0.0))
    return {
        "eps": eps,
        "profile_eps": profile_eps,
        "slots": slots,
        "n_free": len(setup.free),
        "origin": [int(o) for o in origin],
        "crop": list(crop_shape),
        "exact_crop": list(exact_shape),
        "region": list(region) if region else None,
        "capacity": list(st.capacity),
        "shape": (
            f"{'x'.join(map(str, crop_shape))}|{slots}|"
            f"{st.capacity.blocks}x{st.capacity.per_block}|"
            f"{int(crop.frozen_points.shape[0])}|{region}|"
            f"{'x'.join(str(rows_of[k]) for k in ('terms', 'relax', 'forces'))}|"
            f"{st.n_voxels}"
        ),
        "started": started,
        "endpoint_slots": int(crop.frozen_points.shape[0]),
        "ref": {k: v[0] for k, v in terms.items()},
        "grid": pick(0),
        "fd": {k: (pick(1)[k] - pick(2)[k]) / (2 * FD_STEP) for k in terms},
        "fd_offgrid": {k: (pick(3)[k] - pick(4)[k]) / (2 * FD_STEP) for k in terms},
        "history": _floats(history),
        "final": _floats(final),
        "force_rigid": {k: v[:n] for k, v in force.items()},
        "force_offgrid": {k: v[n : 2 * n] for k, v in force.items()},
        "force_profile": {k: v[2 * n : 2 * n + n_profile] for k, v in force.items()},
        "dmargin_rigid": {k: v[:n] for k, v in derivative.items()},
        "dmargin_offgrid": {k: v[n : 2 * n] for k, v in derivative.items()},
        "tangent_norms": [
            float(np.sqrt(sum((t[k] ** 2).sum() for k in t))) for t in tangents
        ],
        "escaped": escaped,
        "free_moves": moves,
        "floor": floor,
        "timing": timing,
        "groups": groups,
        "relax_rows": relax_rows,
        "relaxed": [
            {k: getattr(path, k)[r, -1] for k in FIELDS} for r in range(len(relax_rows))
        ],
        "arrays": arrays,
    }


def _trajectories(path, rows: list[dict]) -> list[dict]:
    """Per relaxation row, every step's parameters as one configuration."""
    out = []
    for r, row in enumerate(rows):
        steps = path.start.shape[1]
        config = {
            k: np.asarray(getattr(path, k)[r]).reshape(-1, *getattr(path, k).shape[3:])
            for k in FIELDS
        }
        config["live"] = np.tile(row["live"], steps)
        out.append(config)
    return out


def classify(margins: np.ndarray, snr: np.ndarray, tol: float) -> str:
    """The class of a total-loss profile over ε, from ε = 0 (Section 10.3).

    ``attracting`` if some margin is negative, ``barrier`` if it rises and
    then falls, ``flat`` if the force is weak at two neighbouring ε,
    ``restoring`` otherwise, and ``undetermined`` with fewer than two ε
    beyond 0.
    """
    if len(margins) < 3:
        return "undetermined"
    if np.any(margins[1:] < -tol):
        return "attracting"
    running = np.maximum.accumulate(margins)
    if np.any(margins < running - tol):
        return "barrier"
    weak = snr[1:] < FLAT_SNR
    if np.any(weak[:-1] & weak[1:]):
        return "flat"
    return "restoring"


def summarise(
    reference: Reference, setup: Setup, result: dict[str, Any], spec: ProbeSpec
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """One set-up's event row, its margin rows and its floor rows."""
    eps = np.array(result["eps"])
    profile_eps = list(result["profile_eps"])
    names = (*spec.terms, "total")
    final = result["final"]
    rigid = {t: result["grid"][t] - result["ref"][t] for t in names}
    profile = {t: np.full(len(eps), np.nan) for t in names}
    force_profile = {t: np.full(len(eps), np.nan) for t in names}
    for i, e in enumerate(profile_eps):
        j = list(eps).index(e)
        for t in names:
            profile[t][j] = final[t][1 + i] - final[t][0]
            force_profile[t][j] = result["force_profile"][t][i]
    n = len(eps)
    norms = np.array(result["tangent_norms"])
    snr_rigid = (
        np.abs(result["force_rigid"]["total"]) / norms[:n] / reference.grad_median
    )
    snr_profile = np.full(n, np.nan)
    for i, e in enumerate(profile_eps):
        j = list(eps).index(e)
        snr_profile[j] = (
            abs(result["force_profile"]["total"][i])
            / norms[2 * n + i]
            / reference.grad_median
        )
    tol = MARGIN_VOXELS / math.prod(reference.shape)
    valid = np.array([e not in setup.collapsed for e in eps])
    relaxed = valid & np.isfinite(profile["total"])

    def fd_agreement(where: str) -> float:
        fd = result["fd" if where == "rigid" else "fd_offgrid"]
        agree = 0
        for t in names:
            jvp = result[f"dmargin_{where}"][t]
            ok = (
                np.abs(jvp - fd[t])
                <= FD_RTOL * np.maximum(np.abs(jvp), np.abs(fd[t])) + FD_ATOL
            )
            agree += int(ok[valid].sum())
        return agree / max(len(names) * int(valid.sum()), 1)

    classes = {
        "rigid": classify(rigid["total"][valid], snr_rigid[valid], tol),
        "profile": classify(profile["total"][relaxed], snr_profile[relaxed], tol),
    }
    positive = relaxed & (eps > 0)
    loss_correct = bool(positive.any() and np.all(profile["total"][positive] > tol))
    floor_rows, floor_total = [], None
    if result["floor"]:
        a, b = len(profile_eps) + 1, len(profile_eps) + 2
        floors = {t: abs(float(final[t][a] - final[t][b])) for t in names}
        floor_total = floors["total"]
        floor_rows = [{"term": t, "floor": v} for t, v in floors.items()]
    event = setup.event
    covariates = {
        k: (",".join(map(str, v)) if isinstance(v, list | tuple) else v)
        for k, v in event.covariates.items()
    }
    row = {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "family": event.family,
        "opacity": setup.opacity,
        **{f"cov_{k}": v for k, v in covariates.items()},
        "r_unit": event.r,
        "n_free": result["n_free"],
        "slots": result["slots"],
        "crop": "x".join(map(str, result["crop"])),
        "shape": result["shape"],
        "started": result["started"],
        "exact_crop": "x".join(map(str, result["exact_crop"])),
        "class_rigid": classes["rigid"],
        "class_profile": classes["profile"],
        "loss_correct": loss_correct,
        "min_profile_margin": float(np.min(profile["total"][positive]))
        if positive.any()
        else np.nan,
        "tol": tol,
        "force_snr_min": float(np.nanmin(snr_profile[positive]))
        if positive.any()
        else np.nan,
        "fd_agree": fd_agreement("offgrid"),
        "fd_agree_on_grid": fd_agreement("rigid"),
        "collapsed": ",".join(map(str, setup.collapsed)),
        "escaped": ",".join(map(str, result["escaped"])),
        "largest_free_move": max(result["free_moves"].values(), default=0.0),
        "floor": floor_total,
        **{
            f"time_{k}": v
            for k, v in result["timing"].items()
            if isinstance(v, int | float)
        },
    }
    if floor_total is not None:
        floor_tol = max(tol, floor_total)
        row["class_profile_floor"] = classify(
            profile["total"][relaxed], snr_profile[relaxed], floor_tol
        )
        row["loss_correct_floor"] = bool(
            positive.any() and np.all(profile["total"][positive] > floor_tol)
        )
    margins = []
    for i, e in enumerate(eps):
        for t in names:
            margins.append(
                {
                    "event_id": event.event_id,
                    "family": event.family,
                    "opacity": setup.opacity,
                    "eps": float(e),
                    "term": t,
                    "rigid": float(rigid[t][i]),
                    "profile": float(profile[t][i]),
                    "force_rigid": float(result["force_rigid"][t][i]),
                    "force_offgrid": float(result["force_offgrid"][t][i]),
                    "force_profile": float(force_profile[t][i]),
                    "dmargin_rigid": float(result["dmargin_rigid"][t][i]),
                    "dmargin_offgrid": float(result["dmargin_offgrid"][t][i]),
                    "fd_rigid": float(result["fd"][t][i]),
                    "fd_offgrid": float(result["fd_offgrid"][t][i]),
                    "collapsed": bool(not valid[i]),
                }
            )
    floor_rows = [
        {
            "event_id": event.event_id,
            "family": event.family,
            "opacity": setup.opacity,
            **r,
        }
        for r in floor_rows
    ]
    return row, margins, floor_rows


# -- checks ---------------------------------------------------------------------------


def full_config(
    reference: Reference, setup: Setup, slot_params: dict
) -> tuple[dict, np.ndarray]:
    """The reference with the free segments replaced, and the new slot appended."""
    n = len(reference.raw["start"])
    raw = {k: v.copy() for k, v in reference.raw.items()}
    live = np.ones(n, bool)
    for k in FIELDS:
        raw[k][setup.free] = slot_params[k][: len(setup.free)]
    if setup.new_slot is not None:
        j = setup.new_slot
        for k in FIELDS:
            raw[k] = np.concatenate([raw[k], slot_params[k][j : j + 1]])
        live = np.append(live, bool(slot_params["live"][j]))
    return raw, live


def check_full(
    reference: Reference, setup: Setup, result: dict[str, Any], spec: ProbeSpec
) -> dict[str, float]:
    """The crop's terms, margins and first profile steps against the full volume.

    The full volume uses the library's loss functions on the full binned
    render, so this checks the crop's decomposition end to end.

    Returns
    -------
    dict
        ``term_error`` and ``margin_error`` (largest absolute differences
        over the reference and the ε grid, every term), and ``profile_error``
        (over the first :data:`CHECK_ITERS` relaxation steps at the first
        relaxed ε > 0).
    """
    names = (*spec.terms, "total")
    groups = result["groups"]
    configs = groups["ref"] + groups["grid"]
    fulls = [full_config(reference, setup, c) for c in configs]
    raws = [{k: raw[k][live] for k in FIELDS} for raw, live in fulls]
    capacity = BinCapacity(1, 1)
    for raw in raws:
        c = capacity_for(
            raw, reference.shape, reference.spacing, reference.static.render, 1.0
        )
        capacity = BinCapacity(
            max(capacity.blocks, c.blocks), max(capacity.per_block, c.per_block)
        )
    st = full_static(spec, reference.shape, reference.spacing, capacity)
    target = jnp.asarray(reference.target)
    full_values = []
    for raw, live in fulls:
        terms, overflow = full_eval(segments_of(raw), jnp.asarray(live), target, st)
        if bool(overflow):
            raise RuntimeError("the full-volume check overflowed")
        full_values.append([float(terms[t]) for t in names])
    full_values = np.array(full_values)
    crop_values = np.array([[result["ref"][t], *result["grid"][t]] for t in names]).T
    term_error = float(np.abs(crop_values - full_values).max())
    margin_error = float(
        np.abs(
            (crop_values[1:] - crop_values[0]) - (full_values[1:] - full_values[0])
        ).max()
    )
    # profile: the first steps at the first relaxed ε > 0, the same fields held
    positive = [e for e in result["profile_eps"] if e > 0]
    profile_error = np.nan
    if positive:
        e = positive[0]
        i = result["profile_eps"].index(e)
        raw, live = full_config(reference, setup, result["relax_rows"][1 + i])
        n = len(reference.raw["start"])
        size = len(live)
        frozen = {k: np.ones((size, *reference.raw[k].shape[1:]), bool) for k in FIELDS}
        held = result["arrays"].frozen
        for k in FIELDS:
            frozen[k][setup.free] = held[k][: len(setup.free)]
            if setup.new_slot is not None:
                frozen[k][n] = held[k][setup.new_slot]
        raws_live = {k: raw[k][live] for k in FIELDS}
        st = st._replace(
            capacity=capacity_for(
                raws_live, reference.shape, reference.spacing, st.render, 1.0
            )
        )
        steps = min(CHECK_ITERS, spec.profile_iterations)
        _, history, _, _, _ = relax_full(raw, live, frozen, reference.target, st, steps)
        crop_history = np.array(
            [[result["history"][t][1 + i, j] for t in names] for j in range(steps)]
        )
        full_history = np.array([[h[t] for t in names] for h in history])
        profile_error = float(np.abs(crop_history - full_history).max())
    return {
        "term_error": term_error,
        "margin_error": margin_error,
        "profile_error": profile_error,
    }


def compare_padding(
    bucketed: dict[str, Any], exact: dict[str, Any], spec: ProbeSpec
) -> dict[str, float]:
    """Largest differences of margins and forces, bucketed against exact sizes."""
    names = (*spec.terms, "total")
    rigid = max(
        float(
            np.abs(
                (bucketed["grid"][t] - bucketed["ref"][t])
                - (exact["grid"][t] - exact["ref"][t])
            ).max()
        )
        for t in names
    )
    force = max(
        float(np.abs(bucketed["force_rigid"][t] - exact["force_rigid"][t]).max())
        for t in names
    )
    rows = 1 + len(exact["profile_eps"])  # the floor's rows are not compared
    profile = max(
        float(
            np.abs(
                (bucketed["final"][t][1:rows] - bucketed["final"][t][0])
                - (exact["final"][t][1:rows] - exact["final"][t][0])
            ).max()
        )
        for t in names
    )
    return {"rigid": rigid, "force_rigid": force, "profile": profile}


def lbfgs_profile(
    reference: Reference,
    setup: Setup,
    result: dict[str, Any],
    spec: ProbeSpec,
    max_iterations: int = 2000,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Relax the probe's profile rows with float64 L-BFGS, to stationarity.

    Compares the stationary profile margins with the Adam profile's (step
    0.5's check of the relaxation length). The held fields keep a zero
    gradient, so L-BFGS never moves them.

    Returns
    -------
    dict
        Per profile ε: the L-BFGS margin, the Adam margin, L-BFGS's
        iterations and final gradient norm.
    """
    arrays = result["arrays"]
    region = tuple(result["region"]) if result["region"] else None
    origin = np.asarray(result["origin"])
    crop_shape = tuple(result["crop"])
    back = background(reference, setup)
    crop, _ = make_crop(
        reference, back, origin, crop_shape, spec.connection_reach, True
    )
    st = crop_static(reference, crop_shape, BinCapacity(*result["capacity"]), region)
    frozen = mask_tree(arrays.frozen)
    solver = optax.lbfgs()

    def loss(params, live):
        return crop_terms(params, live, crop, st)["total"]

    @jax.jit
    def step(params, state, live):
        value, grad = optax.value_and_grad_from_state(lambda p: loss(p, live))(
            params, state=state
        )
        grad = _masked(grad, frozen)
        updates, state = solver.update(
            grad,
            state,
            params,
            value=value,
            grad=grad,
            value_fn=lambda p: loss(p, live),
        )
        return optax.apply_updates(params, updates), state, value, grad

    finals, info = [], []
    for c in result["relax_rows"][: 1 + len(result["profile_eps"])]:
        params, live = stack([c])
        params = jax.tree.map(lambda x: x[0], params)
        live = live[0]
        state = solver.init(params)
        norm, iterations = np.inf, 0
        while iterations < max_iterations and norm >= tolerance:
            params, state, _, grad = step(params, state, live)
            norm = float(
                np.sqrt(sum(float((g**2).sum()) for g in jax.tree.leaves(grad)))
            )
            iterations += 1
        finals.append(float(loss(params, live)))
        info.append({"iterations": iterations, "grad_norm": norm})
    adam = result["final"]["total"]
    return {
        "eps": list(result["profile_eps"]),
        "lbfgs_margin": [f - finals[0] for f in finals[1:]],
        "adam_margin": [
            float(a - adam[0]) for a in adam[1 : 1 + len(result["profile_eps"])]
        ],
        "runs": info,
    }
