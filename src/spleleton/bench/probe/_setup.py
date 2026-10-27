"""Probed events, the relaxed reference, and each event's error path.

**Events** come from :mod:`spleleton.bench.events` (plan, Section 10.2):

- ``junction_detach``: a junction branch (thin ones only, in the slim
  probe); the edge's end at the node moves along the edge by ε r_edge;
- ``break_retract``: a continuous edge window; at its centre both segment
  ends pull back by ε r, from a break of width 0;
- ``bump_spur``: a bump, or a protrusion that holds no bump; a new segment
  from the centreline toward it, of length ε (r + height) and radius half
  its size;
- ``bump_shift``: the same events; the segment under it shifts sideways
  toward it by ε r;
- ``edge_shift``, ``duplicate``, ``radius_scale``: random mid-edge points
  (continuous windows' centres, shared by the three): a shift in a random
  direction across the edge (the control for ``bump_shift``), a copy of the
  segment shifted by ε r, and the segment's radius scaled by 1 + ε.

**The reference** is the true graph cut into pieces of about 2 local radii
(at least 4 voxels), and at every probed event, whose own pieces are kept
whole: 4 r_edge from the node along a junction branch, 4 r either side of a
window's centre (the largest ε, plus one), 1 r either side of a shifted,
duplicated or scaled point. It is relaxed on the full volume, in float64,
with the probed loss, and the pieces the families move keep their positions
and opacity (their radii relax): left free, the connection prior collapses a
thin child's pieces inside its parent into a point, which detaches the child
in the reference itself (P6).

**A set-up** is one (event, family, opacity) on the reference: its free
segments, the parameters at ε = 0, how they move with ε, and which of them
are held. In every profile the moved pieces keep their positions and opacity
and their radii relax, except in the two shift families (``bump_shift``,
``edge_shift``), whose shifted piece keeps its radius too: free, it shrinks
away while its neighbours cover the stretch, and the profile then measures
the piece's removal rather than its shift. ``radius_scale`` holds the radius
it scales, and a new primitive (spur, copy) keeps its ends and opacity.
"""

import itertools
import math
import time
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp
import numpy as np

from spleleton.bench.events import Event, TrueCentreline
from spleleton.bench.probe._spec import NEW_SLOT_FAMILIES, ProbeSpec
from spleleton.bench.probe._terms import (
    DEAD_LOGIT,
    FAR,
    FIELDS,
    Static,
    full_step,
    mask_tree,
    optimizer,
    raw_of,
    segments_of,
)
from spleleton.primitives import SegmentArrays
from spleleton.primitives._primitives_jax import JaxLineSegments
from spleleton.render._config import RenderConfig
from spleleton.render._render_jax import BinCapacity
from spleleton.synthetic.dataset import Sample

#: Reference pieces: about this many local radii long, at least MIN_PIECE.
PIECE_RADII = 2.0
MIN_PIECE = 4.0
#: Cuts stay at least this far from an edge's ends.
END_CLEARANCE = 1.0
#: A piece that retracts is (max ε + 1) radii long, or as long as its edge
#: allows, but at least this many radii (ε above its length less half a voxel
#: is then collapsed and left out).
MIN_MOVED_RADII = 1.5
#: Families that shift a piece sideways; its radius is held in the profile too.
SHIFT_FAMILIES = ("bump_shift", "edge_shift")
#: A shifted, duplicated or scaled piece spans this many radii either side.
POINT_RADII = 1.0
#: Separation between two events' pieces on one edge, in physical units.
SEPARATION = 1.0
#: Initial opacity of the reference's pieces (as ``SegmentArrays.from_graph``).
OPACITY = 0.99
#: "Opacity 1" of a new primitive: the library's concatenation clamp.
OPACITY_ONE = 1 - 1e-6
#: A spur's radius, as a fraction of the bump's (or protrusion's) size.
SPUR_RADIUS = 0.5
#: A moved segment shorter than this is collapsed at that ε.
COLLAPSE_LENGTH = 0.5
#: Headroom of measured bin capacities (the library's).
HEADROOM = 0.25


# -- polylines -------------------------------------------------------------------------


def _arc(path: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    )


def point_at(path: np.ndarray, radii: np.ndarray, s: float) -> tuple[np.ndarray, float]:
    """Point and radius at arc length ``s`` along a polyline."""
    arc = _arc(path)
    point = np.array([np.interp(s, arc, path[:, d]) for d in range(path.shape[1])])
    return point, float(np.interp(s, arc, radii))


def nearest_arc(path: np.ndarray, point: np.ndarray, step: float = 0.05) -> float:
    """Arc length of the polyline's point nearest to ``point``."""
    arc = _arc(path)
    s = np.linspace(0, arc[-1], max(2, int(arc[-1] / step) + 1))
    points = np.stack([np.interp(s, arc, path[:, d]) for d in range(path.shape[1])], 1)
    return float(s[np.argmin(np.linalg.norm(points - point, axis=1))])


def _tangent(path: np.ndarray, s: float, h: float = 0.5) -> np.ndarray:
    radii = np.zeros(len(path))
    length = _arc(path)[-1]
    a, _ = point_at(path, radii, max(0.0, s - h))
    b, _ = point_at(path, radii, min(length, s + h))
    return (b - a) / max(np.linalg.norm(b - a), 1e-12)


def _unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def _across(vector: np.ndarray, tangent: np.ndarray) -> np.ndarray:
    """``vector`` without its component along ``tangent``, normalised."""
    return _unit(vector - (vector @ tangent) * tangent)


def segment_distance(a0, a1, b0, b1, samples: int = 64) -> float:
    """Smallest distance between two segments (by sampling both)."""
    t = np.linspace(0, 1, samples)[:, None]
    pa, pb = a0 + t * (a1 - a0), b0 + t * (b1 - b0)
    return float(np.linalg.norm(pa[:, None] - pb[None], axis=-1).min())


# -- probed events ---------------------------------------------------------------------


@dataclass
class ProbeEvent:
    """An event and family to probe, located on the true graph.

    Attributes
    ----------
    family : str
        The error family.
    event_id : str
        The event's ID (Section 7.2), which joins it with run outcomes.
    event_type : str
        The event's type.
    edge : int
        The true edge it lies on.
    s : float
        Its arc position on the edge.
    r : float
        The unit of ε (the local or the edge's radius).
    intervals : list of (float, float)
        Arc intervals of the edge that must be single reference pieces.
    covariates : dict
        The event's covariates, as the event table has them.
    direction : np.ndarray, optional
        The shift or spur direction.
    node_end : {"start", "end"}, optional
        For junctions: which end of the edge lies at the node.
    spur : dict, optional
        For spurs: ``base``, ``length`` and ``radius``.
    """

    family: str
    event_id: str
    event_type: str
    edge: int
    s: float
    r: float
    intervals: list[tuple[float, float]]
    covariates: dict[str, Any]
    direction: np.ndarray | None = None
    node_end: str | None = None
    spur: dict[str, Any] | None = None


def select_events(
    sample: Sample,
    events: list[Event],
    spec: ProbeSpec,
    seed: int,
    max_events: int | None = None,
) -> tuple[list[ProbeEvent], list[dict[str, Any]]]:
    """The events to probe on a volume, and those skipped with a reason.

    At most ``max_events`` per family (``spec.max_events`` by default),
    chosen in a fixed order: junction branches by ID; bumps by ID, then the
    tallest protrusions that hold no bump; windows and mid-edge points at
    random (seeded by ``seed``, e.g. from the sample's hash), among the
    continuous windows. An event whose pieces would overlap another's on the
    same edge, or that does not fit on its edge, is skipped.
    """
    cap = spec.max_events if max_events is None else max_events
    rng = np.random.default_rng(seed)
    graph = sample.graph
    true = TrueCentreline(graph)
    cut = max(spec.eps) + 1.0

    def room(r: float, available: float) -> float | None:
        """A moved piece's length: ``cut`` radii, or what the edge allows."""
        length = min(cut * r, available)
        return length if length >= MIN_MOVED_RADII * r else None

    taken: dict[int, list[list[tuple[float, float]]]] = {}
    chosen: list[ProbeEvent] = []
    skipped: list[dict[str, Any]] = []

    def fits(edge: int, intervals: list[tuple[float, float]]) -> str | None:
        length = float(_arc(graph.edge_paths[edge])[-1])
        for lo, hi in intervals:
            for x in (lo, hi):
                inner = END_CLEARANCE <= x <= length - END_CLEARANCE
                if not (inner or np.isclose(x, 0.0) or np.isclose(x, length)):
                    return "does not fit on its edge"
        for group in taken.get(edge, []):
            if group == intervals:
                continue  # the shared mid-edge points
            for (a, b), (c, d) in itertools.product(group, intervals):
                if max(a, c) < min(b, d) + SEPARATION:
                    return "overlaps another event's pieces"
        return None

    def accept(candidate: ProbeEvent) -> bool:
        reason = (
            fits(candidate.edge, candidate.intervals) if candidate.intervals else None
        )
        if reason is not None:
            skipped.append(
                {
                    "family": candidate.family,
                    "event_id": candidate.event_id,
                    "reason": reason,
                }
            )
            return False
        if candidate.intervals:
            groups = taken.setdefault(candidate.edge, [])
            if candidate.intervals not in groups:
                groups.append(candidate.intervals)
        chosen.append(candidate)
        return True

    by_type: dict[str, list[Event]] = {}
    for event in events:
        by_type.setdefault(event.event_type, []).append(event)
    families = set(spec.families)

    # junction branches
    if "junction_detach" in families:
        count = 0
        for event in by_type.get("junction", []):
            if count >= cap:
                break
            c = event.covariates
            if spec.thin_only and not c["thin"]:
                continue
            edge, node = c["edge"], c["node"]
            length = float(_arc(graph.edge_paths[edge])[-1])
            r = c["r_edge"]
            at_start = int(graph.edges[edge][0]) == node
            piece = room(r, length - END_CLEARANCE)
            if piece is None:
                skipped.append(
                    {"family": "junction_detach", "event_id": event.event_id,
                     "reason": "its edge is too short"}
                )  # fmt: skip
                continue
            interval = (0.0, piece) if at_start else (length - piece, length)
            count += accept(
                ProbeEvent(
                    "junction_detach",
                    event.event_id,
                    "junction",
                    edge,
                    interval[0] if at_start else length,
                    r,
                    [interval],
                    dict(c),
                    node_end="start" if at_start else "end",
                )
            )

    # bumps, then the tallest protrusions without a bump
    bumpy = list(by_type.get("bump", []))
    bumpy += sorted(
        (e for e in by_type.get("protrusion", []) if not e.covariates["bump_inside"]),
        key=lambda e: -e.covariates["height"],
    )
    for family in ("bump_spur", "bump_shift"):
        if family not in families:
            continue
        count = 0
        for event in bumpy:
            if count >= cap:
                break
            c = event.covariates
            edge = int(true.edge[event.base])
            path = graph.edge_paths[edge]
            s = nearest_arc(path, true.points[event.base])
            base, _ = point_at(path, graph.edge_radii[edge], s)
            r = c["r_local"]
            direction = _across(event.position - base, _tangent(path, s))
            if family == "bump_spur":
                spur = {
                    "base": base,
                    "length": r + c["height"],
                    "radius": SPUR_RADIUS * c["size"],
                }
                count += accept(
                    ProbeEvent(
                        family, event.event_id, event.event_type, edge, s, r, [],
                        dict(c), direction=direction, spur=spur,
                    )
                )  # fmt: skip
            else:
                interval = (s - POINT_RADII * r, s + POINT_RADII * r)
                count += accept(
                    ProbeEvent(
                        family, event.event_id, event.event_type, edge, s, r,
                        [interval], dict(c), direction=direction,
                    )
                )  # fmt: skip

    # windows: breaks, then the shared mid-edge points
    windows = [e for e in by_type.get("window", []) if e.covariates["continuous"]]
    order = rng.permutation(len(windows))
    used: set[str] = set()
    if "break_retract" in families:
        count = 0
        for index in order:
            if count >= min(cap, spec.mid_edge_points):
                break
            event = windows[index]
            c = event.covariates
            s, r = 0.5 * (c["arc_start"] + c["arc_end"]), c["r_local"]
            length = float(_arc(graph.edge_paths[c["edge"]])[-1])
            piece = room(r, min(s, length - s) - END_CLEARANCE)
            if piece is None:
                skipped.append(
                    {"family": "break_retract", "event_id": event.event_id,
                     "reason": "its edge is too short"}
                )  # fmt: skip
                continue
            if accept(
                ProbeEvent(
                    "break_retract", event.event_id, "window", c["edge"], s, r,
                    [(s - piece, s), (s, s + piece)], dict(c),
                )
            ):  # fmt: skip
                count += 1
                used.add(event.event_id)
    controls = [f for f in ("edge_shift", "duplicate", "radius_scale") if f in families]
    if controls:
        count = 0
        for index in order:
            if count >= min(cap, spec.mid_edge_points):
                break
            event = windows[index]
            if event.event_id in used:
                continue
            c = event.covariates
            s, r = 0.5 * (c["arc_start"] + c["arc_end"]), c["r_local"]
            path = graph.edge_paths[c["edge"]]
            direction = _across(rng.normal(size=path.shape[1]), _tangent(path, s))
            interval = [(s - POINT_RADII * r, s + POINT_RADII * r)]
            accepted = [
                accept(
                    ProbeEvent(
                        family,
                        event.event_id,
                        "window",
                        c["edge"],
                        s,
                        r,
                        list(interval),
                        dict(c),
                        direction=direction,
                    )
                )
                for family in controls
            ]
            count += all(accepted)
    return chosen, skipped


# -- the reference ---------------------------------------------------------------------


def subdivide(
    graph, whole: dict[int, list[tuple[float, float]]]
) -> tuple[SegmentArrays, list[dict[str, Any]]]:
    """Pieces of about 2 local radii along every edge; ``whole`` intervals kept single.

    Returns the pieces as segments (opacity 0.99) and, per piece, its edge
    and arc interval.
    """
    starts, ends, radii, pieces = [], [], [], []
    for edge, (path, path_radii) in enumerate(
        zip(graph.edge_paths, graph.edge_radii, strict=True)
    ):
        arc = _arc(path)
        length = arc[-1]
        keep = sorted(set(whole.get(edge, ())))
        forced = {x for interval in keep for x in interval}

        def inside_whole(x: float, keep=keep) -> bool:
            return any(lo + 1e-9 < x < hi - 1e-9 for lo, hi in keep)

        vertices = {float(x) for x in arc[1:-1] if not inside_whole(x)}
        marks = sorted({0.0, float(length), *vertices, *forced})
        for a, b in itertools.pairwise(marks):
            if b - a < 1e-9:
                continue
            _, r_mid = point_at(path, path_radii, 0.5 * (a + b))
            single = any(np.isclose(a, lo) and np.isclose(b, hi) for lo, hi in keep)
            n = (
                1
                if single
                else max(1, round((b - a) / max(MIN_PIECE, PIECE_RADII * r_mid)))
            )
            knots = np.linspace(a, b, n + 1)
            knots[0], knots[-1] = a, b
            for lo, hi in itertools.pairwise(knots):
                p0, r0 = point_at(path, path_radii, lo)
                p1, r1 = point_at(path, path_radii, hi)
                starts.append(p0)
                ends.append(p1)
                radii.append([r0, r1])
                pieces.append({"edge": edge, "lo": float(lo), "hi": float(hi)})
    arrays = SegmentArrays(
        np.array(starts), np.array(ends), radius=np.array(radii), opacity=OPACITY
    )
    return arrays, pieces


def find_piece(pieces: list[dict[str, Any]], edge: int, lo: float, hi: float) -> int:
    """The index of the reference piece of an edge's arc interval."""
    for index, piece in enumerate(pieces):
        if (
            piece["edge"] == edge
            and np.isclose(piece["lo"], lo)
            and np.isclose(piece["hi"], hi)
        ):
            return index
    raise KeyError(f"no piece [{lo}, {hi}] on edge {edge}")


def held_pieces(event: ProbeEvent, pieces: list[dict[str, Any]]) -> list[int]:
    """The reference pieces an event's family moves (none for a spur or copy)."""
    if event.family in NEW_SLOT_FAMILIES:
        return []
    return [find_piece(pieces, event.edge, lo, hi) for lo, hi in event.intervals]


def block_ranges(start, end, radius, shape, spacing, render: RenderConfig, margin=0.0):
    """Block ranges of the padded boxes, as the renderer's ``_overlap`` bins them."""
    spacing = np.asarray(spacing, float)
    size = np.asarray(shape, float)
    lower, upper = np.minimum(start, end), np.maximum(start, end)
    if render.mode == "tube":
        reach = radius.max(axis=-1) + render.cutoff * render.wall_width + margin
    else:
        reach = np.full(len(start), render.cutoff * render.sigma + margin)
    first = np.clip(np.ceil((lower - reach[:, None]) / spacing), 0, size).astype(int)
    last = np.clip(np.floor((upper + reach[:, None]) / spacing), -1, size - 1)
    last = last.astype(int)
    inside = np.all(first <= last, axis=1)
    block = render.block_size
    return first // block, last // block, inside


def bin_need(raw, shape, spacing, render: RenderConfig) -> tuple[int, int]:
    """Active blocks, and the most segments in one block, of some segments."""
    first, last, inside = block_ranges(
        raw["start"], raw["end"], np.exp(raw["log_radius"]), shape, spacing, render
    )
    grid = tuple(math.ceil(n / render.block_size) for n in shape)
    counts = np.zeros(grid, dtype=int)
    for i in np.flatnonzero(inside):
        counts[
            tuple(slice(a, b + 1) for a, b in zip(first[i], last[i], strict=True))
        ] += 1
    return int((counts > 0).sum()), int(counts.max(initial=0))


def capacity_for(raw, shape, spacing, render, headroom=HEADROOM) -> BinCapacity:
    """The bin capacity some segments need on a grid, with headroom."""
    blocks, per_block = bin_need(raw, shape, spacing, render)
    n_blocks = math.prod(math.ceil(n / render.block_size) for n in shape)
    return BinCapacity.for_need(
        blocks, per_block, n_blocks, len(raw["start"]), headroom
    )


@dataclass
class Reference:
    """A volume's relaxed reference.

    Attributes
    ----------
    sample : Sample
        The volume.
    target : np.ndarray
        The probed loss's target field (float64).
    pieces : list of dict
        Per reference segment: its edge and arc interval.
    initial : SegmentArrays
        The cut true graph, before relaxing.
    raw : dict
        The relaxed segments' raw float64 parameters.
    held : list of int
        The pieces held while relaxing (positions and opacity).
    static : Static
        The full volume's settings.
    grad_median : float
        The median endpoint gradient norm at the relaxed reference (the
        unit of a force's signal-to-noise ratio).
    info : dict
        Loss at the start and end, timings, largest move.
    history : list of dict
        The total loss per relaxation step.
    """

    sample: Sample
    target: np.ndarray
    pieces: list[dict[str, Any]]
    initial: SegmentArrays
    raw: dict[str, np.ndarray]
    held: list[int]
    static: Static
    grad_median: float
    info: dict[str, Any] = field(default_factory=dict)
    history: list[float] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, ...]:
        """The volume's shape."""
        return tuple(self.sample.mask.shape)

    @property
    def spacing(self) -> tuple[float, ...]:
        """The volume's spacing."""
        return tuple(float(s) for s in self.sample.spacing)


def full_static(spec: ProbeSpec, shape, spacing, capacity: BinCapacity) -> Static:
    """The settings of the full volume."""
    return Static(
        tuple(int(n) for n in shape),
        tuple(float(s) for s in spacing),
        spec.render,
        capacity,
        math.prod(shape),
        spec.losses,
        spec.connection_reach,
        spec.weights,
        (spec.position_lr, spec.radius_lr, spec.opacity_lr),
    )


def hold_mask(n: int, held, radius: bool = False) -> dict[str, np.ndarray]:
    """Frozen masks holding the positions and opacity (and radii) of ``held``."""
    hold = np.zeros(n, bool)
    hold[list(held)] = True
    ndim_mask = np.repeat(hold[:, None], 3, 1)
    return {
        "start": ndim_mask.copy(),
        "end": ndim_mask.copy(),
        "log_radius": np.repeat(hold[:, None], 2, 1)
        if radius
        else np.zeros((n, 2), bool),
        "opacity_logit": hold.copy(),
    }


def relax_full(raw, live, frozen, target, st: Static, n_iterations: int):
    """Adam on the full volume, growing the bin capacity on overflow.

    Returns the relaxed raw parameters, the terms before each step, the
    settings (with the capacity that sufficed) and the terms and gradient at
    the end.
    """
    segments = segments_of(raw)
    live = jnp.asarray(live)
    frozen_tree = mask_tree(frozen)
    target = jnp.asarray(target)
    state = optimizer(st).init(segments)
    history = []
    for _ in range(n_iterations):
        while True:
            new, new_state, terms, _, overflow = full_step(
                segments, state, live, frozen_tree, target, st
            )
            if not bool(overflow):
                break
            st = st._replace(
                capacity=capacity_for(
                    raw_of(segments), st.shape, st.spacing, st.render, 1.0
                )
            )
        segments, state = new, new_state
        history.append({k: float(v) for k, v in terms.items()})
    while True:
        _, _, terms, grad, overflow = full_step(
            segments, optimizer(st).init(segments), live, frozen_tree, target, st
        )
        if not bool(overflow):
            break
        st = st._replace(
            capacity=capacity_for(
                raw_of(segments), st.shape, st.spacing, st.render, 1.0
            )
        )
    final = {k: float(v) for k, v in terms.items()}
    return raw_of(segments), history, st, final, grad


def relax_reference(
    sample: Sample,
    target: np.ndarray,
    spec: ProbeSpec,
    whole: dict[int, list[tuple[float, float]]],
    held_of: list[ProbeEvent],
    iterations: int | None = None,
) -> Reference:
    """Cut the true graph and relax it on the full volume (float64).

    Parameters
    ----------
    sample : Sample
        The volume.
    target : np.ndarray
        The probed loss's target field.
    spec : ProbeSpec
        The probe.
    whole : dict
        Per edge, the arc intervals kept as single pieces.
    held_of : list of ProbeEvent
        The events whose moved pieces are held.
    iterations : int, optional
        Adam steps (``spec.reference_iterations`` by default).
    """
    started = time.perf_counter()
    arrays, pieces = subdivide(sample.graph, whole)
    held = sorted({i for event in held_of for i in held_pieces(event, pieces)})
    segments = JaxLineSegments.from_arrays(arrays, dtype=jnp.float64)
    raw = raw_of(segments)
    n = len(segments)
    shape, spacing = tuple(sample.mask.shape), tuple(float(s) for s in sample.spacing)
    st = full_static(
        spec, shape, spacing, capacity_for(raw, shape, spacing, spec.render)
    )
    # a scaled piece's radius is held only in its own profiles
    frozen = hold_mask(n, held)
    raw, history, st, final, grad = relax_full(
        raw,
        np.ones(n, bool),
        frozen,
        target,
        st,
        spec.reference_iterations if iterations is None else iterations,
    )
    norms = np.concatenate(
        [
            np.linalg.norm(np.asarray(grad.start), axis=1),
            np.linalg.norm(np.asarray(grad.end), axis=1),
        ]
    )
    moved = float(
        np.abs(
            np.concatenate([raw["start"], raw["end"]])
            - np.concatenate([arrays.start, arrays.end])
        ).max()
    )
    info = {
        "segments": n,
        "held": len(held),
        "iterations": len(history),
        "time_s": time.perf_counter() - started,
        "loss_start": history[0]["total"] if history else final["total"],
        "loss_end": final["total"],
        "terms_end": final,
        "grad_median": float(np.median(norms)),
        "largest_endpoint_move": moved,
        "collapsed_pieces": int(
            (np.linalg.norm(raw["end"] - raw["start"], axis=1) < COLLAPSE_LENGTH).sum()
        ),
        "capacity": list(st.capacity),
    }
    return Reference(
        sample,
        np.asarray(target, dtype=np.float64),
        pieces,
        arrays,
        raw,
        held,
        st,
        float(np.median(norms)),
        info,
        [h["total"] for h in history],
    )


# -- set-ups ---------------------------------------------------------------------------


@dataclass
class Setup:
    """One (event, family, opacity) on a reference, at its exact size.

    Attributes
    ----------
    event : ProbeEvent
        The event.
    opacity : float, optional
        A new primitive's opacity (spurs and duplicates).
    free : np.ndarray
        Reference indices of the free segments (slots ``0..len(free)-1``).
    base : dict
        Slot parameters at ε = 0 (a new primitive in the last slot).
    step : dict
        ``start`` and ``end`` per unit ε.
    scale : np.ndarray
        Per slot, 1 where the log-radius grows by ``log(1 + ε)``.
    frozen : dict
        Per field, the held entries in every profile.
    moved : list of (int, str)
        The (slot, field) endpoints that the error moves (and the off-grid
        evaluation offsets).
    new_slot : int, optional
        The new primitive's slot.
    collapsed : list of float
        The ε at which a moved segment would be shorter than half a voxel.
    eps : tuple of float
        The family's ε grid.
    """

    event: ProbeEvent
    opacity: float | None
    free: np.ndarray
    base: dict[str, np.ndarray]
    step: dict[str, np.ndarray]
    scale: np.ndarray
    frozen: dict[str, np.ndarray]
    moved: list[tuple[int, str]]
    new_slot: int | None
    collapsed: list[float]
    eps: tuple[float, ...]

    @property
    def n_exact(self) -> int:
        """Slots without padding."""
        return len(self.base["start"])

    @property
    def family(self) -> str:
        """The error family."""
        return self.event.family


def make_setup(
    reference: Reference, event: ProbeEvent, spec: ProbeSpec, opacity: float | None
) -> Setup:
    """The free segments, the error path and the held fields of one event."""
    raw, pieces = reference.raw, reference.pieces
    grid = spec.family_eps(event.family)
    top = max(grid)
    family, r = event.family, event.r
    ndim = raw["start"].shape[1]
    # (reference index, field, step per unit ε) of every moved endpoint
    moves: list[tuple[int, str, np.ndarray]] = []
    scaled: list[int] = []
    paths: list[tuple[np.ndarray, np.ndarray]] = []
    reach = spec.free_radii * r
    copy_of = None
    if family == "junction_detach":
        (c,) = held_pieces(event, pieces)
        moving, other = (
            ("start", "end") if event.node_end == "start" else ("end", "start")
        )
        vector = r * _unit(raw[other][c] - raw[moving][c])
        moves = [(c, moving, vector)]
    elif family == "break_retract":
        a, b = held_pieces(event, pieces)
        moves = [
            (a, "end", r * _unit(raw["start"][a] - raw["end"][a])),
            (b, "start", r * _unit(raw["end"][b] - raw["start"][b])),
        ]
    elif family in ("bump_shift", "edge_shift"):
        (p,) = held_pieces(event, pieces)
        moves = [(p, "start", r * event.direction), (p, "end", r * event.direction)]
        paths.append((raw["start"][p], raw["end"][p]))
    elif family == "radius_scale":
        (p,) = held_pieces(event, pieces)
        scaled = [p]
        paths.append((raw["start"][p], raw["end"][p]))
    elif family == "duplicate":
        p = find_piece(pieces, event.edge, *event.intervals[0])
        copy_of = p
        paths.append((raw["start"][p], raw["end"][p]))
        shift = top * r * event.direction
        paths.append((raw["start"][p] + shift, raw["end"][p] + shift))
    elif family == "bump_spur":
        base = event.spur["base"]
        paths.append((base, base + top * event.spur["length"] * event.direction))
    else:
        raise ValueError(f"unknown family {family!r}")
    for index, name, vector in moves:
        here = raw[name][index]
        paths.append((here, here + top * vector))
    n_ref = len(raw["start"])
    free = [
        i
        for i in range(n_ref)
        if any(
            segment_distance(raw["start"][i], raw["end"][i], a, b) <= reach
            for a, b in paths
        )
    ]
    free = sorted({*free, *(m[0] for m in moves), *scaled})
    free = np.array(free, dtype=int)
    slot = {int(index): j for j, index in enumerate(free)}
    new = family in NEW_SLOT_FAMILIES
    n = len(free) + new
    base = {k: np.zeros((n, *raw[k].shape[1:])) for k in FIELDS}
    for k in FIELDS:
        base[k][: len(free)] = raw[k][free]
    step = {"start": np.zeros((n, ndim)), "end": np.zeros((n, ndim))}
    scale = np.zeros(n)
    frozen = {k: np.zeros(base[k].shape, bool) for k in FIELDS}
    moved: list[tuple[int, str]] = []
    collapsed: set[float] = set()
    for index, name, vector in moves:
        j = slot[int(index)]
        step[name][j] = vector
        # both ends and the opacity: with only the moved end held, the other
        # end slides back over the vacated stretch, or the segment fades out
        frozen["start"][j] = frozen["end"][j] = frozen["opacity_logit"][j] = True
        if family in SHIFT_FAMILIES:
            # a shifted piece could otherwise shrink away while its neighbours
            # cover the stretch, and the profile would measure its removal
            frozen["log_radius"][j] = True
        moved.append((j, name))
        if family in ("junction_detach", "break_retract"):
            length = np.linalg.norm(raw["end"][index] - raw["start"][index])
            collapsed |= {e for e in grid if length - e * r < COLLAPSE_LENGTH}
    for index in scaled:
        j = slot[int(index)]
        scale[j] = 1.0
        for k in FIELDS:
            frozen[k][j] = True
        moved += [(j, "start"), (j, "end")]
    new_slot = None
    if new:
        new_slot = len(free)
        value = OPACITY_ONE if opacity == 1.0 else opacity
        logit = np.log(value / (1 - value))
        if family == "bump_spur":
            spur = event.spur
            base["start"][new_slot] = base["end"][new_slot] = spur["base"]
            base["log_radius"][new_slot] = np.log(spur["radius"])
            step["end"][new_slot] = spur["length"] * event.direction
        else:
            for k in ("start", "end", "log_radius"):
                base[k][new_slot] = raw[k][copy_of]
            step["start"][new_slot] = step["end"][new_slot] = r * event.direction
        base["opacity_logit"][new_slot] = logit
        for k in ("start", "end", "opacity_logit"):
            frozen[k][new_slot] = True
        moved += [(new_slot, "start"), (new_slot, "end")]
    return Setup(
        event,
        opacity,
        free,
        base,
        step,
        scale,
        frozen,
        moved,
        new_slot,
        sorted(collapsed),
        grid,
    )


#: Off-grid offset of the event's points (0.3 voxel, not along an axis).
OFFGRID = 0.3 * np.array([1.0, 2.0, 2.0]) / 3.0


@dataclass
class Padded:
    """A set-up's arrays with dead slots up to a bucket size."""

    base: dict[str, np.ndarray]
    step: dict[str, np.ndarray]
    scale: np.ndarray
    frozen: dict[str, np.ndarray]
    live: np.ndarray

    @property
    def slots(self) -> int:
        """Number of slots."""
        return len(self.live)


def padded(setup: Setup, slots: int) -> Padded:
    """The set-up's arrays with ``slots - n_exact`` dead slots (held, outside)."""
    extra = slots - setup.n_exact
    ndim = setup.base["start"].shape[1]
    pad = {
        "start": np.full((extra, ndim), FAR),
        "end": np.full((extra, ndim), FAR),
        "log_radius": np.zeros((extra, 2)),
        "opacity_logit": np.full(extra, DEAD_LOGIT),
    }
    base = {k: np.concatenate([setup.base[k], pad[k]]) for k in FIELDS}
    step = {
        k: np.concatenate([setup.step[k], np.zeros((extra, ndim))]) for k in setup.step
    }
    frozen = {
        k: np.concatenate([setup.frozen[k], np.ones(pad[k].shape, bool)])
        for k in FIELDS
    }
    live = np.concatenate([np.ones(setup.n_exact, bool), np.zeros(extra, bool)])
    return Padded(
        base, step, np.concatenate([setup.scale, np.zeros(extra)]), frozen, live
    )


def config(
    arrays: Padded, setup: Setup, eps: float, offset: bool = False, new: bool = True
) -> dict[str, np.ndarray]:
    """The slot parameters at ε (off-grid with ``offset``; without the new slot)."""
    out = {k: v.copy() for k, v in arrays.base.items()}
    out["start"] = out["start"] + eps * arrays.step["start"]
    out["end"] = out["end"] + eps * arrays.step["end"]
    out["log_radius"] = out["log_radius"] + np.log1p(eps) * arrays.scale[:, None]
    if offset:
        for j, name in setup.moved:
            out[name][j] = out[name][j] + OFFGRID[: out[name].shape[1]]
    out["live"] = arrays.live.copy()
    if setup.new_slot is not None and not new:
        out["start"][setup.new_slot] = out["end"][setup.new_slot] = FAR
        out["opacity_logit"][setup.new_slot] = DEAD_LOGIT
        out["live"][setup.new_slot] = False
    return out


def tangent(arrays: Padded, eps: float) -> dict[str, np.ndarray]:
    """d(parameters)/dε at ε."""
    return {
        "start": arrays.step["start"],
        "end": arrays.step["end"],
        "log_radius": np.repeat((arrays.scale / (1 + eps))[:, None], 2, 1),
        "opacity_logit": np.zeros(arrays.slots),
    }


def jittered(
    c: dict[str, np.ndarray], frozen, scale: float, rng
) -> dict[str, np.ndarray]:
    """A configuration with noise on the endpoints that are free and live."""
    out = {k: v.copy() for k, v in c.items()}
    for name in ("start", "end"):
        noise = rng.normal(scale=scale, size=out[name].shape)
        movable = ~frozen[name] & c["live"][:, None]
        out[name] = out[name] + np.where(movable, noise, 0.0)
    return out
