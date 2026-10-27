"""Repairs: is an error the loss's or the search's (plan, Section 10.6).

A repair asks, of an error a fitted run actually made, whether the loss
would accept its fix. The fitted segments, a jittered copy of them and the
repaired segments are relaxed together for the probe's profile length, with
the repaired segments held in all three (otherwise the relaxation undoes the
repair). The **margin** is L(repaired) - L(fitted) after the same
relaxation; the **floor** is the larger of one voxel's worth of loss and the
jittered null difference. Then:

- **search error**: the repair lowers the loss by more than the floor; the
  loss was right and the fit never got there;
- **loss error**: the repair raises the loss by more than the floor;
- **assembly**: ``q < τ`` (Section 11.2): the fitted segments link, and the
  graph stage drops the link, so no repair of the segments is run;
- **undetermined**: otherwise.

The repairs (ported from stage 0's P7):

========================  =========  ===============================================
error                     repair     what moves
========================  =========  ===============================================
``spur_end``              remove     the leaf's segments with an endpoint off the
                                     true centreline
``break_end``             join       the end's endpoint and its partner (the nearest
                                     alive endpoint in another component, on the
                                     same or a neighbouring true edge, within 6 r)
                                     to their midpoint
``break_end`` near a      reattach   the end's endpoint onto the parent's centreline
junction                             (the thickest other edge at the junction)
detached junction branch  attach     the child's endpoint nearest the junction onto
                                     the parent's centreline
``off_centre``            recentre   the deviating segments and their joints,
                                     projected onto the run's own true edge
========================  =========  ===============================================

A repair that does not remove its error **as built** (the repaired segments
re-assembled and re-scored) is **invalid**: its margin says nothing about the
error. One that moves nothing by more than :data:`NOOP` is a no-op. The same
repair found in both stages is run once. An error may have several repairs;
its class is that of its cheapest valid repair.

Everything runs in float64 on the probe's exact crops
(:mod:`spleleton.bench.probe`), with the probed loss of the run's variant.
"""

import hashlib
import math
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import jax
import numpy as np
from scipy.spatial import cKDTree

from spleleton.bench.events import (
    OFF_CENTRE,
    SAMPLE_SPACING,
    PredictedGraph,
    Scores,
    TrueCentreline,
    fit_view,
    score_sample,
    tolerance,
)
from spleleton.graph import SkeletonGraph, resample_polyline
from spleleton.primitives import SegmentArrays
from spleleton.synthetic.dataset import Sample

#: Error types with a repair, in the order repairs are taken in turn.
REPAIRS = {
    "spur_end": ("remove",),
    "break_end": ("join", "reattach"),
    "junction_detached": ("attach",),
    "off_centre": ("recentre",),
}
#: Repairs run per volume, at most (full and quick, Section 14.3).
MAX_REPAIRS = {"full": 10, "quick": 3}
#: Margins are also read at these relaxation lengths (0 is rigid).
READ_ITERS = (0, 25, 50)
#: Free segments lie within this many local radii of a repaired one.
FREE_RADII = 3.0
#: A break end's partner lies within this many local radii (at least BREAK_MIN).
BREAK_RADII, BREAK_MIN = 6.0, 4.0
#: A break end is at a junction when one of its true edge's junction nodes lies
#: within the node's radius plus this many of the edge's radii.
JUNCTION_RADII = 4.0
#: A repair that moves nothing further than this (physical units) is a no-op.
NOOP = 0.05
#: After a repair, an end of the error's type within this many radii (at least
#: :data:`FIX_MIN`) means the error is still there.
FIX_RADII, FIX_MIN = 2.0, 2.0
#: Points per segment axis when measuring its deviation from the truth.
AXIS_SAMPLES = 9
#: A margin below this many voxels' worth of loss counts as zero.
MARGIN_VOXELS = 1.0
#: How often a crop may grow to cover a relaxation that left it.
MAX_REGROW = 2
#: The classes, and the reasons an error gets none.
CLASSES = ("search", "loss", "assembly", "undetermined")


# -- repairs -----------------------------------------------------------------------


@dataclass
class Repair:
    """One repair of the fitted segments, for one or more errors.

    Attributes
    ----------
    kind : str
        ``remove``, ``join``, ``reattach``, ``attach`` or ``recentre``.
    error_type : str
        The error's type (``junction_detached`` for a detached branch).
    error_ids : list of str
        The errors it repairs (one per stage, when found in both).
    stages : list of str
        Their stages.
    position : np.ndarray
        Where the error is.
    r : float
        The true radius there: the unit of the free reach.
    cause, size
        The error's cause and size.
    q, tau : float, optional
        The decision quantity and its threshold, when the error has one.
    remove : list of int
        Segments the repair removes.
    moves : dict
        ``(segment, "start" | "end") -> new position``.
    info : dict
        What the repair found (for the table).
    status : str
        ``ok``, or why the repair is not run.
    """

    kind: str
    error_type: str
    error_ids: list[str]
    stages: list[str]
    position: np.ndarray
    r: float
    cause: str
    size: float
    q: float | None = None
    tau: float | None = None
    remove: list[int] = field(default_factory=list)
    moves: dict[tuple[int, str], np.ndarray] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    @property
    def repair_id(self) -> str:
        """``<kind>:<first error ID>``."""
        return f"{self.kind}:{self.error_ids[0]}"

    @property
    def touched(self) -> list[int]:
        """The segments the repair changes (held in every relaxation)."""
        return sorted({*self.remove, *(i for i, _ in self.moves)})

    def signature(self) -> tuple:
        """What the repair changes: equal for the same repair from two stages."""
        moves = tuple(
            sorted((i, side, *np.round(p, 6)) for (i, side), p in self.moves.items())
        )
        return (self.kind, tuple(sorted(self.remove)), moves)

    def apply(self, raw: dict[str, np.ndarray]) -> tuple[dict, np.ndarray]:
        """The segments' raw parameters with the repair applied, and which live.

        A removed segment is moved far outside the volume (the renderer draws
        every slot), and marked dead.
        """
        from spleleton.bench.probe._terms import FAR

        out = {k: v.copy() for k, v in raw.items()}
        live = np.ones(len(raw["start"]), bool)
        for (i, side), point in self.moves.items():
            out[side][i] = point
        out["start"][self.remove] = FAR
        out["end"][self.remove] = FAR
        live[self.remove] = False
        return out, live


def _endpoints(arrays: SegmentArrays, index: np.ndarray):
    """Endpoints of the given segments: points, owners and sides."""
    points = np.concatenate([arrays.start[index], arrays.end[index]])
    owners = np.concatenate([index, index])
    sides = np.array(["start"] * len(index) + ["end"] * len(index))
    return points, owners, sides


def _radius_at(true: TrueCentreline, position) -> float:
    return float(true.radii[true.tree.query(position)[1]])


def _end_node(graph: SkeletonGraph, position) -> int:
    ends = graph.end_nodes
    d = np.linalg.norm(graph.node_positions[ends] - np.asarray(position), axis=1)
    return int(ends[np.argmin(d)])


def _edge_samples(graph: SkeletonGraph, edge: int) -> np.ndarray:
    return resample_polyline(graph.edge_paths[edge], SAMPLE_SPACING)[0]


def remove_spur(error, graph, arrays, alive, true) -> tuple[dict, dict]:
    """The leaf's segments with an endpoint off the true centreline.

    The leaf's segments are the alive segments nearer the leaf edge than any
    other edge (within their radius, at least 1); only those with an endpoint
    beyond the match tolerance are removed, so that a leaf which overshoots
    along a true branch keeps its true part.
    """
    node = _end_node(graph, error["position"])
    edge = next(e for e in range(graph.n_edges) if node in graph.edges[e])
    leaf = cKDTree(_edge_samples(graph, edge))
    others = [_edge_samples(graph, e) for e in range(graph.n_edges) if e != edge]
    index = np.flatnonzero(alive)
    middle = 0.5 * (arrays.start[index] + arrays.end[index])
    d_leaf = leaf.query(middle)[0]
    d_other = (
        cKDTree(np.concatenate(others)).query(middle)[0]
        if others
        else np.full(len(index), np.inf)
    )
    reach = np.maximum(1.0, arrays.radius[index].mean(axis=1))
    on_leaf = index[(d_leaf < d_other) & (d_leaf <= reach)]
    off = []
    for i in on_leaf:
        d, j = true.tree.query(np.stack([arrays.start[i], arrays.end[i]]))
        if np.any(d > tolerance(true.radii[j])):
            off.append(int(i))
    if not off:  # the segment that ends at the spur's tip
        points, owners, _ = _endpoints(arrays, index)
        nearest = np.argmin(np.linalg.norm(points - error["position"], axis=1))
        off = [int(owners[nearest])]
    info = {
        "leaf_edge": int(edge),
        "leaf_length": float(graph.edge_lengths[edge]),
        "leaf_segments": len(on_leaf),
    }
    return {"remove": off}, info


def join_ends(error, graph, arrays, alive, true) -> tuple[dict | None, dict]:
    """The end's endpoint and its partner in another component, at their midpoint."""
    pred = PredictedGraph(graph)
    position = np.asarray(error["position"])
    component = pred.components[_end_node(graph, position)]
    index = np.flatnonzero(alive)
    points, owners, sides = _endpoints(arrays, index)
    k = int(np.argmin(np.linalg.norm(points - position, axis=1)))
    _, vertex = pred.nearest(points)
    components = pred.components[vertex]
    _, i_true = true.tree.query(np.concatenate([position[None], points]))
    edge, edges = int(true.edge[i_true[0]]), true.edge[i_true[1:]]
    graph_true = true.graph
    allowed = [
        e
        for e in range(graph_true.n_edges)
        if set(graph_true.edges[e].tolist()) & set(graph_true.edges[edge].tolist())
    ]
    r = _radius_at(true, position)
    reach = max(BREAK_MIN, BREAK_RADII * r)
    distance = np.linalg.norm(points - points[k], axis=1)
    candidate = (
        (components != component)
        & (owners != owners[k])
        & np.isin(edges, allowed)
        & (distance <= reach)
    )
    info = {"true_edge": edge, "reach": reach}
    if not candidate.any():
        return None, info
    m = int(np.flatnonzero(candidate)[np.argmin(distance[candidate])])
    middle = 0.5 * (points[k] + points[m])
    info.update(
        partner_distance=float(distance[m]),
        partner_true_edge=int(edges[m]),
        segments=(int(owners[k]), int(owners[m])),
    )
    moves = {
        (int(owners[k]), str(sides[k])): middle,
        (int(owners[m]), str(sides[m])): middle,
    }
    return {"moves": moves}, info


def _parent(true: TrueCentreline, node: int, child: int) -> int | None:
    """The thickest edge at a node other than ``child``."""
    graph = true.graph
    others = [e for e in range(graph.n_edges) if node in graph.edges[e] and e != child]
    if not others:
        return None
    return max(others, key=lambda e: float(np.mean(graph.edge_radii[e])))


def reattach_end(error, arrays, alive, true) -> tuple[dict | None, dict]:
    """A break end near a junction: its endpoint onto the parent's centreline.

    The end is at a junction when a junction node of its true edge lies
    within the node's radius plus :data:`JUNCTION_RADII` of the edge's radii.
    """
    position = np.asarray(error["position"])
    _, i = true.tree.query(position)
    edge = int(true.edge[i])
    graph = true.graph
    degree = np.bincount(graph.edges.ravel(), minlength=graph.n_nodes)
    r_edge = float(true.radii[i])
    near = [
        int(n)
        for n in graph.edges[edge]
        if degree[n] >= 3
        and np.linalg.norm(graph.node_positions[n] - position)
        <= true.node_radii[n] + JUNCTION_RADII * r_edge
    ]
    info: dict[str, Any] = {"true_edge": edge}
    if not near:
        return None, info
    node = min(near, key=lambda n: np.linalg.norm(graph.node_positions[n] - position))
    parent = _parent(true, node, edge)
    if parent is None:
        return None, info
    index = np.flatnonzero(alive)
    points, owners, sides = _endpoints(arrays, index)
    k = int(np.argmin(np.linalg.norm(points - position, axis=1)))
    target = true.points[true.nearest_on_edge(parent, points[k])]
    info.update(node=node, parent_edge=int(parent))
    return {"moves": {(int(owners[k]), str(sides[k])): target}}, info


def attach_child(event, arrays, alive, true) -> tuple[dict | None, dict]:
    """The child's endpoint nearest the junction, onto the parent's centreline."""
    index = np.flatnonzero(alive)
    middle = 0.5 * (arrays.start[index] + arrays.end[index])
    d, i = true.tree.query(middle)
    child = index[
        (true.edge[i] == int(event["edge"]))
        & (d <= np.maximum(1.5 * true.radii[i], 1.0))
    ]
    info = {
        "fit_class": event.get("fit_class"),
        "radius_ratio": event.get("radius_ratio"),
    }
    if not len(child):
        return None, info
    node = true.graph.node_positions[int(event["node"])]
    points, owners, sides = _endpoints(arrays, child)
    k = int(np.argmin(np.linalg.norm(points - node, axis=1)))
    target = true.points[true.nearest_on_edge(int(event["other"]), points[k])]
    info["end_from_node"] = float(np.linalg.norm(points[k] - node))
    return {"moves": {(int(owners[k]), str(sides[k])): target}}, info


def recentre(error, arrays, alive, true, join: float) -> tuple[dict | None, dict]:
    """Every deviating segment along the run, and its joints, onto the truth.

    Both endpoints of each alive segment that passes the run (within half its
    size plus r of its position) and deviates by more than
    :data:`~spleleton.bench.events.OFF_CENTRE` r somewhere are projected onto
    the run's own true edge, and so is every alive endpoint that assembly
    would cluster with one of them (within ``join`` times the larger radius):
    moving a segment's end alone opens its joint, whose node stays off the
    centreline. Radii are kept.
    """
    position = np.asarray(error["position"])
    r = _radius_at(true, position)
    reach = float(error["size"]) / 2 + r
    edge = int(true.edge[true.tree.query(position)[1]])
    part = true.on_edge(edge)
    edge_points, edge_radii = true.points[part], true.radii[part]
    tree = cKDTree(edge_points)
    t = np.linspace(0, 1, AXIS_SAMPLES)[:, None]
    deviation = {}
    index = np.flatnonzero(alive)
    for i in index:
        samples = arrays.start[i] + t * (arrays.end[i] - arrays.start[i])
        if np.linalg.norm(samples - position, axis=1).min() > reach:
            continue
        if true.edge[true.tree.query(0.5 * (samples[0] + samples[-1]))[1]] != edge:
            continue
        d, j = tree.query(samples)
        worst = float((d / edge_radii[j]).max())
        if worst > OFF_CENTRE:
            deviation[int(i)] = worst
    info: dict[str, Any] = {"true_edge": edge}
    if not deviation:
        return None, info
    points, owners, sides = _endpoints(arrays, index)
    radii = np.concatenate([arrays.radius[index, 0], arrays.radius[index, 1]])
    moved = np.isin(owners, list(deviation))
    distance = np.linalg.norm(points[:, None] - points[None, moved], axis=-1)
    limit = join * np.maximum(radii[:, None], radii[None, moved])
    joint = moved | (distance <= limit).any(axis=1)
    _, nearest = tree.query(points[joint])
    moves = {
        (int(i), str(side)): edge_points[k]
        for i, side, k in zip(owners[joint], sides[joint], nearest, strict=True)
    }
    info.update(
        deviating=len(deviation),
        max_deviation_over_r=max(deviation.values()),
        joint_partners=int((joint & ~moved).sum()),
    )
    return {"moves": moves}, info


@dataclass
class Fitted:
    """A run's fitted segments of one sample, with what the repairs need.

    Attributes
    ----------
    sample : Sample
    raw : dict
        The fitted segments' raw parameters, float64.
    arrays : SegmentArrays
        The same, as positions, radii and opacities.
    graphs : dict
        ``raw`` and ``clean``: the stored graphs.
    graph_config : GraphConfig
    scores : Scores
        The graphs scored (as the run's tables).
    true : TrueCentreline
    alive : np.ndarray
        The segments assembly takes (eligible and not pruned).
    links : Links or None
        The segments' link ratios (``assemble`` only), behind every ``q``.
    """

    sample: Sample
    raw: dict[str, np.ndarray]
    arrays: SegmentArrays
    graphs: dict[str, SkeletonGraph]
    graph_config: Any
    scores: Scores
    true: TrueCentreline
    alive: np.ndarray
    links: Any = None


def fitted(sample, raw, graphs, graph_config, scores=None) -> Fitted:
    """Collect a sample's fitted segments, graphs and scores."""
    raw = {k: np.asarray(v, dtype=np.float64) for k, v in raw.items()}
    arrays = SegmentArrays(
        raw["start"],
        raw["end"],
        radius=np.exp(raw["log_radius"]),
        opacity=1.0 / (1.0 + np.exp(-raw["opacity_logit"])),
    )
    true = TrueCentreline(sample.graph)
    if scores is None:
        scores = score_sample(
            sample, graphs["raw"], graphs["clean"], arrays, graph_config
        )
    view = fit_view(true, arrays, graph_config)
    return Fitted(
        sample, raw, arrays, graphs, graph_config, scores, true, view.alive,
        view.links,
    )  # fmt: skip


def break_quantity(fit: Fitted, segments: tuple[int, int]) -> float | None:
    """A break's ``q``: the bottleneck link between the end's and the partner's segment.

    The link ratio is assembly's (endpoint distance, or a T-junction's, over
    the larger radius), minimised over chains of alive segments; below
    ``join``, assembly would have linked them (plan, Section 11.2).
    """
    if fit.links is None:
        return None
    position = {int(c): k for k, c in enumerate(fit.links.candidate)}
    a, b = (position.get(i) for i in segments)
    if a is None or b is None:
        return None
    return float(fit.links.minimax[a, b])


def _skip(kind, error_type, error_id, stage, position, r, cause, size, status, **kw):
    return Repair(
        kind, error_type, [error_id], [stage], np.asarray(position, float), r,
        cause, float(size), status=status, **kw,
    )  # fmt: skip


def find_repairs(fit: Fitted) -> list[Repair]:
    """Every repair of a sample's errors, merged across stages.

    Returns runnable repairs (``status == "ok"``) first, then those that are
    not run, with the reason: ``assembly`` (``q < τ``), a no-op, or nothing
    to move.
    """
    arrays, alive, true = fit.arrays, fit.alive, fit.true
    join = float(fit.graph_config.join)
    events = [e for e in fit.scores.events if e["event_type"] == "junction"]
    found: dict[tuple, Repair] = {}
    skipped: list[Repair] = []

    def add(repair: Repair, change: dict | None) -> None:
        if change is None:
            repair.status = "nothing to move"
            skipped.append(repair)
            return
        repair.remove = list(change.get("remove", []))
        repair.moves = dict(change.get("moves", {}))
        if not repair.remove:
            largest = max(
                float(np.linalg.norm(p - getattr(arrays, side)[i]))
                for (i, side), p in repair.moves.items()
            )
            repair.info["largest_move"] = largest
            if largest < NOOP:
                repair.status = "no-op: the segments are already there"
                skipped.append(repair)
                return
        key = repair.signature()
        if key in found:
            found[key].error_ids += repair.error_ids
            found[key].stages += repair.stages
        else:
            found[key] = repair

    for stage in ("raw", "clean"):
        graph = fit.graphs[stage]
        for error in fit.scores.errors:
            if error["stage"] != stage or error["type"] not in REPAIRS:
                continue
            kind, error_id = error["type"], error["error_id"]
            position = np.asarray(error["position"], float)
            r, cause, size = _radius_at(true, position), error["cause"], error["size"]
            if kind == "break_end":
                joined, joined_info = join_ends(error, graph, arrays, alive, true)
                q, tau = None, join
                if joined is not None:
                    q = break_quantity(fit, joined_info["segments"])
                if q is not None and q < tau:
                    skipped.append(
                        _skip("join", kind, error_id, stage, position, r, cause,
                              size, "assembly", q=q, tau=tau, info=joined_info)
                    )  # fmt: skip
                    continue
                for name in ("join", "reattach"):
                    if name == "join":
                        change, info = joined, joined_info
                    else:
                        change, info = reattach_end(error, arrays, alive, true)
                        if change is None:
                            continue  # not at a junction: the join is its repair
                    repair = _skip(name, kind, error_id, stage, position, r, cause,
                                   size, "ok", q=q, tau=tau, info=info)  # fmt: skip
                    add(repair, change)
                continue
            if kind == "spur_end":
                change, info = remove_spur(error, graph, arrays, alive, true)
                name = "remove"
            else:
                change, info = recentre(error, arrays, alive, true, join)
                name = "recentre"
            add(
                _skip(name, kind, error_id, stage, position, r, cause, size, "ok",
                      info=info),
                change,
            )  # fmt: skip
        for event in events:
            if event["stage"] != stage or event.get("class") != "detached":
                continue
            node = true.graph.node_positions[int(event["node"])]
            error_id = f"{stage}/{event['event_id']}"
            cause = "near_junction"
            q, tau = event.get("q_join"), event.get("tau_join")
            r = float(event["r_edge"])
            if q is not None and tau is not None and np.isfinite(q) and q < tau:
                skipped.append(
                    _skip("attach", "junction_detached", error_id, stage, node, r,
                          cause, 0.0, "assembly", q=q, tau=tau)
                )  # fmt: skip
                continue
            change, info = attach_child(event, arrays, alive, true)
            info["event_id"] = event["event_id"]
            add(
                _skip("attach", "junction_detached", error_id, stage, node, r,
                      cause, 0.0, "ok", q=q, tau=tau, info=info),
                change,
            )  # fmt: skip
    return [*found.values(), *skipped]


def _cause_rank(cause: str) -> int:
    if cause.split("#")[0] in ("bump", "hole", "contact", "junction_gap", "gap"):
        return 0
    return 1 if cause.startswith("protrusion#") else 2


def select(repairs: list[Repair], cap: int) -> list[Repair]:
    """Up to ``cap`` runnable repairs, taken in turn from the error types.

    Each type's repairs are ordered by cause (a corruption record first, then
    a protrusion, then the rest) and size.
    """
    queues = {
        kind: sorted(
            (r for r in repairs if r.status == "ok" and r.error_type == kind),
            key=lambda r: (_cause_rank(r.cause), -r.size, r.repair_id),
        )
        for kind in REPAIRS
    }
    chosen: list[Repair] = []
    while len(chosen) < cap and any(queues.values()):
        for queue in queues.values():
            if queue and len(chosen) < cap:
                chosen.append(queue.pop(0))
    return chosen


# -- margins -----------------------------------------------------------------------


@dataclass
class Relaxed:
    """The three relaxations of one repair: fitted, jittered, repaired.

    Attributes
    ----------
    free : np.ndarray
        The free segments (slots ``0..len(free)-1``).
    slots : int
    configs : list of dict
        The three starting configurations.
    terms : dict
        Per term (and ``total``), ``(3, iterations + 1)``: before each step and
        after the last; unweighted per term, weighted in the total.
    relaxed : list of dict
        The three configurations after the relaxation.
    origin, crop, capacity
        The crop.
    escaped : bool
        A relaxed segment left the crop (after regrowing it).
    timing : dict
    """

    free: np.ndarray
    slots: int
    configs: list[dict[str, np.ndarray]]
    terms: dict[str, np.ndarray]
    relaxed: list[dict[str, np.ndarray]]
    origin: list[int]
    crop: list[int]
    capacity: list[int]
    escaped: bool
    timing: dict[str, Any]


def fit_reference(fit: Fitted, spec):
    """The fitted segments as the probe's reference (float64, the probed loss)."""
    from spleleton.bench.probe._setup import Reference, capacity_for, full_static
    from spleleton.targets import target_field

    sample = fit.sample
    shape, spacing = tuple(sample.mask.shape), tuple(float(s) for s in sample.spacing)
    target = np.asarray(
        target_field(spec.target, sample.mask, sample.spacing), dtype=np.float64
    )
    st = full_static(
        spec, shape, spacing, capacity_for(fit.raw, shape, spacing, spec.render)
    )
    return Reference(sample, target, [], fit.arrays, fit.raw, [], st, np.nan)


def _seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)


def _slots(raw, live, free, slots) -> dict[str, np.ndarray]:
    from spleleton.bench.probe._terms import DEAD_LOGIT, FAR, FIELDS

    extra = slots - len(free)
    ndim = raw["start"].shape[1]
    pad = {
        "start": np.full((extra, ndim), FAR),
        "end": np.full((extra, ndim), FAR),
        "log_radius": np.zeros((extra, 2)),
        "opacity_logit": np.full(extra, DEAD_LOGIT),
    }
    out = {k: np.concatenate([raw[k][free], pad[k]]) for k in FIELDS}
    out["live"] = np.concatenate([live[free], np.zeros(extra, bool)])
    return out


def relax(reference, spec, repair: Repair, seed: str) -> Relaxed:
    """Relax the fitted, jittered and repaired segments together on a crop."""
    from spleleton.bench.probe._crop import (
        SLOT_BUCKETS,
        background,
        bucket,
        inside_crop,
    )
    from spleleton.bench.probe._probe import _trajectories, place_crop
    from spleleton.bench.probe._setup import jittered, segment_distance
    from spleleton.bench.probe._terms import FIELDS, mask_tree, relax_batch, stack
    from spleleton.bench.timing import stopwatch
    from spleleton.render._render_jax import BinCapacity

    raw = reference.raw
    n = len(raw["start"])
    fixed, fixed_live = repair.apply(raw)
    touched = repair.touched
    paths = [(raw["start"][i], raw["end"][i]) for i in touched]
    paths += [(fixed["start"][i], fixed["end"][i]) for i in touched if fixed_live[i]]
    reach = FREE_RADII * max(repair.r, 1.0)
    free = np.array(
        sorted(
            {
                i
                for i in range(n)
                if any(
                    segment_distance(raw["start"][i], raw["end"][i], a, b) <= reach
                    for a, b in paths
                )
            }
            | set(touched)
        )
    )
    slots = bucket(len(free), SLOT_BUCKETS, 10**9)
    fitted_config = _slots(raw, np.ones(n, bool), free, slots)
    repaired = _slots(fixed, fixed_live, free, slots)
    held = np.isin(np.arange(slots), np.searchsorted(free, touched))
    held |= ~fitted_config["live"]
    ndim = raw["start"].shape[1]
    frozen = {
        "start": np.repeat(held[:, None], ndim, 1),
        "end": np.repeat(held[:, None], ndim, 1),
        "log_radius": np.repeat(~fitted_config["live"][:, None], 2, 1),
        "opacity_logit": held,
    }
    rng = np.random.default_rng(_seed(seed))
    configs = [
        fitted_config,
        jittered(fitted_config, frozen, spec.jitter, rng),
        repaired,
    ]
    frozen_tree = mask_tree(frozen)
    iterations = spec.profile_iterations
    with stopwatch() as watch:
        back = background(reference, SimpleNamespace(free=free))
        trajectory: list[dict] = []
        regrown = -1
        while regrown < MAX_REGROW:
            regrown += 1
            placed = place_crop(
                reference, back, spec, configs, slots, extent=trajectory
            )
            st = placed.static(reference)

            def run(st, placed=placed):
                out = relax_batch(
                    *stack(configs), frozen_tree, placed.crop, st=st,
                    n_iterations=iterations,
                )  # fmt: skip
                history, path, final = jax.device_get(out)
                overflow = bool(
                    np.any(history["overflow"]) or np.any(final["overflow"])
                )
                return history, path, final, overflow

            history, path, final, overflow = run(st)
            if overflow:  # the largest capacity, once
                blocks = math.prod(
                    math.ceil(s / reference.static.render.block_size)
                    for s in placed.shape
                )
                st = st._replace(capacity=BinCapacity(blocks, slots))
                history, path, final, overflow = run(st)
                if overflow:
                    raise RuntimeError(f"{repair.repair_id}: the crop overflowed")
            trajectory = _trajectories(path, configs)
            outside = [
                row
                for row in trajectory
                if not inside_crop(
                    row, row["live"], placed.origin, placed.shape, reference
                )
            ]
            if not outside:
                break
    terms = {
        t: np.concatenate([np.asarray(history[t]), np.asarray(final[t])[:, None]], 1)
        for t in (*spec.terms, "total")
    }
    relaxed = [
        {**{k: np.asarray(getattr(path, k))[r, -1] for k in FIELDS},
         "live": configs[r]["live"]}
        for r in range(3)
    ]  # fmt: skip
    return Relaxed(
        free, slots, configs, terms, relaxed,
        [int(o) for o in placed.origin], list(placed.shape), list(st.capacity),
        bool(outside),
        {"time_s": watch["wall_s"], "compile_s": watch["compile_s"],
         "programs": watch["programs"], "regrown": regrown},
    )  # fmt: skip


def classify(margin: float, null: float, n_voxels: int) -> tuple[str, float]:
    """Search, loss or undetermined, and the floor it was read against."""
    floor = max(MARGIN_VOXELS / n_voxels, abs(null))
    if margin < -floor:
        return "search", floor
    if margin > floor:
        return "loss", floor
    return "undetermined", floor


def _full(raw: dict, free: np.ndarray, config: dict) -> tuple[dict, np.ndarray]:
    from spleleton.bench.probe._terms import FIELDS

    out = {k: v.copy() for k, v in raw.items()}
    live = np.ones(len(raw["start"]), bool)
    for k in FIELDS:
        out[k][free] = config[k][: len(free)]
    live[free] = config["live"][: len(free)]
    return out, live


def _arrays(raw: dict, live: np.ndarray) -> SegmentArrays:
    return SegmentArrays(
        raw["start"][live],
        raw["end"][live],
        radius=np.exp(raw["log_radius"][live]),
        opacity=1.0 / (1.0 + np.exp(-raw["opacity_logit"][live])),
    )


def still_there(repair: Repair, scores: Scores, stage: str) -> bool:
    """Whether the repaired error is still in a stage's graph."""
    if repair.error_type == "junction_detached":
        event_id = repair.info["event_id"]
        (row,) = [
            e
            for e in scores.events
            if e["stage"] == stage and e["event_id"] == event_id
        ]
        return row["class"] != "connected"
    radius = max(FIX_MIN, FIX_RADII * repair.r)
    if repair.error_type == "off_centre":
        radius = repair.size / 2 + repair.r
    return any(
        e["stage"] == stage
        and e["type"] == repair.error_type
        and np.linalg.norm(np.asarray(e["position"]) - repair.position) <= radius
        for e in scores.errors
    )


def fixed_after(fit: Fitted, repair: Repair, raw: dict, live) -> dict[str, bool]:
    """Per stage of the error, whether it is gone from the re-assembled graphs."""
    from spleleton.pipeline import graph_stage

    arrays = _arrays(raw, live)
    sample = fit.sample
    graphs = graph_stage(arrays, None, sample.mask, sample.spacing, fit.graph_config)
    scores = score_sample(sample, *graphs, arrays, fit.graph_config)
    return {s: not still_there(repair, scores, s) for s in sorted(set(repair.stages))}


def check_full(reference, spec, run: Relaxed) -> dict[str, float]:
    """The crop's rigid terms and margin against the full volume's."""
    import jax.numpy as jnp

    from spleleton.bench.probe._setup import capacity_for
    from spleleton.bench.probe._terms import full_eval, segments_of

    values = []
    for row in (0, 2):
        raw, live = _full(reference.raw, run.free, run.configs[row])
        st = reference.static._replace(
            capacity=capacity_for(
                raw, reference.shape, reference.spacing, spec.render, 1.0
            )
        )
        terms, overflow = full_eval(
            segments_of(raw), jnp.asarray(live), jnp.asarray(reference.target), st
        )
        if bool(overflow):
            raise RuntimeError("the full-volume check overflowed")
        values.append({t: float(terms[t]) for t in (*spec.terms, "total")})
    term_error = max(
        abs(values[j][t] - run.terms[t][row, 0])
        for j, row in enumerate((0, 2))
        for t in values[0]
    )
    margin_error = max(
        abs((values[1][t] - values[0][t]) - (run.terms[t][2, 0] - run.terms[t][0, 0]))
        for t in values[0]
    )
    return {"term_error": term_error, "margin_error": margin_error}


# -- one sample --------------------------------------------------------------------


def repair_rows(
    fit: Fitted,
    spec,
    cap: int | None = None,
    seed: str = "",
    types: tuple[str, ...] | None = None,
    check: bool = False,
    progress: Callable[[str], None] | None = None,
) -> list[dict[str, Any]]:
    """Find, run and classify one sample's repairs; one row per (error, repair).

    Every error of either stage gets at least one row: errors of a type with
    no repair, and repairs not run (over the cap, assembly, a no-op, nothing
    to move), get a row with the reason in ``status`` and no margin.

    Parameters
    ----------
    fit : Fitted
        The sample's fitted segments, graphs and scores.
    spec : ProbeSpec
        The probed loss of the run's variant.
    cap : int, optional
        Repairs run, at most (all by default).
    seed : str
        Seeds the jittered null (with the repair's ID).
    types : tuple of str, optional
        Error types to repair (all of :data:`REPAIRS` by default).
    check : bool
        Also check each crop against the full volume (``crop_error``).
    progress : callable, optional
        Called with a line per repair run.

    Returns
    -------
    list of dict
        See the code; ``class`` is ``search``, ``loss``, ``assembly`` or
        ``undetermined``, or missing when the repair did not run.
    """
    started = time.perf_counter()
    repairs = find_repairs(fit)
    if types is not None:
        for repair in repairs:
            if repair.error_type not in types and repair.status == "ok":
                repair.status = "type not selected"
    chosen = select(repairs, cap if cap is not None else len(repairs))
    for repair in repairs:
        if repair.status == "ok" and all(repair is not c for c in chosen):
            repair.status = "over the cap"
    n_voxels = math.prod(fit.sample.mask.shape)
    rows = []
    with jax.enable_x64(True):
        reference = fit_reference(fit, spec) if chosen else None
        weights = dict(zip(spec.terms, spec.weights, strict=True))
        for repair in repairs:
            base = {
                "repair_id": repair.repair_id,
                "repair": repair.kind,
                "error_type": repair.error_type,
                "cause": repair.cause,
                "size": repair.size,
                "r_local": repair.r,
                "position": np.asarray(repair.position, float).tolist(),
                "q": repair.q,
                "tau": repair.tau,
                "status": repair.status,
                "removed": list(map(int, repair.remove)),
                "moved": [f"{i}:{side}" for i, side in repair.moves],
                "info": {k: _plain(v) for k, v in repair.info.items()},
            }
            if repair.status == "assembly":
                base["class"] = "assembly"
            if any(repair is c for c in chosen):
                run = relax(reference, spec, repair, f"{seed}|{repair.repair_id}")
                k = spec.profile_iterations
                total = run.terms["total"]
                for at in (*READ_ITERS, k):
                    margin = float(total[2, at] - total[0, at])
                    null = float(total[1, at] - total[0, at])
                    label, floor = classify(margin, null, n_voxels)
                    suffix = "" if at == k else f"_{at}"
                    base[f"margin{suffix}"] = margin
                    base[f"null{suffix}"] = null
                    base[f"floor{suffix}"] = floor
                    base[f"class{suffix}"] = label
                base["margin_voxels"] = base["margin"] * n_voxels
                for term in spec.terms:
                    base[f"term_{term}"] = float(
                        weights[term] * (run.terms[term][2, k] - run.terms[term][0, k])
                    )
                as_built = repair.apply(reference.raw)
                fixed = fixed_after(fit, repair, *as_built)
                relaxed = fixed_after(
                    fit, repair, *_full(reference.raw, run.free, run.relaxed[2])
                )
                base.update(
                    fixed=all(fixed.values()),
                    fixed_relaxed=all(relaxed.values()),
                    valid=all(fixed.values()),
                    n_free=len(run.free),
                    slots=run.slots,
                    crop=run.crop,
                    escaped=run.escaped,
                    time_s=run.timing["time_s"],
                    compile_s=run.timing["compile_s"],
                )
                if check:
                    errors = check_full(reference, spec, run)
                    base["crop_error"] = max(errors.values())
                if progress:
                    progress(
                        f"  {repair.kind:8s} {repair.repair_id:34s} "
                        f"{len(run.free):2d} free, {run.timing['time_s']:5.1f} s: "
                        f"margin {base['margin_voxels']:+9.2f} voxels -> "
                        f"{base['class']}{'' if base['valid'] else ' (invalid)'}"
                    )
            for error_id, stage in zip(repair.error_ids, repair.stages, strict=True):
                rows.append({"error_id": error_id, "stage": stage, **base})
    # errors with no repair
    covered = {row["error_id"] for row in rows}
    for error in fit.scores.errors:
        if error["error_id"] in covered:
            continue
        rows.append(
            {
                "error_id": error["error_id"],
                "stage": error["stage"],
                "error_type": error["type"],
                "cause": error["cause"],
                "size": float(error["size"]),
                "position": np.asarray(error["position"], float).tolist(),
                "status": "no repair for this type"
                if error["type"] not in REPAIRS
                else "type not selected",
            }
        )
    _choose(rows)
    elapsed = time.perf_counter() - started
    for row in rows:
        row["volume_s"] = elapsed
    return rows


def _choose(rows: list[dict[str, Any]]) -> None:
    """Per error, mark the repair its class comes from: the cheapest valid one.

    An error with no valid repair that ran takes ``assembly`` if that is a
    row's class, and otherwise has no class (``error_class`` is ``None``,
    and ``reason`` says why).
    """
    by_error: dict[str, list[dict]] = {}
    for row in rows:
        by_error.setdefault(row["error_id"], []).append(row)
    for group in by_error.values():
        ran = [r for r in group if r.get("valid")]
        assembly = [r for r in group if r.get("class") == "assembly"]
        best = min(ran, key=lambda r: r["margin"]) if ran else None
        if best is None and assembly:
            best = assembly[0]
        for row in group:
            row["chosen"] = row is best
            row["error_class"] = None if best is None else best["class"]
            if best is None:
                invalid = [r for r in group if "valid" in r and not r["valid"]]
                row["reason"] = (
                    "every repair run left its error in place"
                    if invalid
                    else "; ".join(sorted({r["status"] for r in group}))
                )
            else:
                row["reason"] = None


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    return value


# -- a run -------------------------------------------------------------------------


def repair_run(
    store,
    directory: Path,
    cap: int | None = MAX_REPAIRS["full"],
    types: tuple[str, ...] | None = None,
    subsets: tuple[str, ...] | None = None,
    fresh: bool = False,
    progress: Callable[[str], None] | None = print,
) -> Path:
    """``bench repair``: repair a run's errors, sample by sample (resumable).

    Writes ``parts/<sample>.repairs.parquet`` per sample and gathers them in
    the run's ``repairs.parquet``, keyed by ``(subset, seed, error_id)``.
    The repairs' settings are in ``repairs.json``; a part made with other
    settings is redone.

    Returns
    -------
    Path
        ``repairs.parquet``.
    """
    import pandas as pd

    from spleleton.bench.probe import probe_key, probe_spec
    from spleleton.bench.runner import load_fit, load_graphs, read_run
    from spleleton.bench.store import key_hash, write_json
    from spleleton.bench.suites import load_split
    from spleleton.bench.variants import Variant

    meta = read_run(directory)
    variant = Variant.from_dict(meta["spec"])
    spec = probe_spec(variant, slim=True)
    settings = {
        "cap": cap,
        "types": None if types is None else sorted(types),
        "spec": probe_key(spec),
        "read_iters": list(READ_ITERS),
        "iterations": spec.profile_iterations,
    }
    settings_key = key_hash(settings)
    parts = directory / "parts"
    parts.mkdir(exist_ok=True)
    for label, entry in meta["samples"].items():
        subset, seed = label.rsplit("_", 1)
        if entry["status"] != "done" or (subsets and subset not in subsets):
            continue
        part = parts / f"{label}.repairs.parquet"
        if part.exists() and not fresh:
            done = pd.read_parquet(part, columns=["settings_key"])
            if len(done) and (done["settings_key"] == settings_key).all():
                continue
        (loaded,) = load_split(store, meta["suite"], subset, meta["split"], [int(seed)])
        record = load_fit(store.fit_path(entry["fit_key"]))
        raw_graph, clean_graph, _ = load_graphs(store.graph_path(entry["graph_key"]))
        from spleleton.bench.probe._terms import raw_of

        fit = fitted(
            loaded.sample, raw_of(record.segments),
            {"raw": raw_graph, "clean": clean_graph}, variant.graph,
        )  # fmt: skip
        if progress:
            progress(f"{meta['name']} {label}: {len(fit.scores.errors)} errors")
        rows = repair_rows(fit, spec, cap, loaded.sha256, types, progress=progress)
        frame = pd.DataFrame(rows).assign(
            subset=subset, seed=int(seed), settings_key=settings_key
        )
        for column in ("info", "position", "removed", "moved", "crop"):
            if column in frame:
                frame[column] = frame[column].map(
                    lambda v: _json(v) if isinstance(v, (list, dict)) else None
                )
        temporary = part.with_suffix(".tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(part)
    write_json(directory / "repairs.json", settings)
    frames = [
        pd.read_parquet(p)
        for p in sorted(parts.glob("*.repairs.parquet"))
        if not subsets or p.name.rsplit("_", 1)[0] in subsets
    ]
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    path = directory / "repairs.parquet"
    table.to_parquet(path, index=False)
    return path


def _json(value: Any) -> str:
    import json

    return json.dumps(_plain(value))


def summary(table) -> dict[str, Any]:
    """Shares of each class per error type, from a run's ``repairs.parquet``.

    One count per error (its chosen repair's class), over the errors of the
    repairable types. Also the counts of repairs run, valid and escaped.
    """
    if table is None or not len(table):
        return {"errors": 0}
    errors = table.drop_duplicates(["subset", "seed", "error_id"])
    repairable = errors[errors["error_type"].isin(list(REPAIRS))]
    by_type = {
        kind: dict(Counter(group["error_class"].fillna("none")))
        for kind, group in repairable.groupby("error_type")
    }
    ran = (
        table.dropna(subset=["margin"]).drop_duplicates(["subset", "seed", "repair_id"])
        if "margin" in table
        else table.iloc[:0]
    )
    return {
        "errors": len(errors),
        "repairable": len(repairable),
        "classified": int(repairable["error_class"].notna().sum()),
        "by_type": by_type,
        "repairs_run": len(ran),
        "repairs_valid": int(ran["valid"].astype(bool).sum()) if len(ran) else 0,
        "escaped": int(ran["escaped"].astype(bool).sum()) if len(ran) else 0,
    }
