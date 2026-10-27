"""Events: the places where the target errors can happen (plan, Section 7.2).

Events come from the dataset alone (the true graph, the corruption records
and the clean and corrupted masks), so their IDs are stable across runs and
across rebuilds of a suite: "which thin junctions did variant B connect that
A did not" is a join on ``event_id``. This module extracts the events with
their covariates, and scores a run's graphs (:func:`score_sample`): each
event's outcomes, the errors with their causes, and the sample's rows.

- **junction** (``junction#<node>-<edge>``): every (true junction, incident
  edge). Its covariates are the edge's radius at the node over the largest
  incident radius there (``radius_ratio``; *thin* when at most 0.5), that
  largest radius, and the angle to the thickest other edge. Its anchors lie
  ``r_node + 2 r_edge`` from the node along the edge and along the thickest
  other edge.
- **bump** (``bump#<k>``): every ``bump`` record, with its height and size
  over the local radius.
- **window** (``window#<edge>-<k>``): every true edge cut into windows of
  about 4 local radii (at least 4 voxels), with the local radius, its change,
  the curvature, the distances to the nearest junction and the nearest
  corruption record (over the radius), and the local roughness. A window is
  *continuous* when no corruption record lies within 2 local radii.
- **protrusion** (``protrusion#<k>``): every connected component of ``mask &
  ~clean_mask`` of at least 4 voxels, numbered in raster order of its first
  voxel, with its height outside the clean surface and its size over the
  local radius, and the bumps it recovers (those whose ball holds one of its
  voxels).

**Outcomes** (plan, Sections 7.1 to 7.3), per stage (the raw and the clean
graph), matched as ``compare_graphs`` matches (within ``max(1, 0.5 r)``):

- junction branch: ``connected`` (both anchors matched, and joined by a
  predicted path within ``1.5 * true_distance + 2 r_node``),
  ``anchors_matched``, ``class`` (``connected``; ``detached`` when both
  anchors matched but are not joined; otherwise ``missing``) and
  ``position_error`` of the matched predicted junction over ``r_node``;
- window: ``connected`` (its anchors snap inward to the matched samples
  nearest each end, within min(r, L / 4)), ``covered``, ``spur_ends`` and
  ``deviation`` (over r);
- bump and protrusion: ``spur`` (an unmatched predicted end off the
  centreline within ``size + height`` plus the tolerance of its centre),
  ``spur_leaf_length`` and ``deviation`` near it.

At the fit stage (the same in both stages' rows): a junction branch's
``fit_class`` (``reaches``, ``short``, ``missing``), a window's
``fit_covered``, and a bump's or protrusion's ``fit_spur`` (an alive segment
with an endpoint outside the true tube within the spur's reach); and the
decision quantities of :mod:`spleleton.bench.decisions` with their
thresholds (``tau_*``).

**Errors**: spur and break ends, missed, false-positive and off-centre runs
(with their ``source``), missed and extra junctions; each with its cause
(:class:`Causes`).

Positions are physical coordinates, as in the true graph and the corruption
records. Ported from stage 0's ``scripts/proto/p03_score.py`` (P3), without
its gap events: gaps are out of scope (plan, Section 2). The fixes stage 0
proposed are in: ``detached`` needs both anchors (P7), off-centre runs have a
``source`` (P7), break ends look for their cause along the true graph (P10),
missed runs where the mask lost the branch are ``mask_lost`` (P3), and
``fit_spur`` is looser than P3's (which flagged 1 of 4 raw spurs).
"""

import itertools
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

from spleleton.bench.decisions import (
    Links,
    assemble_settings,
    connection_quantities,
    segment_links,
    spur_quantities,
)
from spleleton.graph import SkeletonGraph, resample_polyline
from spleleton.metrics import _match, compare_graphs
from spleleton.primitives import SegmentArrays
from spleleton.primitives._arrays import as_segment_arrays
from spleleton.synthetic.dataset import Sample

#: Arc-length spacing of the sampled true centreline (as ``compare_graphs``).
SAMPLE_SPACING = 0.25
#: A junction branch is thin when its radius over the largest is at most this.
THIN_RATIO = 0.5
#: Windows are about this many local radii long ...
WINDOW_RADII = 4.0
#: ... and at least this many voxels.
WINDOW_MIN_VOXELS = 4.0
#: Continuity and causes look this many local radii around a window or error.
K_RADII = 2.0
#: Protrusion components smaller than this many voxels are not events.
MIN_PROTRUSION_VOXELS = 4
#: Corruption records without a position (not located).
_UNLOCATED = ("surface_noise",)
_EPS = 1e-9


# -- the true centreline --------------------------------------------------------------


class TrueCentreline:
    """The true graph sampled every :data:`SAMPLE_SPACING` along its edges.

    Attributes
    ----------
    graph : SkeletonGraph
        The true graph.
    points, radii : np.ndarray
        The samples (``(N, D)``) and their radii, edge by edge.
    edge : np.ndarray
        The edge of each sample.
    arc : np.ndarray
        Each sample's arc length from its edge's first node.
    offsets : np.ndarray
        Edge ``e``'s samples are ``offsets[e]:offsets[e + 1]``.
    lengths : np.ndarray
        Each edge's length.
    tree : cKDTree
        Over ``points``.
    """

    def __init__(self, graph: SkeletonGraph) -> None:
        self.graph = graph
        points, radii, edge, arc = [], [], [], []
        for index, path in enumerate(graph.edge_paths):
            p, r = resample_polyline(path, SAMPLE_SPACING, graph.edge_radii[index])
            points.append(p)
            radii.append(r)
            edge.append(np.full(len(p), index))
            steps = np.linalg.norm(np.diff(p, axis=0), axis=1)
            arc.append(np.concatenate([[0.0], np.cumsum(steps)]))
        self.points = np.concatenate(points)
        self.radii = np.concatenate(radii)
        self.edge = np.concatenate(edge)
        self.arc = np.concatenate(arc)
        self.offsets = np.concatenate([[0], np.cumsum([len(p) for p in points])])
        self.lengths = np.array([a[-1] for a in arc])
        self.tree = cKDTree(self.points)
        self.node_radii = graph.node_radii()
        self._links = None

    def on_edge(self, edge: int) -> slice:
        """The samples of one edge."""
        return slice(int(self.offsets[edge]), int(self.offsets[edge + 1]))

    def at_arc(self, edge: int, s: float) -> int:
        """Index of the sample of ``edge`` nearest to arc length ``s``."""
        part = self.on_edge(edge)
        return part.start + int(np.argmin(np.abs(self.arc[part] - s)))

    def nearest_on_edge(self, edge: int, point: np.ndarray) -> int:
        """Index of the sample of ``edge`` nearest to a point."""
        part = self.on_edge(edge)
        d = np.linalg.norm(self.points[part] - np.asarray(point), axis=1)
        return part.start + int(np.argmin(d))

    def from_node(self, edge: int, node: int, distance: float) -> int:
        """The sample ``distance`` along ``edge`` from its end at ``node``."""
        u, _ = self.graph.edges[edge]
        length = self.lengths[edge]
        distance = min(distance, length)
        return self.at_arc(edge, distance if u == node else length - distance)

    def direction_from(self, edge: int, node: int, distance: float = 2.0) -> np.ndarray:
        """Unit vector from ``node`` along ``edge``."""
        here = self.graph.node_positions[node]
        there = self.points[self.from_node(edge, node, distance)]
        vector = there - here
        return vector / max(np.linalg.norm(vector), _EPS)

    def geodesic(self, index: int, limit: float) -> np.ndarray:
        """Distance along the true graph from one sample to every other.

        Paths run along the edges and through the nodes; samples further
        than ``limit`` get ``inf``.
        """
        if self._links is None:
            rows, cols, weights = [], [], []
            for edge in range(self.graph.n_edges):
                part = self.on_edge(edge)
                ids = np.arange(part.start, part.stop)
                rows += list(ids[:-1])
                cols += list(ids[1:])
                weights += list(np.maximum(np.diff(self.arc[part]), _EPS))
            for node in range(self.graph.n_nodes):
                ends = [
                    self.on_edge(e).start if u == node else self.on_edge(e).stop - 1
                    for e, (u, v) in enumerate(self.graph.edges)
                    if node in (u, v)
                ]
                for a, b in itertools.combinations(ends, 2):
                    rows.append(a)
                    cols.append(b)
                    weights.append(_EPS)
            n = len(self.points)
            self._links = coo_matrix(
                (weights + weights, (rows + cols, cols + rows)), shape=(n, n)
            ).tocsr()
        return dijkstra(self._links, directed=False, indices=index, limit=limit)


# -- corruption records ---------------------------------------------------------------


@dataclass(frozen=True)
class LocatedRecord:
    """A corruption record with a position: a segment ``start -> end`` and an extent.

    Attributes
    ----------
    id : str
        ``<type>#<k>``, numbered per type in the records' order.
    type : str
        The record's type.
    start, end : np.ndarray
        The record's axis (equal for point-like records).
    extent : float
        How far the corruption reaches from its axis.
    record : dict
        The record itself.
    """

    id: str
    type: str
    start: np.ndarray
    end: np.ndarray
    extent: float
    record: dict

    def distance(self, points: np.ndarray) -> np.ndarray:
        """Distance from points to the record's axis, minus its extent."""
        points = np.atleast_2d(points)
        vector = self.end - self.start
        t = np.clip(((points - self.start) @ vector) / max(vector @ vector, _EPS), 0, 1)
        closest = self.start + t[:, None] * vector
        return np.linalg.norm(closest - points, axis=1) - self.extent


def located_records(sample: Sample) -> list[LocatedRecord]:
    """The sample's corruption records that have a position, numbered per type."""
    out, counter = [], {}
    for record in sample.corruptions:
        kind = record["type"]
        if kind in _UNLOCATED:
            continue
        index = counter.get(kind, 0)
        counter[kind] = index + 1
        if kind == "contact":
            start, end = np.array(record["start"]), np.array(record["end"])
            extent = float(record["radius"])
        else:
            start = end = np.array(record["position"], dtype=float)
            extent = {
                "gap": record.get("width", 0) / 2,
                "junction_gap": record.get("width", 0) / 2,
                "bump": record.get("radius", 0),
                "hole": record.get("radius", 0),
            }.get(kind, 0.0)
        out.append(LocatedRecord(f"{kind}#{index}", kind, start, end, extent, record))
    return out


# -- events ---------------------------------------------------------------------------


@dataclass
class Event:
    """One event of a sample, with its covariates (no outcomes yet).

    Attributes
    ----------
    event_type : {"junction", "bump", "window", "protrusion"}
        The kind of event.
    event_id : str
        Stable across runs and rebuilds (see the module's docstring).
    position : np.ndarray
        A representative point (physical): the junction's node, the bump's
        centre, the window's middle sample, the protrusion's tip.
    covariates : dict
        Plain values (floats, ints, strings, bools, lists of strings).
    region : np.ndarray
        Indices of the true samples the event concerns (the junction branch's
        edge, the window's samples); empty for bumps and protrusions.
    anchors : tuple of int, optional
        Two true samples that a correct prediction connects (junctions and
        windows).
    true_distance : float, optional
        The true distance between the anchors.
    slack : float, optional
        What a predicted path between the anchors may add to ``1.5 *
        true_distance`` (2 r_node for junctions, 2 r for windows).
    base : int, optional
        The true sample nearest to a bump or a protrusion.
    """

    event_type: str
    event_id: str
    position: np.ndarray
    covariates: dict[str, Any]
    region: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    anchors: tuple[int, int] | None = None
    true_distance: float | None = None
    slack: float | None = None
    base: int | None = None

    def row(self) -> dict[str, Any]:
        """The event as a table row: type, ID, position and covariates."""
        return {
            "event_type": self.event_type,
            "event_id": self.event_id,
            "position": [float(x) for x in self.position],
            **self.covariates,
        }


def extract_events(sample: Sample, true: TrueCentreline | None = None) -> list[Event]:
    """Every event of a sample, in a fixed order.

    Junction branches, bumps, protrusions, then windows; within each type, in
    the order of their IDs' construction (nodes and edges, records, raster
    order, edges and arc length).

    Parameters
    ----------
    sample : Sample
        The sample.
    true : TrueCentreline, optional
        Its sampled true graph, if already built.

    Returns
    -------
    list of Event
    """
    true = TrueCentreline(sample.graph) if true is None else true
    records = located_records(sample)
    spacing = np.asarray(sample.spacing, dtype=float)
    return [
        *junction_events(true),
        *bump_events(true, records),
        *protrusion_events(sample, true, records, spacing),
        *window_events(sample, true, records, spacing),
    ]


def event_rows(events: list[Event]) -> list[dict[str, Any]]:
    """The events as table rows (see :meth:`Event.row`)."""
    return [event.row() for event in events]


def junction_events(true: TrueCentreline) -> list[Event]:
    """One event per (true junction, incident edge)."""
    graph = true.graph
    node_radii = graph.node_radii()
    events = []
    for node in graph.junction_nodes:
        node = int(node)
        incident = [e for e in range(graph.n_edges) if node in graph.edges[e]]
        # each edge's radius where it meets the node
        r_at = {e: float(true.radii[true.from_node(e, node, 0.0)]) for e in incident}
        r_max = max(r_at.values())
        r_node = float(node_radii[node])
        for edge in incident:
            # the thickest other edge (the lowest index among equals)
            other = min((e for e in incident if e != edge), key=lambda e: -r_at[e])
            r_edge = r_at[edge]
            distance = r_node + 2 * r_edge
            a = true.from_node(edge, node, distance)
            b = true.from_node(other, node, distance)
            cos = true.direction_from(edge, node) @ true.direction_from(other, node)
            angle = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
            position = graph.node_positions[node]
            part = true.on_edge(edge)
            events.append(
                Event(
                    "junction",
                    f"junction#{node}-{edge}",
                    position,
                    {
                        "node": node,
                        "edge": int(edge),
                        "other": int(other),
                        "degree": len(incident),
                        "r_edge": r_edge,
                        "r_node": r_node,
                        "r_max": r_max,
                        "radius_ratio": r_edge / r_max,
                        "thin": r_edge / r_max <= THIN_RATIO,
                        "angle": angle,
                    },
                    region=np.arange(part.start, part.stop),
                    anchors=(a, b),
                    true_distance=float(
                        np.linalg.norm(true.points[a] - position)
                        + np.linalg.norm(true.points[b] - position)
                    ),
                    slack=2 * r_node,
                )
            )
    return events


def bump_events(true: TrueCentreline, records: list[LocatedRecord]) -> list[Event]:
    """One event per ``bump`` record."""
    events = []
    for record in records:
        if record.type != "bump":
            continue
        edge = int(record.record["edge"])
        base = true.nearest_on_edge(edge, record.start)
        r = float(true.radii[base])
        height, size = float(record.record["height"]), float(record.record["radius"])
        events.append(
            Event(
                "bump",
                record.id,
                record.start,
                {
                    "edge": edge,
                    "r_local": r,
                    "height": height,
                    "size": size,
                    "height_over_r": height / r,
                    "size_over_r": size / r,
                },
                base=base,
            )
        )
    return events


def protrusion_events(
    sample: Sample,
    true: TrueCentreline,
    records: list[LocatedRecord],
    spacing: np.ndarray,
) -> list[Event]:
    """One event per component of ``mask & ~clean_mask`` (at least 4 voxels)."""
    components = protrusion_components(sample)
    if not components:
        return []
    outside = ndimage.distance_transform_edt(~sample.clean_mask, sampling=spacing)
    voxel_volume = float(np.prod(spacing))
    bumps = [r for r in records if r.type == "bump"]
    events = []
    for index in components:
        voxels = index * spacing
        heights = outside[tuple(index.T)]
        tip = voxels[np.argmax(heights)]
        _, base = true.tree.query(tip)
        r = float(true.radii[base])
        height = float(heights.max())
        size = float((3 * len(voxels) * voxel_volume / (4 * np.pi)) ** (1 / 3))
        tree = cKDTree(voxels)
        # a bump's ball (centre ``start``, radius ``extent``) holds one of the
        # component's voxels; its centre can lie inside the clean tube
        recovered = [b.id for b in bumps if tree.query(b.start)[0] <= b.extent]
        events.append(
            Event(
                "protrusion",
                f"protrusion#{len(events)}",
                tip,
                {
                    "voxels": len(voxels),
                    "centre": [float(x) for x in voxels.mean(axis=0)],
                    "r_local": r,
                    "height": height,
                    "size": size,
                    "height_over_r": height / r,
                    "size_over_r": size / r,
                    "bump_inside": recovered,
                },
                base=int(base),
            )
        )
    return events


def protrusion_components(sample: Sample) -> list[np.ndarray]:
    """The voxel indices of each protrusion, in the order of their IDs.

    The components of ``mask & ~clean_mask`` (26-connected) of at least
    :data:`MIN_PROTRUSION_VOXELS` voxels, in raster order of their first
    voxel, each with its voxels in raster order.
    """
    added = sample.mask & ~sample.clean_mask
    structure = np.ones((3,) * added.ndim, dtype=bool)
    labels, n = ndimage.label(added, structure=structure)
    if n == 0:
        return []
    index = np.argwhere(labels)
    label = labels[tuple(index.T)]
    order = np.argsort(label, kind="stable")  # raster order within a component
    groups = np.split(
        index[order], np.cumsum(np.bincount(label, minlength=n + 1))[1:-1]
    )
    return [g for g in groups if len(g) >= MIN_PROTRUSION_VOXELS]


def window_events(
    sample: Sample,
    true: TrueCentreline,
    records: list[LocatedRecord],
    spacing: np.ndarray,
) -> list[Event]:
    """Every true edge cut into windows of about 4 local radii."""
    graph = true.graph
    voxel = float(np.prod(spacing) ** (1 / len(spacing)))
    windows = []
    for edge in range(graph.n_edges):
        part = true.on_edge(edge)
        arc, length = true.arc[part], true.lengths[edge]
        s, index = 0.0, 0
        while s < length - _EPS:
            r_here = float(true.radii[true.at_arc(edge, s)])
            step = max(WINDOW_MIN_VOXELS * voxel, WINDOW_RADII * r_here)
            end = min(s + step, length)
            if length - end < 0.5 * step:  # no short last window: extend this one
                end = length
            members = np.arange(part.start, part.stop)[
                (arc >= s - _EPS) & (arc <= end + _EPS)
            ]
            windows.append((edge, index, s, end, members))
            s, index = end, index + 1
    # roughness: the voxels of mask XOR clean_mask, each given to the window of
    # its nearest true sample if it lies within that sample's r + 2 voxels
    window_of = np.full(len(true.points), -1)
    for number, (*_, members) in enumerate(windows):
        window_of[members] = number
    changed = np.argwhere(sample.mask ^ sample.clean_mask) * spacing
    if len(changed):
        d, nearest = true.tree.query(changed)
        near = d <= true.radii[nearest] + 2 * voxel
        counts = np.bincount(window_of[nearest[near]], minlength=len(windows))
    else:
        counts = np.zeros(len(windows), dtype=int)
    voxel_volume = float(np.prod(spacing))
    junctions = graph.node_positions[graph.junction_nodes]
    events = []
    for number, (edge, index, s, end, members) in enumerate(windows):
        points, radii = true.points[members], true.radii[members]
        r = float(radii.mean())
        length = end - s
        centre = points[len(points) // 2]
        # the turn between the tangents at the window's two ends, per length
        t0 = points[min(4, len(points) - 1)] - points[0]
        t1 = points[-1] - points[max(len(points) - 5, 0)]
        cos = t0 @ t1 / max(np.linalg.norm(t0) * np.linalg.norm(t1), _EPS)
        curvature = float(np.arccos(np.clip(cos, -1, 1)) / max(length, _EPS))
        junction_distance = (
            float(np.linalg.norm(junctions - centre, axis=1).min() / r)
            if len(junctions)
            else np.inf
        )
        record_distance, nearest_record = np.inf, None
        for record in records:
            distance = float(record.distance(points).min()) / r
            if distance < record_distance:
                record_distance, nearest_record = distance, record.id
        surface = 2 * np.pi * r * max(length, _EPS)
        events.append(
            Event(
                "window",
                f"window#{edge}-{index}",
                centre,
                {
                    "edge": edge,
                    "index": index,
                    "arc_start": float(s),
                    "arc_end": float(end),
                    "length": float(length),
                    "r_local": r,
                    "radius_change": float(radii[-1] - radii[0]),
                    "curvature": curvature,
                    "junction_distance_over_r": junction_distance,
                    "record_distance_over_r": record_distance,
                    "nearest_record": nearest_record,
                    "continuous": bool(record_distance > K_RADII),
                    "roughness": float(counts[number]) * voxel_volume / surface,
                },
                region=members,
                anchors=(int(members[0]), int(members[-1])),
                true_distance=float(length),
                slack=2 * r,
            )
        )
    return events


# == scoring: outcomes, errors and sample rows (step 0.7) ==============================

#: Match tolerance ``max(MIN_TOLERANCE, RELATIVE_TOLERANCE * r)``, and the
#: junction and end tolerances in radii (as ``compare_graphs``'s defaults).
RELATIVE_TOLERANCE, MIN_TOLERANCE = 0.5, 1.0
JUNCTION_TOLERANCE, END_TOLERANCE = 1.0, 1.5
#: Off-centre: within the match tolerance, but further than this many radii.
OFF_CENTRE = 0.25
#: Error runs (missed, false positive, off-centre) shorter than this are dropped.
MIN_RUN = 1.0
#: A missed run is ``mask_lost`` when less than this share lies in the mask's
#: main component.
MASK_LOST = 0.5
#: The graphs that are scored.
STAGES = ("raw", "clean")
#: The error types, in the order the error table lists them.
ERROR_TYPES = (
    "spur_end",
    "break_end",
    "missed",
    "false_positive",
    "off_centre",
    "junction_missed",
    "junction_extra",
)
#: Causes that make a break end an algorithmic break ("breaks in edges").
ALGORITHMIC_BREAK = ("unexplained", "near_junction")


def tolerance(radius, relative: float = RELATIVE_TOLERANCE) -> np.ndarray:
    """The match tolerance of ``compare_graphs`` at a true radius."""
    return np.maximum(MIN_TOLERANCE, relative * np.asarray(radius, dtype=float))


class PredictedGraph:
    """A predicted graph sampled every :data:`SAMPLE_SPACING`, for paths.

    Its vertices are the graph's nodes followed by the interior samples of
    each edge; consecutive vertices along an edge are linked by their
    distance, so that paths between any two vertices are Dijkstra paths.

    Attributes
    ----------
    graph : SkeletonGraph
    points : np.ndarray
        The vertices.
    edge : np.ndarray
        The edge of each vertex (-1 for the graph's nodes).
    components : np.ndarray
        The connected component of each vertex.
    tree : cKDTree or None
        Over ``points`` (``None`` for an empty graph).
    """

    def __init__(self, graph: SkeletonGraph) -> None:
        self.graph = graph
        points = [graph.node_positions]
        edge_of = [np.full(graph.n_nodes, -1)]
        rows, cols, weights = [], [], []
        next_id = graph.n_nodes
        for index, (u, v) in enumerate(graph.edges):
            p, _ = resample_polyline(graph.edge_paths[index], SAMPLE_SPACING)
            interior = p[1:-1]
            ids = np.concatenate(
                [[u], np.arange(next_id, next_id + len(interior)), [v]]
            ).astype(int)
            next_id += len(interior)
            points.append(interior)
            edge_of.append(np.full(len(interior), index))
            chain = np.concatenate(
                [[graph.node_positions[u]], interior, [graph.node_positions[v]]]
            )
            rows += list(ids[:-1])
            cols += list(ids[1:])
            steps = np.linalg.norm(np.diff(chain, axis=0), axis=1)
            weights += list(np.maximum(steps, _EPS))
        self.points = np.concatenate(points).reshape(-1, graph.ndim)
        self.edge = np.concatenate(edge_of)
        n = len(self.points)
        self.matrix = coo_matrix(
            (weights + weights, (rows + cols, cols + rows)), shape=(n, n)
        ).tocsr()
        self.components = (
            connected_components(self.matrix, directed=False)[1]
            if n
            else np.zeros(0, dtype=int)
        )
        self.tree = cKDTree(self.points) if n else None

    def nearest(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Distance to, and index of, the nearest vertex (``inf`` and -1 if none)."""
        points = np.atleast_2d(points)
        if self.tree is None:
            return np.full(len(points), np.inf), np.full(len(points), -1)
        return self.tree.query(points)

    def path_length(self, a: int, b: int, limit: float) -> float:
        """The shortest path between two vertices (``inf`` beyond ``limit``)."""
        if a < 0 or b < 0:
            return np.inf
        if a == b:
            return 0.0
        return float(dijkstra(self.matrix, directed=False, indices=a, limit=limit)[b])


@dataclass
class End:
    """A predicted end that matches no true end: a spur tip or a break end.

    Attributes
    ----------
    type : {"spur_end", "break_end"}
        ``break_end`` when it lies on the true centreline (within tolerance).
    node : int
        The predicted node.
    position : np.ndarray
    leaf_length : float
        The length of the end's edge.
    true_index : int
        The nearest true sample.
    break_length : float
        The distance to the nearest predicted point in another component
        (the far side of the break), or 0 if there is none.
    """

    type: str
    node: int
    position: np.ndarray
    leaf_length: float
    true_index: int
    break_length: float


@dataclass
class StageView:
    """One predicted graph matched against the truth, as ``compare_graphs`` does.

    Attributes
    ----------
    pred : PredictedGraph
    matched_true : np.ndarray
        Per true sample: a predicted point lies within tolerance.
    nearest_pred : np.ndarray
        Per true sample: the nearest predicted vertex.
    pred_distance, pred_nearest_true : np.ndarray
        Per predicted vertex: the distance to, and index of, the nearest true
        sample.
    ends : list of End
    junction_match : dict
        True junction node -> matched predicted junction node.
    junction_missed, junction_extra : np.ndarray
        Unmatched true and predicted junction nodes.
    """

    pred: PredictedGraph
    matched_true: np.ndarray
    nearest_pred: np.ndarray
    pred_distance: np.ndarray
    pred_nearest_true: np.ndarray
    ends: list[End]
    junction_match: dict[int, int]
    junction_missed: np.ndarray
    junction_extra: np.ndarray


def stage_view(graph: SkeletonGraph, true: TrueCentreline) -> StageView:
    """Match a predicted graph against the truth."""
    pred = PredictedGraph(graph)
    d_tp, i_tp = pred.nearest(true.points)
    matched = d_tp <= tolerance(true.radii)
    if len(pred.points):
        d_pt, i_pt = true.tree.query(pred.points)
    else:
        d_pt, i_pt = np.zeros(0), np.zeros(0, dtype=int)
    target = true.graph
    ndim = target.ndim
    pred_ends = graph.end_nodes
    target_ends = target.end_nodes
    matches = _match(
        graph.node_positions[pred_ends].reshape(-1, ndim),
        target.node_positions[target_ends],
        tolerance(true.node_radii[target_ends], END_TOLERANCE),
    )
    matched_ends = {i for i, _ in matches}
    lengths = graph.edge_lengths
    ends = []
    for k, node in enumerate(pred_ends):
        if k in matched_ends:
            continue
        position = graph.node_positions[node]
        d, i = true.tree.query(position)
        kind = "break_end" if d <= tolerance(true.radii[i]) else "spur_end"
        edge = next(e for e in range(graph.n_edges) if node in graph.edges[e])
        others = pred.components != pred.components[node]
        partner = (
            float(np.linalg.norm(pred.points[others] - position, axis=1).min())
            if others.any()
            else 0.0
        )
        ends.append(
            End(kind, int(node), position, float(lengths[edge]), int(i), partner)
        )
    pj, tj = graph.junction_nodes, target.junction_nodes
    junctions = _match(
        graph.node_positions[pj].reshape(-1, ndim),
        target.node_positions[tj],
        tolerance(true.node_radii[tj], JUNCTION_TOLERANCE),
    )
    return StageView(
        pred,
        matched,
        i_tp,
        d_pt,
        i_pt,
        ends,
        {int(tj[j]): int(pj[i]) for i, j in junctions},
        tj[np.setdiff1d(np.arange(len(tj)), [j for _, j in junctions])],
        pj[np.setdiff1d(np.arange(len(pj)), [i for i, _ in junctions])],
    )


def root_fraction(view: StageView, true: TrueCentreline) -> float:
    """The share of the true centreline matched within the root's component.

    The root is the true graph's first node; its component is that of the
    predicted vertex nearest to it. A break weighs by how much tree it cuts
    off.
    """
    if view.pred.tree is None:
        return 0.0
    _, vertex = view.pred.tree.query(true.graph.node_positions[0])
    same = view.pred.components[view.nearest_pred] == view.pred.components[vertex]
    return float(np.mean(view.matched_true & same))


# -- the fitted segments ---------------------------------------------------------------


@dataclass
class FitView:
    """The fitted segments against the truth: the fit-stage outcomes' inputs.

    Attributes
    ----------
    segments : SegmentArrays
        Every fitted segment.
    eligible : np.ndarray
        Segments that pass the assembly's radius filter (all of them for
        methods without one).
    alive : np.ndarray
        Eligible and not pruned (opacity at least ``prune_threshold``).
    inside : np.ndarray
        Per true sample: inside some alive fitted tube.
    end_outside : np.ndarray
        ``(N, 2)``: each endpoint lies outside the true tube (further from
        the true centreline than the true radius there).
    axis_points, axis_deviation : np.ndarray
        The alive segments' axes sampled every :data:`SAMPLE_SPACING`, and
        each sample's distance to the true centreline over the true radius.
    links : Links or None
        The segment links of ``assemble`` (``None`` for other methods).
    settings : dict
        The graph stage's thresholds (``tau``) that the events compare with.
    """

    segments: SegmentArrays
    eligible: np.ndarray
    alive: np.ndarray
    inside: np.ndarray
    end_outside: np.ndarray
    axis_points: np.ndarray
    axis_deviation: np.ndarray
    links: Links | None
    settings: dict[str, float]


def fit_view(true: TrueCentreline, segments: Any, config: Any) -> FitView:
    """The fitted segments of a run, with the graph stage's settings.

    Parameters
    ----------
    true : TrueCentreline
        The sampled true graph.
    segments : SegmentArrays or JaxLineSegments
        The fitted segments.
    config : GraphConfig
        The graph stage's settings.
    """
    segments = as_segment_arrays(segments)
    prune = float(config.prune_threshold)
    eligible = np.ones(len(segments), dtype=bool)
    links, settings = None, {"prune_threshold": prune}
    if config.method == "assemble":
        settings = assemble_settings(config)
        eligible = segments.radius.mean(axis=1) >= settings["min_radius"]
        links = segment_links(segments, **settings)
        clean = config.clean
        settings |= {
            "bridge_reach": clean.bridge_reach,
            "bridge_angle": clean.bridge_angle,
            "min_protrusion": clean.min_protrusion,
        }
    alive = eligible & (np.asarray(segments.opacity) >= prune)
    start, end, radius = (
        segments.start[alive],
        segments.end[alive],
        segments.radius[alive],
    )
    # inside some alive tube, per true sample
    inside = np.zeros(len(true.points), dtype=bool)
    if alive.any():
        vector = end - start
        squared = np.maximum((vector**2).sum(1), _EPS)
        for chunk in np.array_split(
            np.arange(len(true.points)), max(1, len(true.points) // 2000)
        ):
            p = true.points[chunk]
            t = np.clip(
                ((p[:, None] - start[None]) * vector[None]).sum(-1) / squared[None],
                0,
                1,
            )
            d = np.linalg.norm(
                start[None] + t[..., None] * vector[None] - p[:, None], axis=-1
            )
            r = radius[None, :, 0] + t * (radius[None, :, 1] - radius[None, :, 0])
            inside[chunk] = (d <= r).any(axis=1)
    ends = np.stack([segments.start, segments.end], axis=1)
    d, i = true.tree.query(ends.reshape(-1, true.points.shape[1]))
    end_outside = (d > true.radii[i]).reshape(len(segments), 2)
    axes = [
        resample_polyline(np.stack([a, b]), SAMPLE_SPACING)[0]
        for a, b in zip(start, end, strict=True)
    ]
    axis_points = np.concatenate(axes) if axes else np.zeros((0, true.points.shape[1]))
    if len(axis_points):
        d, i = true.tree.query(axis_points)
        axis_deviation = d / true.radii[i]
    else:
        axis_deviation = np.zeros(0)
    return FitView(
        segments, eligible, alive, inside, end_outside, axis_points,
        axis_deviation, links, settings,
    )  # fmt: skip


def _positions(fit: FitView, indices: np.ndarray) -> np.ndarray:
    """Positions in ``fit.links.candidate`` of segment indices."""
    return np.flatnonzero(np.isin(fit.links.candidate, indices))


def junction_fit_class(true: TrueCentreline, fit: FitView, event: Event) -> str:
    """``reaches``, ``short`` or ``missing``, from the alive fitted segments.

    A segment is on the child when its midpoint's nearest true sample lies on
    the child's edge, within 1.5 radii (at least 1 voxel). The branch
    **reaches** when an endpoint of such a segment lies within the parent's
    radius of the parent's centreline, is **short** when none does, and is
    **missing** when no alive segment is on the child.
    """
    alive = np.flatnonzero(fit.alive)
    start, end = fit.segments.start[alive], fit.segments.end[alive]
    if not len(alive):
        return "missing"
    d, i = true.tree.query(0.5 * (start + end))
    on_child = (true.edge[i] == event.covariates["edge"]) & (
        d <= np.maximum(1.5 * true.radii[i], 1.0)
    )
    if not on_child.any():
        return "missing"
    part = true.on_edge(event.covariates["other"])
    tree = cKDTree(true.points[part])
    dist, k = tree.query(np.concatenate([start[on_child], end[on_child]]))
    return "reaches" if (dist <= true.radii[part][k]).any() else "short"


def spur_segments(fit: FitView, centre: np.ndarray, reach: float) -> np.ndarray:
    """Eligible segments with an endpoint outside the true tube near a point.

    These are the fit's spurs into a bump or protrusion: a segment that
    leaves the true tube there, with the leaving endpoint within ``reach`` of
    the event's centre (the reach of a spur end, ``size + height`` plus the
    match tolerance).
    """
    ends = np.stack([fit.segments.start, fit.segments.end], axis=1)
    near = np.linalg.norm(ends - centre, axis=-1) <= reach
    return np.flatnonzero(fit.eligible & (fit.end_outside & near).any(axis=1))


# -- outcomes --------------------------------------------------------------------------


def _anchor_reach(event: Event) -> int:
    """How many samples a window's anchors may snap inward: min(r, L / 4)."""
    covariates = event.covariates
    reach = min(covariates["r_local"], covariates["length"] / 4)
    return max(int(reach / SAMPLE_SPACING), 1)


def _anchors(event: Event, view: StageView) -> tuple[int, int]:
    """The event's anchors in one stage; a window's snap inward to matched samples.

    A window that ends at a junction or a leaf would otherwise fail whenever
    the predicted node sits a little off the true one.
    """
    a, b = event.anchors
    if event.event_type != "window":
        return a, b
    reach = _anchor_reach(event)
    head, tail = event.region[:reach], event.region[::-1][:reach]
    a = next((int(m) for m in head if view.matched_true[m]), a)
    b = next((int(m) for m in tail if view.matched_true[m]), b)
    return a, b


def _connected(view: StageView, event: Event) -> tuple[bool, bool]:
    """``connected``, and whether both anchors matched, in one stage."""
    a, b = _anchors(event, view)
    matched = bool(view.matched_true[a] and view.matched_true[b])
    if not matched:
        return False, False
    limit = 1.5 * event.true_distance + event.slack
    length = view.pred.path_length(
        int(view.nearest_pred[a]), int(view.nearest_pred[b]), limit
    )
    return bool(length <= limit), True


def _deviation(view: StageView, true: TrueCentreline, near: np.ndarray, r: float):
    """The largest distance of the given predicted vertices from the truth, over r."""
    return float(view.pred_distance[near].max() / r) if near.any() else np.nan


def stage_outcomes(
    event: Event, view: StageView, true: TrueCentreline
) -> dict[str, Any]:
    """An event's outcomes in one predicted graph."""
    covariates = event.covariates
    out: dict[str, Any] = {}
    if event.event_type in ("junction", "window"):
        out["connected"], out["anchors_matched"] = _connected(view, event)
    if event.event_type == "junction":
        out["class"] = (
            "connected"
            if out["connected"]
            else "detached"
            if out["anchors_matched"]
            else "missing"
        )
        node = covariates["node"]
        matched = view.junction_match.get(node)
        out["position_error"] = (
            float(
                np.linalg.norm(
                    view.pred.graph.node_positions[matched]
                    - true.graph.node_positions[node]
                )
                / covariates["r_node"]
            )
            if matched is not None
            else np.nan
        )
    elif event.event_type == "window":
        region = event.region
        out["covered"] = bool(view.matched_true[region].all())
        members = set(region.tolist())
        out["spur_ends"] = sum(
            e.type == "spur_end" and e.true_index in members for e in view.ends
        )
        near = np.isin(view.pred_nearest_true, region)
        out["deviation"] = _deviation(view, true, near, covariates["r_local"])
    elif event.event_type in ("bump", "protrusion"):
        centre, base, reach = _spur_place(event, true)
        spurs = [
            e
            for e in view.ends
            if e.type == "spur_end" and np.linalg.norm(e.position - centre) <= reach
        ]
        out["spur"] = bool(spurs)
        out["spur_leaf_length"] = max((e.leaf_length for e in spurs), default=0.0)
        r = float(true.radii[base])
        near = np.zeros(len(view.pred.points), dtype=bool)
        if view.pred.tree is not None:
            extent = covariates["size"] + covariates["height"]
            near[view.pred.tree.query_ball_point(true.points[base], extent + r)] = True
        out["deviation"] = _deviation(view, true, near, r)
    return out


def _spur_place(event: Event, true: TrueCentreline) -> tuple[np.ndarray, int, float]:
    """A bump's or protrusion's centre, base sample and spur reach."""
    covariates = event.covariates
    centre = (
        np.asarray(covariates["centre"])
        if event.event_type == "protrusion"
        else np.asarray(event.position)
    )
    r = float(true.radii[event.base])
    reach = covariates["size"] + covariates["height"] + float(tolerance(r))
    return centre, event.base, reach


def fit_outcomes(event: Event, fit: FitView, true: TrueCentreline) -> dict[str, Any]:
    """An event's fit-stage outcomes and decision quantities with thresholds."""
    out: dict[str, Any] = {}
    settings = fit.settings
    if event.event_type == "junction":
        out["fit_class"] = junction_fit_class(true, fit, event)
    if event.event_type == "window":
        out["fit_covered"] = bool(fit.inside[event.region].all())
    if event.event_type in ("junction", "window") and fit.links is not None:
        a, b = event.anchors
        if event.event_type == "window":
            reach = _anchor_reach(event)
            a_list, b_list = event.region[:reach], event.region[::-1][:reach]
        else:
            a_list, b_list = [a], [b]
        out.update(
            connection_quantities(
                fit.links,
                [(true.points[i], float(tolerance(true.radii[i]))) for i in a_list],
                [(true.points[i], float(tolerance(true.radii[i]))) for i in b_list],
            )
        )
        out |= {
            "tau_join": settings["join"],
            "tau_prune": settings["prune_threshold"],
            "tau_bridge": settings["bridge_reach"],
            "tau_bridge_angle": settings["bridge_angle"],
        }
    if event.event_type in ("bump", "protrusion"):
        centre, _, reach = _spur_place(event, true)
        spur = spur_segments(fit, centre, reach)
        out["fit_spur"] = bool(fit.alive[spur].any())
        if fit.links is not None:
            out.update(spur_quantities(fit.links, _positions(fit, spur)))
            out |= {
                "tau_prune": settings["prune_threshold"],
                "tau_protrusion": settings["min_protrusion"],
            }
    return out


# -- errors ----------------------------------------------------------------------------


def _runs(flags: np.ndarray, groups: np.ndarray) -> list[np.ndarray]:
    """Contiguous runs of True within each group (items ordered by group)."""
    index = np.flatnonzero(flags)
    if not len(index):
        return []
    breaks = np.flatnonzero((np.diff(index) != 1) | (np.diff(groups[index]) != 0)) + 1
    return np.split(index, breaks)


class Causes:
    """Assigns a cause to an error's position (plan, Section 7.3).

    In order: the nearest corruption record within :data:`K_RADII` local
    radii (for a break end, also along the true graph as far as the break's
    own length: ends retract from a hole); otherwise the nearest protrusion
    within that reach; otherwise ``near_junction`` or ``near_leaf`` (within
    :data:`K_RADII` times the larger of the local and the node's radius);
    otherwise ``unexplained``. ``mask_lost`` is assigned before, to missed
    runs (see :func:`error_rows`).
    """

    def __init__(
        self,
        sample: Sample,
        true: TrueCentreline,
        records: list[LocatedRecord],
        spacing: np.ndarray,
    ) -> None:
        self.true = true
        self.records = records
        self.protrusions = [
            (f"protrusion#{k}", cKDTree(index * spacing))
            for k, index in enumerate(protrusion_components(sample))
        ]

    def __call__(
        self, position: np.ndarray, start: int | None = None, along: float = 0.0
    ) -> str:
        """The cause at a position; ``start`` and ``along`` search a break's edge."""
        true = self.true
        _, i = true.tree.query(position)
        r = float(true.radii[i])
        best, best_d = None, np.inf
        for record in self.records:
            d = float(record.distance(position)[0])
            if d <= K_RADII * r and d < best_d:
                best, best_d = record.id, d
        if best is None and start is not None and along > 0 and self.records:
            reached = true.geodesic(start, along)
            samples = np.flatnonzero(np.isfinite(reached))
            for record in self.records:
                distance = record.distance(true.points[samples])
                hit = distance <= K_RADII * true.radii[samples]
                if hit.any() and reached[samples[hit]].min() < best_d:
                    best, best_d = record.id, float(reached[samples[hit]].min())
        if best is not None:
            return best
        for event_id, tree in self.protrusions:
            d = tree.query(position)[0]
            if d <= K_RADII * r and d < best_d:
                best, best_d = event_id, d
        if best is not None:
            return best
        graph = true.graph
        for name, nodes in (
            ("near_junction", graph.junction_nodes),
            ("near_leaf", graph.end_nodes),
        ):
            for node in nodes:
                reach = K_RADII * max(r, float(true.node_radii[node]))
                if np.linalg.norm(graph.node_positions[node] - position) <= reach:
                    return name
        return "unexplained"


def _mask_info(sample: Sample, spacing: np.ndarray) -> dict[str, Any]:
    """The mask's components (its largest is the main one) and distance transform."""
    labels, n = ndimage.label(sample.mask)
    main = int(np.argmax(np.bincount(labels.ravel())[1:]) + 1) if n else 0
    edt = ndimage.distance_transform_edt(sample.mask, sampling=spacing)
    return {"labels": labels, "main": main, "edt": edt}


def error_rows(
    stage: str,
    view: StageView,
    true: TrueCentreline,
    causes: Causes,
    sample: Sample,
    fit: FitView | None,
    window_of: np.ndarray,
    mask_info: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Every error of one stage, located, with its size and cause.

    Parameters
    ----------
    stage : str
        ``raw`` or ``clean``.
    view : StageView
        The stage's graph against the truth.
    true : TrueCentreline
        The sampled true graph.
    causes : Causes
        The sample's cause finder.
    sample : Sample
        The sample (its masks, for missed runs).
    fit : FitView, optional
        The fitted segments, which tell an off-centre run's ``source``.
    window_of : np.ndarray
        Per true sample, the ID of its window event.
    mask_info : dict, optional
        From :func:`_mask_info`, computed when needed if not given.

    Returns
    -------
    list of dict
        One row per error: ``error_id`` (``<stage>/<type>#<k>``), ``stage``,
        ``type``, ``position``, ``size``, ``cause``, ``window_id`` (the window
        of the nearest true sample), and per type: ``leaf_length`` and
        ``break_length`` for ends; ``edge``, ``r_mean``, ``in_mask``,
        ``in_clean_mask``, ``in_main_component`` and ``mask_radius_ratio``
        for missed runs; ``max_offset_over_r`` and ``source`` (``segment``,
        ``node``, or ``None`` without segments) for off-centre runs.
    """
    spacing = np.asarray(sample.spacing, dtype=float)
    rows: list[dict[str, Any]] = []
    for end in view.ends:
        rows.append(
            {
                "type": end.type,
                "position": end.position,
                "size": end.leaf_length,
                "leaf_length": end.leaf_length,
                "break_length": end.break_length,
                "cause": causes(
                    end.position,
                    end.true_index if end.type == "break_end" else None,
                    end.break_length,
                ),
            }
        )
    for run in _runs(~view.matched_true, true.edge):
        length = float(true.arc[run[-1]] - true.arc[run[0]])
        if length < MIN_RUN:
            continue
        if mask_info is None:
            mask_info = _mask_info(sample, spacing)
        voxels = np.clip(
            np.round(true.points[run] / spacing).astype(int),
            0,
            np.array(sample.mask.shape) - 1,
        )
        index = tuple(voxels.T)
        in_main = float((mask_info["labels"][index] == mask_info["main"]).mean())
        position = true.points[run[len(run) // 2]]
        rows.append(
            {
                "type": "missed",
                "position": position,
                "size": length,
                "edge": int(true.edge[run[0]]),
                "r_mean": float(true.radii[run].mean()),
                "in_mask": float(sample.mask[index].mean()),
                "in_clean_mask": float(sample.clean_mask[index].mean()),
                "in_main_component": in_main,
                "mask_radius_ratio": float(
                    np.median(mask_info["edt"][index] / true.radii[run])
                ),
                "cause": "mask_lost" if in_main < MASK_LOST else causes(position),
            }
        )
    pred = view.pred
    if len(pred.points):
        r_true = true.radii[view.pred_nearest_true]
        correct = view.pred_distance <= tolerance(r_true)
        off = correct & (view.pred_distance > OFF_CENTRE * r_true)
        order = np.argsort(pred.edge, kind="stable")
        for kind, flags in (("false_positive", ~correct), ("off_centre", off)):
            for run in _runs(flags[order], pred.edge[order]):
                members = order[run]
                if pred.edge[members[0]] < 0:
                    continue  # the graph's nodes: covered by their edges
                length = SAMPLE_SPACING * (len(members) - 1)
                if length < MIN_RUN:
                    continue
                position = pred.points[members[len(members) // 2]]
                row = {"type": kind, "position": position, "size": length}
                if kind == "off_centre":
                    row["max_offset_over_r"] = float(
                        (view.pred_distance[members] / r_true[members]).max()
                    )
                    row["source"] = _off_centre_source(fit, true, position, length)
                row["cause"] = causes(position)
                rows.append(row)
    for node in view.junction_missed:
        position = true.graph.node_positions[node]
        rows.append(
            {"type": "junction_missed", "position": position, "size": 0.0,
             "cause": causes(position)}
        )  # fmt: skip
    for node in view.junction_extra:
        position = pred.graph.node_positions[node]
        rows.append(
            {"type": "junction_extra", "position": position, "size": 0.0,
             "cause": causes(position)}
        )  # fmt: skip
    counter: dict[str, int] = {}
    out = []
    for row in rows:
        k = counter.get(row["type"], 0)
        counter[row["type"]] = k + 1
        _, nearest = true.tree.query(row["position"])
        out.append(
            {
                "error_id": f"{stage}/{row['type']}#{k}",
                "stage": stage,
                **row,
                "position": [float(x) for x in row["position"]],
                "window_id": window_of[nearest],
            }
        )
    return out


def _off_centre_source(
    fit: FitView | None, true: TrueCentreline, position: np.ndarray, size: float
) -> str | None:
    """``segment`` if alive segments deviate near an off-centre run, else ``node``.

    Alive segments deviate when a sample of their axes within half the run's
    length plus the local radius of its middle lies more than
    :data:`OFF_CENTRE` radii from the true centreline. Otherwise the run is
    made by the graph's geometry: joint nodes (the mean of their endpoints)
    and corners cut at junctions (P7).
    """
    if fit is None:
        return None
    _, i = true.tree.query(position)
    reach = size / 2 + float(true.radii[i])
    near = np.linalg.norm(fit.axis_points - position, axis=1) <= reach
    return "segment" if (fit.axis_deviation[near] > OFF_CENTRE).any() else "node"


# -- one sample ------------------------------------------------------------------------


@dataclass
class Scores:
    """The score tables' rows of one sample (plan, Section 7).

    Attributes
    ----------
    samples : list of dict
        One row per stage: the graph metrics, the root-connected fraction,
        summaries of the events, and error counts by type and cause.
    events : list of dict
        One row per (event, stage): the event's ID and covariates, its
        outcomes in that stage's graph, and its fit-stage outcomes and
        decision quantities (the same in both stages).
    errors : list of dict
        One row per error of either stage (see :func:`error_rows`).
    """

    samples: list[dict[str, Any]]
    events: list[dict[str, Any]]
    errors: list[dict[str, Any]]


def score_sample(
    sample: Sample,
    raw: SkeletonGraph,
    clean: SkeletonGraph,
    segments: Any = None,
    config: Any = None,
) -> Scores:
    """Score a run's two graphs of one sample (plan, Sections 7.1 to 7.3).

    Parameters
    ----------
    sample : Sample
        The sample, with its true graph, records and masks.
    raw, clean : SkeletonGraph
        The graph stage's output.
    segments : SegmentArrays or JaxLineSegments, optional
        The fitted segments (``None`` for variants that fit nothing). With
        them come the fit-stage outcomes and, for ``assemble``, the decision
        quantities.
    config : GraphConfig, optional
        The graph stage's settings; needed with ``segments``.

    Returns
    -------
    Scores
    """
    true = TrueCentreline(sample.graph)
    events = extract_events(sample, true)
    spacing = np.asarray(sample.spacing, dtype=float)
    fit = None
    if segments is not None:
        if config is None:
            raise ValueError("scoring fitted segments needs the graph config")
        fit = fit_view(true, segments, config)
    window_of = np.full(len(true.points), None, dtype=object)
    for event in events:
        if event.event_type == "window":
            window_of[event.region] = event.event_id
    causes = Causes(sample, true, located_records(sample), spacing)
    mask_info = None
    fitted = [fit_outcomes(e, fit, true) if fit else {} for e in events]
    sample_rows, event_rows_, error_rows_ = [], [], []
    for stage, graph in zip(STAGES, (raw, clean), strict=True):
        view = stage_view(graph, true)
        stage_events = [
            {**event.row(), "stage": stage, **stage_outcomes(event, view, true), **f}
            for event, f in zip(events, fitted, strict=True)
        ]
        if mask_info is None and (~view.matched_true).any():
            mask_info = _mask_info(sample, spacing)
        errors = error_rows(
            stage, view, true, causes, sample, fit, window_of, mask_info
        )
        sample_rows.append(
            _sample_row(stage, graph, sample, view, true, stage_events, errors, fit)
        )
        event_rows_ += stage_events
        error_rows_ += errors
    return Scores(sample_rows, event_rows_, error_rows_)


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([bool(r[key]) for r in rows])) if rows else np.nan


def _sample_row(
    stage: str,
    graph: SkeletonGraph,
    sample: Sample,
    view: StageView,
    true: TrueCentreline,
    events: list[dict[str, Any]],
    errors: list[dict[str, Any]],
    fit: FitView | None,
) -> dict[str, Any]:
    """One stage's row of ``samples.parquet``."""
    by_type: dict[str, list[dict[str, Any]]] = {}
    for row in events:
        by_type.setdefault(row["event_type"], []).append(row)
    junctions = by_type.get("junction", [])
    windows = [w for w in by_type.get("window", []) if w["continuous"]]
    bumps = by_type.get("bump", [])
    row = {
        "stage": stage,
        **compare_graphs(graph, sample.graph).to_dict(),
        "root_connected_fraction": root_fraction(view, true),
        "junction_connected_rate": _rate(junctions, "connected"),
        "thin_junction_connected_rate": _rate(
            [j for j in junctions if j["thin"]], "connected"
        ),
        "continuous_window_break_rate": 1 - _rate(windows, "connected"),
        "bump_spur_rate": _rate(bumps, "spur"),
        "bump_deviation_mean": float(np.nanmean([b["deviation"] for b in bumps]))
        if bumps and not np.isnan([b["deviation"] for b in bumps]).all()
        else np.nan,
        "protrusion_spur_rate": _rate(by_type.get("protrusion", []), "spur"),
    }
    for kind in ERROR_TYPES:
        row[f"n_{kind}"] = sum(e["type"] == kind for e in errors)
    causes = sorted({e["cause"].split("#")[0] for e in errors})
    for cause in causes:
        row[f"cause_{cause}"] = sum(e["cause"].split("#")[0] == cause for e in errors)
    row["n_algorithmic_breaks"] = sum(
        e["type"] == "break_end" and e["cause"] in ALGORITHMIC_BREAK for e in errors
    )
    missed = [e for e in errors if e["type"] == "missed"]
    row["missed_length"] = float(sum(e["size"] for e in missed))
    row["missed_length_mask_lost"] = float(
        sum(e["size"] for e in missed if e["cause"] == "mask_lost")
    )
    if fit is not None:
        row["n_alive"] = int(fit.alive.sum())
    return row
