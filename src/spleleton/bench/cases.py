"""``proto-cases``: hand-built volumes with known answers (plan, Section 14.1).

64^3 volumes, spacing 1, each with its true graph, its corruption records in
the generator's format, and hand-made predictions: the true segments, and the
true segments with one planted error. Unit tests and quick checks define
their expected answers on these. Ported from stage 0's
``scripts/proto/_cases.py``; the samples are the same.

- ``tube``: a straight tube, r = 4. Predictions ``true``; ``split``, a break
  of 2 r mid-tube; ``shift``, the whole tube moved 0.4 r sideways.
- ``tube-gap``: the tube with one gap of width 4. ``true``; ``split``, the
  chain broken at the gap, its ends 0.5 r outside it.
- ``y-thin``: a Y junction, parent r = 6, thin child r = 1.8 (ratio 0.3).
  ``true``; ``retract``, the child's first end moved to ``r_parent + 2
  r_child`` from the node, outside the parent's wall.
- ``y-thin-gap``: the Y with a cut across the thin child just outside the
  parent's wall (width 3). ``true``; ``retract``, as above: the end lies
  beyond the cut.
- ``tube-bump``: the tube with one bump, size 1 r, height 1 r. ``true``;
  ``spur``, an extra segment from the centreline to the bump's tip.
- ``contact``: two parallel tubes, r = 3.5, 5 apart, bridged once. ``true``;
  ``join``, an extra segment along the bridge.

The two gap cases are kept for the probe's exactness tests; no benchmark
outcome depends on them (gaps are out of scope).

Two choices differ from the plan's wording, deliberately:

- ``shift`` is 0.4 r, not 0.5 r: ``compare_graphs`` matches centreline points
  within 0.5 r, so a 0.5 r shift sits exactly on the match tolerance.
- ``retract`` moves the child's end to ``r_parent + 2 r_child`` from the
  junction node, not by ``2 r_child``: with a ratio of 0.3, an end 2 r_child
  from the node is still inside the parent's tube (3.6 < 6), which is a
  connected junction, not a planted error.
"""

import dataclasses

import numpy as np

from spleleton.graph import SkeletonGraph, resample_polyline
from spleleton.primitives import SegmentArrays
from spleleton.synthetic._geometry import (
    bounding_box,
    closest_on_segment,
    distance_to_polyline,
    physical_grid,
)
from spleleton.synthetic.corruptions import (
    CorruptionConfig,
    _arc_length,
    _paint_ball,
    _paint_capsule,
)
from spleleton.synthetic.dataset import Sample, SampleConfig
from spleleton.synthetic.rasterize import rasterize_graph

SHAPE = (64, 64, 64)
SPACING = np.ones(3)
#: Opacity of hand-made predictions (as ``SegmentArrays.from_graph``).
OPACITY = 0.99


@dataclasses.dataclass
class Case:
    """One hand-built case: a sample, its predictions and their errors."""

    name: str
    sample: Sample
    predictions: dict[str, SegmentArrays]
    planted: dict[str, dict]
    description: str


def build_cases() -> list[Case]:
    """Every case, in a fixed order."""
    return [_tube(), _tube_gap(), _y(gap=False), _y(gap=True), _tube_bump(), _contact()]


# -- geometry ---------------------------------------------------------------------


def _graph(nodes, edges, radii) -> SkeletonGraph:
    nodes = np.asarray(nodes, dtype=float)
    return SkeletonGraph(
        node_positions=nodes,
        edges=np.asarray(edges, dtype=np.int64),
        edge_paths=[nodes[[a, b]] for a, b in edges],
        edge_radii=[np.asarray(r, dtype=float) for r in radii],
    )


def _sample(graph: SkeletonGraph, corrupt=None, records=()) -> Sample:
    """Rasterise a graph; ``corrupt(mask)`` may fill ``records`` as it runs."""
    clean = rasterize_graph(graph, SHAPE)
    mask = clean.copy() if corrupt is None else corrupt(clean.copy())
    return Sample(
        graph=graph,
        clean_mask=clean,
        mask=mask,
        corruptions=list(records),
        config=SampleConfig(shape=SHAPE, corruptions=CorruptionConfig()),
        seed=0,
    )


def _chain(start, end, r_start, r_end, lo=0.0, hi=1.0) -> SegmentArrays:
    """Segments of about 2 r along ``start -> end``, between fractions lo, hi."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = np.linalg.norm(end - start) * (hi - lo)
    piece = max(2.0, r_start + r_end)
    n = max(1, round(length / piece))
    t = np.linspace(lo, hi, n + 1)
    points = start + t[:, None] * (end - start)
    radii = r_start + t * (r_end - r_start)
    return SegmentArrays(
        points[:-1],
        points[1:],
        radius=np.stack([radii[:-1], radii[1:]], axis=1),
        opacity=OPACITY,
    )


def _from_edges(graph: SkeletonGraph, cuts=None) -> SegmentArrays:
    """Chains along every edge; ``cuts[edge] = [(lo, hi), ...]`` keeps parts."""
    cuts = cuts or {}
    parts = []
    for edge, (path, radii) in enumerate(
        zip(graph.edge_paths, graph.edge_radii, strict=True)
    ):
        for lo, hi in cuts.get(edge, [(0.0, 1.0)]):
            parts.append(_chain(path[0], path[1], radii[0], radii[1], lo, hi))
    return _concatenate(parts)


def _concatenate(parts: list[SegmentArrays]) -> SegmentArrays:
    return SegmentArrays(
        np.concatenate([p.start for p in parts]),
        np.concatenate([p.end for p in parts]),
        radius=np.concatenate([p.radius for p in parts]),
        opacity=np.concatenate([p.opacity for p in parts]),
    )


def _cut_slab(mask, position, tangent, width, radius) -> None:
    """Remove the voxels within ``width / 2`` along and ``2 r`` across."""
    lower = np.maximum(np.floor(position - 2 * radius - width).astype(int), 0)
    upper = np.minimum(np.ceil(position + 2 * radius + width).astype(int) + 1, SHAPE)
    box, points = physical_grid(lower, upper, SPACING)
    along = np.abs((points - position) @ tangent)
    _, across = closest_on_segment(
        points, position - 2 * width * tangent, position + 2 * width * tangent
    )
    mask[box] &= ~((along <= width / 2) & (across <= 2 * radius))


def _junction_gap(
    mask: np.ndarray, graph: SkeletonGraph, parent: int, edge: int, width: float
) -> dict:
    """Cut a thin child just outside its parent's wall (stage 0's junction gap).

    The cut is a slab ``width`` wide along the child, centred ``r_parent +
    r_child`` from the junction, and ``2 r_child`` across it. Voxels inside
    any other edge's true tube are never cut, so the parent's wall stays.
    """
    node = int(graph.edges[edge][0])
    r_parent = float(graph.edge_radii[parent][-1])
    r_child = float(graph.edge_radii[edge][0])
    distance = r_parent + r_child
    path, radii = graph.edge_paths[edge], graph.edge_radii[edge]
    points, point_radii = resample_polyline(path, 0.05, radii)
    arc = _arc_length(points)
    at = int(np.searchsorted(arc, distance))
    position, radius = points[at], float(point_radii[at])
    tangent = points[min(at + 1, len(points) - 1)] - points[max(at - 1, 0)]
    tangent /= np.linalg.norm(tangent)

    lower, upper = bounding_box(position[None], 2 * radius + width, SPACING, SHAPE)
    box, grid = physical_grid(lower, upper, SPACING)
    flat = grid.reshape(-1, mask.ndim)
    along = np.abs((flat - position) @ tangent)
    distances, inside_other = [], np.zeros(len(flat), dtype=bool)
    for other, (other_path, other_radii) in enumerate(
        zip(graph.edge_paths, graph.edge_radii, strict=True)
    ):
        d, r, _ = distance_to_polyline(flat, other_path, other_radii)
        distances.append(d)
        if other != edge:
            inside_other |= d <= r
    distances = np.stack(distances)
    nearest = np.argmin(distances, axis=0) == edge
    cut = (along <= width / 2) & (distances[edge] <= 2 * radius) & nearest
    cut &= ~inside_other
    before = int(mask[box].sum())
    mask[box] &= ~cut.reshape(grid.shape[:-1])
    # how far outside the other tubes the cut's centre is, in child radii
    clearance = np.inf
    for other, (other_path, other_radii) in enumerate(
        zip(graph.edge_paths, graph.edge_radii, strict=True)
    ):
        if other != edge:
            d, r, _ = distance_to_polyline(position[None], other_path, other_radii)
            clearance = min(clearance, float(d[0] - r[0]))
    return {
        "type": "junction_gap",
        "node": node,
        "parent_edge": parent,
        "edge": edge,
        "r_parent": r_parent,
        "r_child": r_child,
        "ratio": r_child / r_parent,
        "mode": "cut",
        "width": width,
        "offset": 1.0,
        "distance": distance,
        "position": position.tolist(),
        "tangent": tangent.tolist(),
        "removed_voxels": before - int(mask[box].sum()),
        "clearance": clearance / r_child,
    }


# -- the cases ----------------------------------------------------------------------

_TUBE = ([6.0, 32.0, 32.0], [58.0, 32.0, 32.0], 4.0)


def _tube_graph() -> SkeletonGraph:
    start, end, r = _TUBE
    return _graph([start, end], [(0, 1)], [[r, r]])


def _tube() -> Case:
    graph = _tube_graph()
    r = _TUBE[2]
    length = _TUBE[1][0] - _TUBE[0][0]
    # a break of 2 r centred mid-tube
    half = r / length
    split = _from_edges(graph, {0: [(0.0, 0.5 - half), (0.5 + half, 1.0)]})
    true = _from_edges(graph)
    shifted = dataclasses.replace(
        true,
        start=true.start + np.array([0.0, 0.4 * r, 0.0]),
        end=true.end + np.array([0.0, 0.4 * r, 0.0]),
    )
    return Case(
        "tube",
        _sample(graph),
        {"true": true, "split": split, "shift": shifted},
        {
            "true": {},
            "split": {"type": "break", "position": [32.0, 32.0, 32.0], "width": 2 * r},
            "shift": {"type": "off_centre", "offset": 0.4 * r, "direction": [0, 1, 0]},
        },
        "straight tube, r = 4, no corruption",
    )


def _tube_gap() -> Case:
    graph = _tube_graph()
    start, end, r = _TUBE
    width, centre = 4.0, np.array([32.0, 32.0, 32.0])
    tangent = np.array([1.0, 0.0, 0.0])
    record = {
        "type": "gap",
        "edge": 0,
        "position": centre.tolist(),
        "tangent": tangent.tolist(),
        "width": width,
    }

    def corrupt(mask):
        _cut_slab(mask, centre, tangent, width, r)
        return mask

    length = end[0] - start[0]
    half = (width / 2 + 0.5 * r) / length
    split = _from_edges(graph, {0: [(0.0, 0.5 - half), (0.5 + half, 1.0)]})
    return Case(
        "tube-gap",
        _sample(graph, corrupt, [record]),
        {"true": _from_edges(graph), "split": split},
        {
            "true": {},
            "split": {"type": "break", "position": centre.tolist(), "width": width + r},
        },
        "straight tube, r = 4, one gap of width 4 mid-tube",
    )


#: The Y: parent along +x into the junction, major child at 20 degrees, thin
#: child at 60 degrees on the other side.
_Y_NODE = np.array([28.0, 32.0, 32.0])
_Y_PARENT_R, _Y_MAJOR_R, _Y_CHILD_R = 6.0, 5.95, 1.8


def _y_graph() -> SkeletonGraph:
    major = np.array([np.cos(np.deg2rad(20)), np.sin(np.deg2rad(20)), 0.0])
    minor = np.array([np.cos(np.deg2rad(60)), -np.sin(np.deg2rad(60)), 0.0])
    nodes = [[4.0, 32.0, 32.0], _Y_NODE, _Y_NODE + 28 * major, _Y_NODE + 22 * minor]
    return _graph(
        nodes,
        [(0, 1), (1, 2), (1, 3)],
        [[_Y_PARENT_R] * 2, [_Y_MAJOR_R] * 2, [_Y_CHILD_R] * 2],
    )


def _y(gap: bool) -> Case:
    graph = _y_graph()
    records: list[dict] = []
    corrupt = None
    width = 3.0
    if gap:

        def corrupt(mask):
            records.append(_junction_gap(mask, graph, parent=0, edge=2, width=width))
            return mask

    sample = _sample(graph, corrupt, records)
    child_length = np.linalg.norm(graph.node_positions[3] - _Y_NODE)
    retract_to = _Y_PARENT_R + 2 * _Y_CHILD_R
    retract = _from_edges(graph, {2: [(retract_to / child_length, 1.0)]})
    name = "y-thin-gap" if gap else "y-thin"
    return Case(
        name,
        sample,
        {"true": _from_edges(graph), "retract": retract},
        {
            "true": {},
            "retract": {
                "type": "junction_detach",
                "junction": _Y_NODE.tolist(),
                "edge": 2,
                "end_distance": retract_to,
            },
        },
        "Y junction, r_parent = 6, thin child r = 1.8 (ratio 0.3)"
        + (f", junction gap of width {width}" if gap else ""),
    )


def _tube_bump() -> Case:
    graph = _tube_graph()
    r = _TUBE[2]
    base = np.array([32.0, 32.0, 32.0])
    direction = np.array([0.0, 1.0, 0.0])
    size = height = 1.0 * r
    centre = base + (r + height - size) * direction
    record = {
        "type": "bump",
        "edge": 0,
        "position": centre.tolist(),
        "radius": size,
        "height": height,
    }

    def corrupt(mask):
        _paint_ball(mask, centre, size, SPACING, value=True)
        return mask

    tip = base + (r + height) * direction
    spur = SegmentArrays(base[None], tip[None], radius=0.5 * size, opacity=OPACITY)
    true = _from_edges(graph)
    return Case(
        "tube-bump",
        _sample(graph, corrupt, [record]),
        {"true": true, "spur": _concatenate([true, spur])},
        {
            "true": {},
            "spur": {"type": "spur", "start": base.tolist(), "end": tip.tolist()},
        },
        "straight tube, r = 4, one bump of size 1 r and height 1 r",
    )


def _contact() -> Case:
    r, bridge = 3.5, 1.5
    a = ([6.0, 26.0, 32.0], [58.0, 26.0, 32.0])
    b = ([6.0, 38.0, 32.0], [58.0, 38.0, 32.0])
    graph = _graph([a[0], a[1], b[0], b[1]], [(0, 1), (2, 3)], [[r, r], [r, r]])
    point_a, point_b = np.array([32.0, 26.0, 32.0]), np.array([32.0, 38.0, 32.0])
    record = {
        "type": "contact",
        "edges": [0, 1],
        "start": point_a.tolist(),
        "end": point_b.tolist(),
        "gap": float(np.linalg.norm(point_b - point_a) - 2 * r),
        "radius": bridge,
    }

    def corrupt(mask):
        _paint_capsule(mask, point_a, point_b, bridge, SPACING)
        return mask

    true = _from_edges(graph)
    join = SegmentArrays(point_a[None], point_b[None], radius=bridge, opacity=OPACITY)
    return Case(
        "contact",
        _sample(graph, corrupt, [record]),
        {"true": true, "join": _concatenate([true, join])},
        {
            "true": {},
            "join": {
                "type": "false_join",
                "start": point_a.tolist(),
                "end": point_b.tolist(),
            },
        },
        "two parallel tubes, r = 3.5, surfaces 5 apart, one bridge of r = 1.5",
    )


def case_names() -> tuple[str, ...]:
    """The names of the cases, in order."""
    return ("tube", "tube-gap", "y-thin", "y-thin-gap", "tube-bump", "contact")


def build_case(name: str) -> Case:
    """One case by name."""
    builders = {
        "tube": _tube,
        "tube-gap": _tube_gap,
        "y-thin": lambda: _y(gap=False),
        "y-thin-gap": lambda: _y(gap=True),
        "tube-bump": _tube_bump,
        "contact": _contact,
    }
    return builders[name]()


def prediction_payload(case: Case) -> dict[str, np.ndarray]:
    """The predictions as flat arrays, ``<name>__<field>``."""
    payload = {}
    for name, arrays in case.predictions.items():
        for key in ("start", "end", "radius", "opacity"):
            payload[f"{name}__{key}"] = getattr(arrays, key)
    return payload
