"""Corruptions that turn a clean tube mask into a realistic segmentation.

Each corruption takes the mask, the ground-truth graph and a random
generator, and returns the corrupted mask plus a list of records describing
what it changed and where. The ground-truth graph itself is never modified:
it always describes the true, uncorrupted tree.

All lengths are physical (voxel units when the spacing is 1).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from spleleton.graph import SkeletonGraph, resample_polyline
from spleleton.synthetic._geometry import (
    bounding_box,
    closest_on_segment,
    distance_to_polyline,
    physical_grid,
)


@dataclass(frozen=True)
class SurfaceNoiseConfig:
    """Smooth random displacement of the tube surface.

    Parameters
    ----------
    amplitude : float
        Standard deviation of the surface displacement.
    correlation_length : float
        Standard deviation of the Gaussian that smooths the noise field.
    """

    amplitude: float = 0.75
    correlation_length: float = 2.0


@dataclass(frozen=True)
class BumpConfig:
    """Balls attached to the tube surface, the classic cause of spurs.

    Parameters
    ----------
    count : int
        Number of bumps.
    height : tuple of float
        Range of the protrusion beyond the surface, relative to the local
        radius. It is capped at 1.5 times the bump radius, so every bump
        overlaps the tube.
    size : tuple of float
        Range of the bump radius, relative to the local radius.
    """

    count: int = 4
    height: tuple[float, float] = (0.75, 1.5)
    size: tuple[float, float] = (0.5, 1.0)


@dataclass(frozen=True)
class ContactConfig:
    """Bridges between branches that come close, creating false joins.

    Parameters
    ----------
    count : int
        Maximum number of bridges. Fewer are made if fewer pairs of branches
        come within ``max_gap`` of each other.
    max_gap : float
        Largest surface-to-surface gap that is bridged.
    min_gap : float
        Smallest length of a bridge outside every tube. Of each pair of
        branches, the candidate is the point pair with the smallest gap in
        ``[min_gap, max_gap]`` whose straight bridge runs at least ``min_gap``
        outside all tubes, so every bridge adds voxels to the mask. (The
        gap alone is not enough: next to where two curved tubes almost
        touch, a pair with a large gap is joined by a slanted bridge that
        runs almost entirely inside the tubes.) With the default of 0, the
        pair with the smallest positive gap is the candidate, and bridges
        join tubes that almost touch already.
    bridge_radius : float
        Radius of the bridge.
    min_path_ratio : float
        Only bridge points whose distance along the tree is at least this
        many times their straight-line distance. This skips points that are
        already joined through a nearby junction, where a bridge would add
        nothing.
    """

    count: int = 1
    max_gap: float = 6.0
    min_gap: float = 0.0
    bridge_radius: float = 1.5
    min_path_ratio: float = 3.0


@dataclass(frozen=True)
class HoleConfig:
    """Small cavities inside the tubes.

    Parameters
    ----------
    count : int
        Number of holes.
    relative_radius : tuple of float
        Range of the hole radius, relative to the local tube radius. Holes
        are placed so that they stay at least one voxel inside the surface;
        tubes too thin for that are skipped.
    """

    count: int = 3
    relative_radius: tuple[float, float] = (0.2, 0.4)


@dataclass(frozen=True)
class GapConfig:
    """Slabs cut across branches, the classic cause of breaks.

    Parameters
    ----------
    count : int
        Number of gaps.
    width : tuple of float
        Range of the gap width along the branch.
    min_radius : float
        Only cut branches whose radius at the cut is at least this.
    junction_distance : float
        Minimum distance of the cut from either end node, in units of the
        local radius, so gaps do not remove junctions.
    """

    count: int = 2
    width: tuple[float, float] = (2.0, 4.0)
    min_radius: float = 1.5
    junction_distance: float = 2.0


@dataclass(frozen=True)
class CorruptionConfig:
    """Which corruptions to apply. ``None`` disables a corruption.

    They are applied in the order: surface noise, bumps, contacts, holes,
    gaps. Gaps come last so they cut through everything else.
    """

    surface_noise: SurfaceNoiseConfig | None = None
    bumps: BumpConfig | None = None
    contacts: ContactConfig | None = None
    holes: HoleConfig | None = None
    gaps: GapConfig | None = None


@dataclass
class _Context:
    graph: SkeletonGraph
    spacing: np.ndarray
    rng: np.random.Generator
    records: list[dict] = field(default_factory=list)


def corrupt(
    mask: np.ndarray,
    graph: SkeletonGraph,
    config: CorruptionConfig,
    rng: np.random.Generator,
    spacing: tuple[float, ...] | None = None,
) -> tuple[np.ndarray, list[dict]]:
    """Apply the configured corruptions to a mask.

    Parameters
    ----------
    mask : np.ndarray
        Clean binary mask.
    graph : SkeletonGraph
        Ground-truth graph with radii, in physical coordinates.
    config : CorruptionConfig
        Which corruptions to apply, and how strongly.
    rng : np.random.Generator
        Source of randomness.
    spacing : tuple of float, optional
        Physical voxel size per axis. Defaults to 1 on every axis.

    Returns
    -------
    mask : np.ndarray
        The corrupted mask (a new array).
    records : list of dict
        One record per change, with at least a ``"type"`` key. Positions are
        physical coordinates.
    """
    spacing_array = (
        np.ones(mask.ndim) if spacing is None else np.asarray(spacing, float)
    )
    context = _Context(graph=graph, spacing=spacing_array, rng=rng)
    mask = mask.copy()
    if config.surface_noise is not None:
        mask = _surface_noise(mask, config.surface_noise, context)
    if config.bumps is not None:
        _bumps(mask, config.bumps, context)
    if config.contacts is not None:
        _contacts(mask, config.contacts, context)
    if config.holes is not None:
        _holes(mask, config.holes, context)
    if config.gaps is not None:
        _gaps(mask, config.gaps, context)
    return mask, context.records


def _surface_noise(
    mask: np.ndarray, config: SurfaceNoiseConfig, context: _Context
) -> np.ndarray:
    # signed distance to the surface, positive inside; the surface lies
    # halfway between the last inside and the first outside voxel
    inside = ndimage.distance_transform_edt(mask, sampling=context.spacing)
    outside = ndimage.distance_transform_edt(~mask, sampling=context.spacing)
    signed_distance = np.where(mask, inside - 0.5, 0.5 - outside)

    noise = context.rng.standard_normal(mask.shape)
    noise = ndimage.gaussian_filter(
        noise, sigma=config.correlation_length / context.spacing
    )
    noise /= max(float(noise.std()), 1e-12)
    context.records.append(
        {
            "type": "surface_noise",
            "amplitude": config.amplitude,
            "correlation_length": config.correlation_length,
        }
    )
    return signed_distance + config.amplitude * noise > 0


def _bumps(mask: np.ndarray, config: BumpConfig, context: _Context) -> None:
    for _ in range(config.count):
        edge, position, tangent, radius = _sample_on_graph(context, margin=1.0)
        if edge is None:
            return
        direction = _random_perpendicular(tangent, context.rng)
        size = context.rng.uniform(*config.size) * radius
        height = min(context.rng.uniform(*config.height) * radius, 1.5 * size)
        # the bump's outermost point lies `height` beyond the surface
        center = position + (radius + height - size) * direction
        _paint_ball(mask, center, size, context.spacing, value=True)
        context.records.append(
            {
                "type": "bump",
                "edge": edge,
                "position": center.tolist(),
                "radius": size,
                "height": height,
            }
        )


def _contacts(mask: np.ndarray, config: ContactConfig, context: _Context) -> None:
    graph = context.graph
    samples = [
        resample_polyline(path, 0.5, radii)
        for path, radii in zip(graph.edge_paths, graph.edge_radii, strict=True)
    ]
    arcs = [_arc_length(points) for points, _ in samples]
    node_distance = _node_distances(graph)
    candidates = []
    for a in range(graph.n_edges):
        for b in range(a + 1, graph.n_edges):
            (points_a, radii_a), (points_b, radii_b) = samples[a], samples[b]
            distance = np.linalg.norm(points_a[:, None] - points_b[None], axis=-1)
            gap = distance - radii_a[:, None] - radii_b[None, :]
            # distance along the tree, leaving each edge through either end
            along_a = np.stack([arcs[a], arcs[a][-1] - arcs[a]])
            along_b = np.stack([arcs[b], arcs[b][-1] - arcs[b]])
            tree_distance = np.min(
                [
                    along_a[i][:, None]
                    + node_distance[graph.edges[a][i], graph.edges[b][j]]
                    + along_b[j][None, :]
                    for i in range(2)
                    for j in range(2)
                ],
                axis=0,
            )
            # the tubes must be apart, and far apart along the tree
            valid = (
                (gap > 0)
                & (gap >= config.min_gap)
                & (gap <= config.max_gap)
                & (tree_distance >= config.min_path_ratio * distance)
            )
            if not valid.any():
                continue
            if config.min_gap > 0:
                found = _first_long_bridge(
                    points_a, points_b, gap, valid, graph, config.min_gap
                )
                if found is None:
                    continue
                (i, j), outside = found
            else:
                i, j = np.unravel_index(
                    np.argmin(np.where(valid, gap, np.inf)), gap.shape
                )
                outside = None
            candidates.append((gap[i, j], a, b, points_a[i], points_b[j], outside))

    order = context.rng.permutation(len(candidates))[: config.count]
    for index in order:
        gap, a, b, point_a, point_b, outside = candidates[index]
        _paint_capsule(mask, point_a, point_b, config.bridge_radius, context.spacing)
        record = {
            "type": "contact",
            "edges": [int(a), int(b)],
            "start": point_a.tolist(),
            "end": point_b.tolist(),
            "gap": float(gap),
            "radius": config.bridge_radius,
        }
        if outside is not None:
            record["outside"] = outside
        context.records.append(record)


def _first_long_bridge(
    points_a: np.ndarray,
    points_b: np.ndarray,
    gap: np.ndarray,
    valid: np.ndarray,
    graph: SkeletonGraph,
    min_length: float,
    step: float = 0.25,
    chunk: int = 64,
) -> tuple[tuple[int, int], float] | None:
    """The valid pair with the smallest gap whose bridge is long enough outside.

    Pairs are tried in order of their gap. A bridge's length outside the
    tubes is measured on points ``step`` apart along it, against every
    edge's true tube.

    Returns
    -------
    tuple or None
        The pair's indices and its bridge's length outside every tube, or
        ``None`` if no valid pair has a long enough bridge.
    """
    flat = np.flatnonzero(valid)
    flat = flat[np.argsort(gap.ravel()[flat], kind="stable")]
    for first in range(0, len(flat), chunk):
        i, j = np.unravel_index(flat[first : first + chunk], gap.shape)
        starts, ends = points_a[i], points_b[j]
        lengths = np.linalg.norm(ends - starts, axis=1)
        n_points = int(np.ceil(lengths.max() / step)) + 1
        fractions = np.linspace(0.0, 1.0, n_points)
        points = starts[:, None] + fractions[None, :, None] * (ends - starts)[:, None]
        flat_points = points.reshape(-1, points.shape[-1])
        inside = np.zeros(len(flat_points), dtype=bool)
        for path, radii in zip(graph.edge_paths, graph.edge_radii, strict=True):
            distance, radius, _ = distance_to_polyline(flat_points, path, radii)
            inside |= distance <= radius
        outside = (~inside.reshape(len(i), n_points)).sum(axis=1)
        outside = outside * lengths / (n_points - 1)
        long_enough = np.flatnonzero(outside >= min_length)
        if len(long_enough):
            k = long_enough[0]
            return (int(i[k]), int(j[k])), float(outside[k])
    return None


def _holes(mask: np.ndarray, config: HoleConfig, context: _Context) -> None:
    voxel = float(context.spacing.max())
    for _ in range(config.count):
        edge, position, tangent, radius = _sample_on_graph(context, margin=1.0)
        if edge is None:
            return
        hole_radius = context.rng.uniform(*config.relative_radius) * radius
        free = radius - hole_radius - voxel
        if hole_radius < 0.5 * voxel or free < 0:
            continue
        offset = context.rng.uniform(0, free) * _random_perpendicular(
            tangent, context.rng
        )
        center = position + offset
        _paint_ball(mask, center, hole_radius, context.spacing, value=False)
        context.records.append(
            {
                "type": "hole",
                "edge": edge,
                "position": center.tolist(),
                "radius": hole_radius,
            }
        )


def _gaps(mask: np.ndarray, config: GapConfig, context: _Context) -> None:
    graph = context.graph
    for _ in range(config.count):
        edge, position, tangent, radius = _sample_on_graph(
            context,
            margin=config.junction_distance,
            min_radius=config.min_radius,
        )
        if edge is None:
            return
        width = context.rng.uniform(*config.width)
        # remove everything near this edge within the slab, including bumps
        reach = 2 * radius + width
        lower, upper = bounding_box(position[None], reach, context.spacing, mask.shape)
        if np.any(upper <= lower):
            continue
        box, points = physical_grid(lower, upper, context.spacing)
        flat = points.reshape(-1, mask.ndim)
        along = np.abs((flat - position) @ tangent)
        # only cut voxels that belong to this edge, not to a nearby branch
        distances = np.stack(
            [distance_to_polyline(flat, path)[0] for path in graph.edge_paths]
        )
        nearest = np.argmin(distances, axis=0) == edge
        cut = (along <= width / 2) & (distances[edge] <= 2 * radius) & nearest
        mask[box] &= ~cut.reshape(points.shape[:-1])
        context.records.append(
            {
                "type": "gap",
                "edge": edge,
                "position": position.tolist(),
                "tangent": tangent.tolist(),
                "width": width,
            }
        )


def _sample_on_graph(
    context: _Context, margin: float, min_radius: float = 0.0
) -> tuple[int | None, np.ndarray, np.ndarray, float]:
    """Pick a random point on an edge, away from its end nodes.

    Edges are chosen with probability proportional to their length. The
    point keeps ``margin`` local radii from both ends.

    Returns
    -------
    edge, position, tangent, radius
        ``edge`` is ``None`` if no edge qualifies.
    """
    graph = context.graph
    candidates = []
    for edge, (path, radii) in enumerate(
        zip(graph.edge_paths, graph.edge_radii, strict=True)
    ):
        points, point_radii = resample_polyline(path, 0.25, radii)
        arc = _arc_length(points)
        allowed = (
            (arc >= margin * point_radii)
            & (arc[-1] - arc >= margin * point_radii)
            & (point_radii >= min_radius)
        )
        indices = np.flatnonzero(allowed)
        if len(indices):
            candidates.append((edge, points, point_radii, indices))
    if not candidates:
        return None, np.empty(0), np.empty(0), 0.0

    weights = np.array([len(indices) for *_, indices in candidates], dtype=float)
    edge, points, radii, indices = candidates[
        context.rng.choice(len(candidates), p=weights / weights.sum())
    ]
    index = context.rng.choice(indices)
    neighbour = index + 1 if index + 1 < len(points) else index - 1
    tangent = points[max(index, neighbour)] - points[min(index, neighbour)]
    tangent /= np.linalg.norm(tangent)
    return edge, points[index], tangent, float(radii[index])


def _arc_length(points: np.ndarray) -> np.ndarray:
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def _node_distances(graph: SkeletonGraph) -> np.ndarray:
    """Shortest distances between all nodes along the graph's edges."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import shortest_path

    weights = coo_matrix(
        (graph.edge_lengths, (graph.edges[:, 0], graph.edges[:, 1])),
        shape=(graph.n_nodes, graph.n_nodes),
    ).tocsr()
    return shortest_path(weights, directed=False)


def _random_perpendicular(tangent: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=len(tangent))
    vector -= np.dot(vector, tangent) * tangent
    return vector / np.linalg.norm(vector)


def _paint_ball(
    mask: np.ndarray,
    center: np.ndarray,
    radius: float,
    spacing: np.ndarray,
    value: bool,
) -> None:
    lower, upper = bounding_box(center[None], radius, spacing, mask.shape)
    if np.any(upper <= lower):
        return
    box, points = physical_grid(lower, upper, spacing)
    inside = np.linalg.norm(points - center, axis=-1) <= radius
    if value:
        mask[box] |= inside
    else:
        mask[box] &= ~inside


def _paint_capsule(
    mask: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    spacing: np.ndarray,
) -> None:
    lower, upper = bounding_box(np.stack([start, end]), radius, spacing, mask.shape)
    if np.any(upper <= lower):
        return
    box, points = physical_grid(lower, upper, spacing)
    _, distance = closest_on_segment(points, start, end)
    mask[box] |= distance <= radius
