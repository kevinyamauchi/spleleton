"""Rasterise skeleton graphs into binary masks of tubes."""

import numpy as np

from spleleton.graph import SkeletonGraph
from spleleton.synthetic._geometry import bounding_box, closest_on_segment


def rasterize_graph(
    graph: SkeletonGraph,
    shape: tuple[int, ...],
    spacing: tuple[float, ...] | None = None,
    supersample: int = 3,
) -> np.ndarray:
    """Rasterise the tubes around a graph's edges into a binary mask.

    Every polyline piece is a capsule whose radius is interpolated linearly
    between its two vertices; the tube is the union of the capsules. Each
    voxel is split into ``supersample`` sub-samples per axis, and it is inside
    if at least half of its sub-samples are.

    Parameters
    ----------
    graph : SkeletonGraph
        Graph with ``edge_radii``, in physical coordinates.
    shape : tuple of int
        Shape of the output mask.
    spacing : tuple of float, optional
        Physical voxel size per axis. Defaults to 1 on every axis.
    supersample : int
        Sub-samples per voxel and axis. ``1`` tests only voxel centres.

    Returns
    -------
    np.ndarray
        Boolean mask of the given shape.
    """
    if graph.edge_radii is None:
        raise ValueError("rasterizing requires a graph with edge_radii")
    if len(shape) != graph.ndim:
        raise ValueError(f"shape has {len(shape)} axes but the graph is {graph.ndim}D")
    spacing_array = _spacing_array(spacing, len(shape))

    # rasterise on a fine grid, then average blocks of sub-samples
    fine_shape = tuple(n * supersample for n in shape)
    fine_spacing = spacing_array / supersample
    # fine sample j sits at voxel coordinate (j + 0.5) / supersample - 0.5
    fine_origin = spacing_array * (0.5 / supersample - 0.5)
    fine = np.zeros(fine_shape, dtype=bool)

    for path, radii in zip(graph.edge_paths, graph.edge_radii, strict=True):
        for piece in range(len(path) - 1):
            start, end = path[piece], path[piece + 1]
            r_start, r_end = radii[piece], radii[piece + 1]
            lower, upper = bounding_box(
                np.stack([start, end]) - fine_origin,
                max(r_start, r_end),
                fine_spacing,
                fine_shape,
            )
            if np.any(upper <= lower):
                continue
            box = tuple(slice(lo, hi) for lo, hi in zip(lower, upper, strict=True))
            axes = [
                fine_origin[axis] + np.arange(lo, hi) * fine_spacing[axis]
                for axis, (lo, hi) in enumerate(zip(lower, upper, strict=True))
            ]
            points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
            t, distance = closest_on_segment(points, start, end)
            fine[box] |= distance <= r_start + t * (r_end - r_start)

    blocks = fine.reshape([n for size in shape for n in (size, supersample)]).mean(
        axis=tuple(range(1, 2 * len(shape), 2))
    )
    return blocks >= 0.5


def _spacing_array(spacing: tuple[float, ...] | None, ndim: int) -> np.ndarray:
    if spacing is None:
        return np.ones(ndim)
    spacing_array = np.asarray(spacing, dtype=float)
    if spacing_array.shape != (ndim,):
        raise ValueError(f"spacing must have {ndim} entries, got {spacing_array.shape}")
    return spacing_array
