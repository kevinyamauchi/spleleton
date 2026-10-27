"""NumPy closest-point helpers for data generation.

These mirror :mod:`spleleton.geometry`, whose PyTorch and JAX versions are
for optimisation. Data generation runs on the CPU in NumPy and does not need
gradients, so it uses this small duplicate instead of depending on a backend.
"""

import numpy as np


def closest_on_segment(
    points: np.ndarray, start: np.ndarray, end: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Parameter and distance of the closest point on a segment.

    Parameters
    ----------
    points : np.ndarray
        Query points, shape ``(..., D)``.
    start : np.ndarray
        Segment start, shape ``(D,)`` or broadcastable with ``points``.
    end : np.ndarray
        Segment end, same shape as ``start``.

    Returns
    -------
    t : np.ndarray
        Parameter of the closest point in ``[0, 1]``, shape ``(...)``.
    distance : np.ndarray
        Distance to the closest point, shape ``(...)``.
    """
    vector = end - start
    squared_length = max(float(np.sum(vector * vector)), 1e-12)
    t = np.clip(np.sum((points - start) * vector, axis=-1) / squared_length, 0.0, 1.0)
    closest = start + t[..., None] * vector
    return t, np.linalg.norm(points - closest, axis=-1)


def distance_to_polyline(
    points: np.ndarray, path: np.ndarray, radii: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Distance from points to a polyline, with the radius and tangent there.

    Parameters
    ----------
    points : np.ndarray
        Query points, shape ``(M, D)``.
    path : np.ndarray
        Polyline vertices, shape ``(P, D)``.
    radii : np.ndarray, optional
        Radius at each vertex, shape ``(P,)``, interpolated linearly.

    Returns
    -------
    distance : np.ndarray
        Distance to the closest point on the polyline, shape ``(M,)``.
    radius : np.ndarray or None
        Radius at the closest point, shape ``(M,)``.
    tangent : np.ndarray
        Unit tangent of the closest piece, shape ``(M, D)``.
    """
    best_distance = np.full(len(points), np.inf)
    best_radius = None if radii is None else np.zeros(len(points))
    best_tangent = np.zeros_like(points, dtype=float)
    for piece in range(len(path) - 1):
        start, end = path[piece], path[piece + 1]
        t, distance = closest_on_segment(points, start, end)
        closer = distance < best_distance
        best_distance[closer] = distance[closer]
        vector = end - start
        best_tangent[closer] = vector / max(float(np.linalg.norm(vector)), 1e-12)
        if radii is not None:
            best_radius[closer] = (
                radii[piece] + t * (radii[piece + 1] - radii[piece])
            )[closer]
    return best_distance, best_radius, best_tangent


def physical_grid(
    lower: np.ndarray, upper: np.ndarray, spacing: np.ndarray
) -> tuple[tuple[slice, ...], np.ndarray]:
    """Voxel centres inside an index-space bounding box.

    Parameters
    ----------
    lower, upper : np.ndarray
        Inclusive and exclusive voxel-index bounds, shape ``(D,)``.
    spacing : np.ndarray
        Voxel size, shape ``(D,)``.

    Returns
    -------
    box : tuple of slice
        The bounding box as slices into the image.
    points : np.ndarray
        Physical coordinates of the voxel centres, shape ``(*box_shape, D)``.
    """
    box = tuple(slice(int(lo), int(hi)) for lo, hi in zip(lower, upper, strict=True))
    axes = [
        np.arange(s.start, s.stop) * step for s, step in zip(box, spacing, strict=True)
    ]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    return box, points


def bounding_box(
    points: np.ndarray, margin: float, spacing: np.ndarray, shape: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Index-space bounding box of physical points grown by a margin.

    Returns inclusive lower and exclusive upper voxel indices, clipped to the
    image. The box is empty (``upper <= lower`` on some axis) if the points
    lie outside the image.
    """
    lower = np.floor((points.min(axis=0) - margin) / spacing).astype(int)
    upper = np.ceil((points.max(axis=0) + margin) / spacing).astype(int) + 1
    return np.clip(lower, 0, shape), np.clip(upper, 0, shape)
