"""Seed line segments directly from a mask, without a skeleton (Phase 5).

Seeds are placed on the ridge of the Euclidean distance transform (EDT),
thinned so that seeds are about ``spacing_factor`` local radii apart. Each
seed is a short segment through its point, along the local tube direction
(the principal axis of the mask voxels around it), with the EDT as radius.
"""

import numpy as np
from scipy import ndimage

from spleleton.primitives._arrays import SegmentArrays


def seed_segments(
    mask: np.ndarray,
    spacing: tuple[float, ...] | None = None,
    spacing_factor: float = 2.0,
    length_factor: float = 1.0,
    min_radius: float = 0.75,
    candidates: np.ndarray | None = None,
    opacity: float = 0.5,
) -> SegmentArrays:
    """Short segments on the ridge of the distance transform.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask, 2D or 3D.
    spacing : tuple of float, optional
        Physical voxel size per axis.
    spacing_factor : float
        Seeds are at least this many local radii apart.
    length_factor : float
        Each seed extends this many radii to either side of its point.
    min_radius : float
        Ridge points with a smaller EDT are not used.
    candidates : np.ndarray, optional
        Boolean image restricting where seeds may be placed (e.g. regions not
        yet covered). Radii still come from the EDT of the whole mask.
    opacity : float
        Initial opacity.

    Returns
    -------
    SegmentArrays
        The seeds (possibly none); convert them with
        ``TorchLineSegments.from_arrays`` or ``JaxLineSegments.from_arrays`` to
        fit them.
    """
    mask = np.asarray(mask, dtype=bool)
    ndim = mask.ndim
    spacing_array = np.ones(ndim) if spacing is None else np.asarray(spacing, float)
    edt = ndimage.distance_transform_edt(mask, sampling=spacing_array)
    ridge = (
        mask & (edt >= ndimage.maximum_filter(edt, size=3) - 1e-9) & (edt >= min_radius)
    )
    if candidates is not None:
        ridge &= np.asarray(candidates, dtype=bool)
    points = np.argwhere(ridge)
    radii = edt[ridge]
    order = np.argsort(-radii, kind="stable")

    # greedy thinning, thickest first: a seed blocks a ball around it
    blocked = np.zeros(mask.shape, dtype=bool)
    starts, ends, seed_radii = [], [], []
    for index in order:
        voxel = points[index]
        if blocked[tuple(voxel)]:
            continue
        radius = float(radii[index])
        centre = voxel * spacing_array
        axis = _principal_axis(mask, voxel, 2 * radius, spacing_array)
        starts.append(centre - length_factor * radius * axis)
        ends.append(centre + length_factor * radius * axis)
        seed_radii.append(radius)
        _block_ball(blocked, voxel, spacing_factor * radius, spacing_array)

    def array(values: list, width: int) -> np.ndarray:
        return np.array(values, dtype=float).reshape(-1, width)

    return SegmentArrays(
        array(starts, ndim),
        array(ends, ndim),
        radius=array(seed_radii, 1).reshape(-1),
        opacity=opacity,
    )


def _principal_axis(
    mask: np.ndarray, voxel: np.ndarray, radius: float, spacing: np.ndarray
) -> np.ndarray:
    """Principal axis of the mask voxels within a ball around a voxel."""
    lower = np.maximum(np.floor(voxel - radius / spacing).astype(int), 0)
    upper = np.minimum(np.ceil(voxel + radius / spacing).astype(int) + 1, mask.shape)
    box = tuple(slice(lo, hi) for lo, hi in zip(lower, upper, strict=True))
    offsets = (np.argwhere(mask[box]) + lower - voxel) * spacing
    offsets = offsets[np.linalg.norm(offsets, axis=1) <= radius]
    if len(offsets) < mask.ndim + 1:
        axis = np.zeros(mask.ndim)
        axis[0] = 1.0
        return axis
    _, vectors = np.linalg.eigh(np.cov(offsets.T))
    return vectors[:, -1]


def _block_ball(
    blocked: np.ndarray, voxel: np.ndarray, radius: float, spacing: np.ndarray
) -> None:
    lower = np.maximum(np.floor(voxel - radius / spacing).astype(int), 0)
    upper = np.minimum(np.ceil(voxel + radius / spacing).astype(int) + 1, blocked.shape)
    box = tuple(slice(lo, hi) for lo, hi in zip(lower, upper, strict=True))
    axes = [
        (np.arange(lo, hi) - v) * step
        for lo, hi, v, step in zip(lower, upper, voxel, spacing, strict=True)
    ]
    grid = np.meshgrid(*axes, indexing="ij")
    blocked[box] |= sum(g**2 for g in grid) <= radius**2
