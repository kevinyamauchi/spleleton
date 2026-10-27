"""Host-side steps of the pipelines (shared by all backends)."""

import numpy as np
from scipy import ndimage

from spleleton.pipeline._config import ScratchFitConfig
from spleleton.primitives._arrays import SegmentArrays
from spleleton.seeding import seed_segments


def initial_seeds(
    mask: np.ndarray, spacing: tuple[float, ...] | None, config: ScratchFitConfig
) -> SegmentArrays:
    """The seeds of the first round of :func:`scratch_fit_torch` and its kin."""
    return seed_segments(mask, spacing, spacing_factor=config.spacing_factor)


def spawn_seeds(
    mask: np.ndarray,
    covered: np.ndarray,
    spacing: tuple[float, ...] | None,
    config: ScratchFitConfig,
) -> SegmentArrays:
    """New seeds in the parts of the mask that the rendering does not cover.

    Uncovered connected regions (26- or 8-connected) are seeded if they have
    at least ``config.min_spawn_voxels`` voxels and are at least
    ``config.min_spawn_thickness`` thick somewhere (see
    :class:`ScratchFitConfig`).

    Parameters
    ----------
    mask : np.ndarray
        Binary mask.
    covered : np.ndarray
        Where the current segments' rendering is at least 0.5.
    spacing : tuple of float, optional
        Physical voxel size per axis.
    config : ScratchFitConfig
        Pipeline settings.

    Returns
    -------
    SegmentArrays
        The new seeds (possibly none).
    """
    uncovered = mask & ~covered
    labels, n_labels = ndimage.label(uncovered, np.ones((3,) * mask.ndim))
    sizes = np.bincount(labels.ravel(), minlength=n_labels + 1)
    keep = sizes >= config.min_spawn_voxels
    min_thickness = config.min_spawn_thickness
    if min_thickness is None:
        min_thickness = 0.0 if mask.ndim == 2 else 1.5
    if min_thickness > 0 and n_labels:
        depth = ndimage.distance_transform_edt(uncovered, sampling=spacing)
        thickness = ndimage.maximum(depth, labels, np.arange(n_labels + 1))
        keep &= np.asarray(thickness) >= min_thickness
    keep[0] = False
    large = np.isin(labels, np.flatnonzero(keep))
    return seed_segments(
        mask, spacing, spacing_factor=config.spacing_factor, candidates=large
    )
