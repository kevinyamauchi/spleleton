"""Differentiable geometry utilities for line segments in 2D or 3D.

All functions act on the last dimension (size D, 2 or 3) and broadcast over the
leading dimensions. They therefore work for a single segment, a batch of segments, or a
points-by-segments grid (e.g. points of shape ``(M, 1, D)`` against segments of
shape ``(1, N, D)``).

Coordinates are in array-index space: axis 0, 1 (, 2) of the image, with voxel
centres at integer coordinates.

Each function exists once per backend, with a ``_torch`` or ``_jax`` suffix.
"""

from typing import TYPE_CHECKING

from spleleton._backends import lazy_exports
from spleleton.geometry._common import DEFAULT_EPS, ClosestPoints

if TYPE_CHECKING:
    from spleleton.geometry._geometry_jax import (
        closest_points_on_segments_jax,
        pairwise_closest_points_jax,
        points_on_segments_jax,
        safe_norm_jax,
        segment_lengths_jax,
        segment_tangents_jax,
    )
    from spleleton.geometry._geometry_torch import (
        closest_points_on_segments_torch,
        pairwise_closest_points_torch,
        points_on_segments_torch,
        safe_norm_torch,
        segment_lengths_torch,
        segment_tangents_torch,
    )

__all__ = [
    "DEFAULT_EPS",
    "ClosestPoints",
    "closest_points_on_segments_jax",
    "closest_points_on_segments_torch",
    "pairwise_closest_points_jax",
    "pairwise_closest_points_torch",
    "points_on_segments_jax",
    "points_on_segments_torch",
    "safe_norm_jax",
    "safe_norm_torch",
    "segment_lengths_jax",
    "segment_lengths_torch",
    "segment_tangents_jax",
    "segment_tangents_torch",
]

__getattr__, __dir__ = lazy_exports(
    __name__,
    torch={
        "._geometry_torch": [
            "closest_points_on_segments_torch",
            "pairwise_closest_points_torch",
            "points_on_segments_torch",
            "safe_norm_torch",
            "segment_lengths_torch",
            "segment_tangents_torch",
        ]
    },
    jax={
        "._geometry_jax": [
            "closest_points_on_segments_jax",
            "pairwise_closest_points_jax",
            "points_on_segments_jax",
            "safe_norm_jax",
            "segment_lengths_jax",
            "segment_tangents_jax",
        ]
    },
)
