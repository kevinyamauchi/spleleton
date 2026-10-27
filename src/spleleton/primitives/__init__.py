"""Differentiable skeleton primitives.

Renderers rely only on this interface, so that line segments can later be
swapped for splines:

- ``query_indexed`` returns, for (point, primitive) pairs, the parameter of
  the closest point, the closest point, the distance to it and the radius
  there (a :class:`PrimitiveQuery`);
- ``bounds`` gives each primitive's axis-aligned bounding box (of its
  centreline, without the radius);
- ``radius`` and ``opacity`` give the per-primitive tube radius and opacity.

Line segments exist once per backend (``TorchLineSegments``, and
``JaxLineSegments``). :class:`SegmentArrays` holds the same parameters as
NumPy arrays, for host-side steps that work with either backend.
"""

from typing import TYPE_CHECKING

from spleleton._backends import lazy_exports
from spleleton.primitives._arrays import (
    PrimitiveQuery,
    SegmentArrays,
    as_segment_arrays,
)

if TYPE_CHECKING:
    from spleleton.primitives._primitives_jax import JaxLineSegments
    from spleleton.primitives._primitives_torch import TorchLineSegments

__all__ = [
    "JaxLineSegments",
    "PrimitiveQuery",
    "SegmentArrays",
    "TorchLineSegments",
    "as_segment_arrays",
]

__getattr__, __dir__ = lazy_exports(
    __name__,
    torch={"._primitives_torch": ["TorchLineSegments"]},
    jax={"._primitives_jax": ["JaxLineSegments"]},
)
