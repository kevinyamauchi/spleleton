"""Backend-neutral segment parameters and query results."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from spleleton.graph import SkeletonGraph

#: Opacities are kept this far from 0 and 1 when segment sets are concatenated.
_CONCATENATE_OPACITY_EPS = 1e-6


class PrimitiveQuery(NamedTuple):
    """Closest-point information for every (point, primitive) pair.

    The fields are arrays of the backend that computed them (torch tensors or
    JAX arrays).

    Attributes
    ----------
    t : array
        Parameter of the closest point on each primitive, shape ``(M, N)``.
    point : array
        Closest point on each primitive, shape ``(M, N, D)``.
    distance : array
        Distance from each query point to each primitive, shape ``(M, N)``.
    radius : array
        Tube radius of each primitive at its closest point, shape ``(M, N)``.
    """

    t: Any
    point: Any
    distance: Any
    radius: Any


@dataclass(frozen=True)
class SegmentArrays:
    """Line segment parameters as plain NumPy arrays, for either backend.

    Host-side steps (seeding, initialisation from a skeleton, assembly,
    recording) work on these. Convert to a differentiable backend with
    ``TorchLineSegments.from_arrays`` or ``JaxLineSegments.from_arrays``, and
    back with their ``to_arrays`` method.

    Parameters
    ----------
    start : array_like
        Segment start points, shape ``(N, D)``.
    end : array_like
        Segment end points, shape ``(N, D)``.
    radius : float or array_like
        Tube radius: a scalar, shape ``(N,)`` for a constant radius per
        segment, or shape ``(N, 2)`` for the radius at the start and end.
        Stored with shape ``(N, 2)``. Defaults to 1.
    opacity : float or array_like
        Opacity, a scalar or shape ``(N,)``. Stored with shape ``(N,)``.
        Defaults to 0.5.
    """

    start: np.ndarray
    end: np.ndarray
    radius: np.ndarray = 1.0  # type: ignore[assignment]
    opacity: np.ndarray = 0.5  # type: ignore[assignment]

    def __post_init__(self) -> None:
        """Convert to float64 arrays and broadcast radius and opacity."""
        start = np.asarray(self.start, dtype=np.float64)
        end = np.asarray(self.end, dtype=np.float64)
        if start.ndim != 2 or end.shape != start.shape:
            raise ValueError(
                "start and end must have the same shape (N, D), "
                f"got {start.shape} and {end.shape}"
            )
        n_segments = len(start)
        radius = np.asarray(self.radius, dtype=np.float64)
        if radius.ndim == 1:
            radius = radius[:, None]
        try:
            radius = np.broadcast_to(radius, (n_segments, 2)).copy()
        except ValueError as error:
            raise ValueError(
                "radius must be a scalar or have shape (N,) or (N, 2), "
                f"got {np.shape(self.radius)}"
            ) from error
        opacity = np.asarray(self.opacity, dtype=np.float64)
        try:
            opacity = np.broadcast_to(opacity, (n_segments,)).copy()
        except ValueError as error:
            raise ValueError(
                "opacity must be a scalar or have shape (N,), "
                f"got {np.shape(self.opacity)}"
            ) from error
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "radius", radius)
        object.__setattr__(self, "opacity", opacity)

    @classmethod
    def from_graph(
        cls, graph: "SkeletonGraph", radius: float = 1.0, opacity: float = 0.99
    ) -> "SegmentArrays":
        """One segment per polyline piece of every edge of a graph.

        Parameters
        ----------
        graph : SkeletonGraph
            The graph, in physical coordinates.
        radius : float
            Radius used if the graph has no radii.
        opacity : float
            Opacity of every segment.

        Returns
        -------
        SegmentArrays
            The segments, ordered edge by edge.
        """
        empty = np.empty((0, graph.ndim))
        starts = [path[:-1] for path in graph.edge_paths]
        ends = [path[1:] for path in graph.edge_paths]
        if graph.edge_radii is None:
            radii = radius
        elif graph.edge_radii:
            radii = np.concatenate(
                [np.stack([r[:-1], r[1:]], axis=1) for r in graph.edge_radii]
            )
        else:
            radii = np.empty((0, 2))
        return cls(
            np.concatenate(starts) if starts else empty,
            np.concatenate(ends) if ends else empty,
            radius=radii,
            opacity=opacity,
        )

    def __len__(self) -> int:
        """Number of segments."""
        return len(self.start)

    @property
    def ndim(self) -> int:
        """Dimension D of the space."""
        return self.start.shape[1]

    def to_arrays(self) -> "SegmentArrays":
        """Return itself, so that any segment set can be converted alike."""
        return self

    def select(self, index: np.ndarray) -> "SegmentArrays":
        """A subset of the segments, by boolean mask or integer indices."""
        return SegmentArrays(
            self.start[index], self.end[index], self.radius[index], self.opacity[index]
        )

    def concatenate(self, other: "SegmentArrays") -> "SegmentArrays":
        """Both sets of segments, this one first.

        Opacities are clamped to ``[1e-6, 1 - 1e-6]``, so that saturated
        segments keep a finite opacity logit when converted back to a backend.
        """
        opacity = np.clip(
            np.concatenate([self.opacity, other.opacity]),
            _CONCATENATE_OPACITY_EPS,
            1 - _CONCATENATE_OPACITY_EPS,
        )
        return SegmentArrays(
            np.concatenate([self.start, other.start]),
            np.concatenate([self.end, other.end]),
            radius=np.concatenate([self.radius, other.radius]),
            opacity=opacity,
        )


def as_segment_arrays(segments: Any) -> SegmentArrays:
    """Parameters of any segment set (either backend, or arrays) as NumPy.

    Parameters
    ----------
    segments : SegmentArrays, TorchLineSegments or JaxLineSegments
        The segments. Backend segments are detached and copied to the host.

    Returns
    -------
    SegmentArrays
        Float64 copies of the parameters.
    """
    to_arrays = getattr(segments, "to_arrays", None)
    if to_arrays is None:
        raise TypeError(
            "expected SegmentArrays, TorchLineSegments or JaxLineSegments, "
            f"got {type(segments).__name__}"
        )
    return to_arrays()
