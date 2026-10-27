"""Differentiable skeleton primitives, in JAX.

See :mod:`spleleton.primitives` for the interface renderers rely on.
"""

import dataclasses
from typing import TYPE_CHECKING, ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from spleleton.geometry._common import DEFAULT_EPS
from spleleton.geometry._geometry_jax import (
    closest_points_on_segments_jax,
    lerp,
    pairwise_closest_points_jax,
    segment_lengths_jax,
    segment_tangents_jax,
)
from spleleton.primitives._arrays import PrimitiveQuery, SegmentArrays

if TYPE_CHECKING:
    from spleleton.graph import SkeletonGraph


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True, eq=False)
class JaxLineSegments:
    """A set of line segments in 2D or 3D with a radius profile and an opacity.

    The radius is defined at both endpoints and interpolated linearly along
    the segment. Radius and opacity are stored unconstrained (log-radius and
    opacity logit) so that gradient steps keep them valid.

    This is an immutable pytree: ``jax.grad`` of a function of the segments
    returns a ``JaxLineSegments`` of gradients, and an optimiser step returns
    new segments. The constructor takes the stored (unconstrained) fields
    directly and does not validate them, because JAX calls it when it
    rebuilds pytrees; use :meth:`create` to build segments from endpoints,
    radii and opacities.

    Parameters
    ----------
    start : jax.Array
        Segment start points, shape ``(N, D)``.
    end : jax.Array
        Segment end points, shape ``(N, D)``.
    log_radius : jax.Array
        Log of the radius at the start and end, shape ``(N, 2)``.
    opacity_logit : jax.Array
        Logit of the opacity, shape ``(N,)``.
    """

    start: Array
    end: Array
    log_radius: Array
    opacity_logit: Array

    #: Name of the array backend.
    backend: ClassVar[str] = "jax"

    @classmethod
    def create(
        cls,
        start: Array,
        end: Array,
        radius: float | Array = 1.0,
        opacity: float | Array = 0.5,
        dtype: jnp.dtype | None = None,
    ) -> "JaxLineSegments":
        """Segments from endpoints, radii and opacities.

        Shapes are always checked. Values (positive radii, opacities inside
        ``(0, 1)``) are checked unless they are traced, i.e. inside ``jit``.

        Parameters
        ----------
        start : array_like
            Segment start points, shape ``(N, D)`` with D = 2 or 3.
        end : array_like
            Segment end points, the same shape as ``start``.
        radius : float or array_like
            Tube radius. A scalar, shape ``(N,)`` for a constant radius per
            segment, or shape ``(N, 2)`` for the radius at the start and end.
            Must be positive.
        opacity : float or array_like
            Opacity, a scalar or shape ``(N,)``, in the open interval
            ``(0, 1)``.
        dtype : dtype, optional
            Floating-point type. By default, floating-point ``start`` keeps
            its type (as far as JAX's precision setting allows) and anything
            else gets JAX's default float type.

        Returns
        -------
        JaxLineSegments
            The segments.
        """
        start = _as_float_array(start, dtype)
        end = jnp.asarray(end, dtype=start.dtype)
        if start.ndim != 2 or start.shape[-1] not in (2, 3):
            raise ValueError(
                f"start must have shape (N, D) with D = 2 or 3, got {start.shape}"
            )
        if end.shape != start.shape:
            raise ValueError(
                f"end must have the same shape as start {start.shape}, got {end.shape}"
            )
        n_segments = start.shape[0]

        radius = jnp.asarray(radius, dtype=start.dtype)
        if radius.ndim == 1:
            radius = radius[:, None]
        try:
            radius = jnp.broadcast_to(radius, (n_segments, 2))
        except ValueError as error:
            raise ValueError(
                "radius must be a scalar or have shape (N,) or (N, 2), "
                f"got {radius.shape}"
            ) from error
        if _is_concrete(radius) and not bool((radius > 0).all()):
            raise ValueError("radius must be positive")

        opacity = jnp.asarray(opacity, dtype=start.dtype)
        try:
            opacity = jnp.broadcast_to(opacity, (n_segments,))
        except ValueError as error:
            raise ValueError(
                f"opacity must be a scalar or have shape (N,), got {opacity.shape}"
            ) from error
        if _is_concrete(opacity) and not bool(((opacity > 0) & (opacity < 1)).all()):
            raise ValueError("opacity must lie in the open interval (0, 1)")

        return cls(
            start=start,
            end=end,
            log_radius=jnp.log(radius),
            # as torch.logit computes it, so both backends start identically
            opacity_logit=jnp.log(opacity / (1 - opacity)),
        )

    @classmethod
    def from_arrays(
        cls,
        arrays: SegmentArrays,
        dtype: jnp.dtype | None = None,
        device: jax.Device | None = None,
    ) -> "JaxLineSegments":
        """Segments with the given (NumPy) parameters.

        Parameters
        ----------
        arrays : SegmentArrays
            The parameters.
        dtype : dtype, optional
            Floating-point type. Defaults to JAX's default float type
            (float32 unless 64-bit mode is on).
        device : jax.Device, optional
            Device of the parameters. Defaults to JAX's default device.

        Returns
        -------
        JaxLineSegments
            The segments.
        """
        dtype = dtype or jnp.result_type(float)
        segments = cls.create(
            jnp.asarray(arrays.start, dtype=dtype),
            arrays.end,
            radius=arrays.radius,
            opacity=arrays.opacity,
        )
        if device is not None:
            segments = jax.device_put(segments, device)
        return segments

    @classmethod
    def from_graph(
        cls,
        graph: "SkeletonGraph",
        radius: float = 1.0,
        opacity: float = 0.99,
        dtype: jnp.dtype | None = None,
        device: jax.Device | None = None,
    ) -> "JaxLineSegments":
        """One segment per polyline piece of every edge of a graph.

        Parameters
        ----------
        graph : SkeletonGraph
            The graph, in physical coordinates.
        radius : float
            Radius used if the graph has no radii.
        opacity : float
            Opacity of every segment.
        dtype : dtype, optional
            Floating-point type. Defaults to JAX's default float type.
        device : jax.Device, optional
            Device of the parameters.

        Returns
        -------
        JaxLineSegments
            The segments, ordered edge by edge.
        """
        return cls.from_arrays(
            SegmentArrays.from_graph(graph, radius=radius, opacity=opacity),
            dtype=dtype,
            device=device,
        )

    def to_arrays(self) -> SegmentArrays:
        """The parameters, copied to NumPy (float64). Not jittable."""

        def array(values: Array) -> np.ndarray:
            return np.asarray(values, dtype=np.float64)

        return SegmentArrays(
            array(self.start),
            array(self.end),
            radius=array(self.radius),
            opacity=array(self.opacity),
        )

    def __len__(self) -> int:
        """Number of segments."""
        return self.start.shape[0]

    @property
    def radius(self) -> Array:
        """Radius at the start and end of each segment, shape ``(N, 2)``."""
        return jnp.exp(self.log_radius)

    @property
    def opacity(self) -> Array:
        """Opacity of each segment, shape ``(N,)``."""
        return jax.nn.sigmoid(self.opacity_logit)

    @property
    def lengths(self) -> Array:
        """Length of each segment, shape ``(N,)``."""
        return segment_lengths_jax(self.start, self.end)

    @property
    def tangents(self) -> Array:
        """Unit tangent of each segment, from start to end, shape ``(N, D)``."""
        return segment_tangents_jax(self.start, self.end)

    def tube_volumes(self) -> Array:
        """Volume (area in 2D) of each segment's tube, a capsule, shape ``(N,)``.

        Uses the segment's mean radius ``r`` and length ``L``: ``2 r L + pi
        r^2`` in 2D and ``pi r^2 L + 4/3 pi r^3`` in 3D, in physical units.
        """
        r = self.radius.mean(axis=-1)
        length = self.lengths
        if self.start.shape[-1] == 2:
            return 2 * r * length + jnp.pi * r**2
        return jnp.pi * r**2 * length + 4 / 3 * jnp.pi * r**3

    @property
    def midpoints(self) -> Array:
        """Midpoint of each segment, shape ``(N, D)``."""
        return 0.5 * (self.start + self.end)

    def radius_at(self, t: Array) -> Array:
        """Radius at parameter values along each segment.

        Parameters
        ----------
        t : jax.Array
            Parameter values, shape ``(..., N)``. Each column refers to the
            segment with the same index.

        Returns
        -------
        jax.Array
            Radii, shape ``(..., N)``.
        """
        radius = self.radius
        return lerp(radius[:, 0], radius[:, 1], t)

    def bounds(self) -> tuple[Array, Array]:
        """Axis-aligned bounding box of each segment, without its radius.

        Returns
        -------
        lower, upper : jax.Array
            Corners of the boxes, each of shape ``(N, D)``.
        """
        return jnp.minimum(self.start, self.end), jnp.maximum(self.start, self.end)

    def query_indexed(
        self, points: Array, index: Array, eps: float = DEFAULT_EPS
    ) -> PrimitiveQuery:
        """Closest point, distance and radius for groups of points and segments.

        Each group of points is paired with its own list of segments, which
        is how block-binned renderers evaluate only nearby pairs.

        Parameters
        ----------
        points : jax.Array
            Query points, shape ``(..., P, D)``.
        index : jax.Array
            Segment indices, shape ``(..., K)``, matching the leading
            dimensions of ``points``.
        eps : float
            Floor on squared segment lengths.

        Returns
        -------
        PrimitiveQuery
            Fields of shape ``(..., P, K)``, and ``(..., P, K, D)`` for
            ``point``.
        """
        start = self.start[index][..., None, :, :]
        end = self.end[index][..., None, :, :]
        closest = closest_points_on_segments_jax(
            points[..., :, None, :], start, end, eps=eps
        )
        radius = self.radius[index][..., None, :, :]
        return PrimitiveQuery(
            t=closest.t,
            point=closest.point,
            distance=closest.distance,
            radius=lerp(radius[..., 0], radius[..., 1], closest.t),
        )

    def query(self, points: Array, eps: float = DEFAULT_EPS) -> PrimitiveQuery:
        """Closest point, distance and radius for every (point, segment) pair.

        Parameters
        ----------
        points : jax.Array
            Query points, shape ``(M, D)``.
        eps : float
            Floor on squared segment lengths.

        Returns
        -------
        PrimitiveQuery
            Fields of shape ``(M, N)``, and ``(M, N, D)`` for ``point``.
        """
        closest = pairwise_closest_points_jax(points, self.start, self.end, eps=eps)
        return PrimitiveQuery(
            t=closest.t,
            point=closest.point,
            distance=closest.distance,
            radius=self.radius_at(closest.t),
        )

    def __repr__(self) -> str:
        """Summary, without the arrays."""
        return f"JaxLineSegments(n_segments={len(self)}, dtype={self.start.dtype})"


def _as_float_array(value: Array, dtype: jnp.dtype | None) -> Array:
    """A floating-point JAX array, as ``_as_float_tensor`` does for torch."""
    if dtype is not None:
        return jnp.asarray(value, dtype=dtype)
    array = jnp.asarray(value)
    if not jnp.issubdtype(array.dtype, jnp.floating):
        array = array.astype(jnp.result_type(float))
    return array


def _is_concrete(value: Array) -> bool:
    """Whether a value is known now (not traced inside ``jit``)."""
    return not isinstance(value, jax.core.Tracer)
