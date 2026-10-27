"""Differentiable geometry utilities for line segments, in JAX.

See :mod:`spleleton.geometry` for the conventions (broadcasting, coordinates).

Every function is jittable and differentiable. Where torch and JAX differ in
how a gradient is split at a kink, these functions follow torch, so that the
two backends give the same gradients:

- clamping passes the gradient only where the value is strictly inside the
  bounds, and none at a bound itself (as ``torch.clamp`` does);
  ``jnp.clip`` would split it there;
- linear interpolation uses ``torch.lerp``'s two-sided formula, which is also
  what keeps float32 results of both backends close.

The closest-point computation works one coordinate at a time, so that pairs of
points and segments never form an array with a trailing coordinate axis (see
:func:`closest_points_on_segments_jax`).
"""

import functools
import operator

import jax.numpy as jnp
from jax import Array

from spleleton.geometry._common import DEFAULT_EPS, ClosestPoints


def safe_norm_jax(vectors: Array) -> Array:
    """Euclidean norm along the last dimension, with a zero gradient at zero.

    ``jnp.linalg.norm`` has an undefined (NaN) gradient at the zero vector.
    Points that lie exactly on a segment are common (e.g. voxel centres on an
    axis-aligned centreline), so the distance must stay differentiable there.

    Parameters
    ----------
    vectors : jax.Array
        Array with a trailing dimension holding the vector components.

    Returns
    -------
    jax.Array
        The norms, with the last dimension removed.
    """
    return _safe_sqrt((vectors * vectors).sum(axis=-1))


def _safe_sqrt(squared: Array) -> Array:
    """Square root of squared norms, with a zero gradient at zero."""
    nonzero = squared > 0
    safe_squared = jnp.where(nonzero, squared, jnp.ones_like(squared))
    return jnp.where(nonzero, jnp.sqrt(safe_squared), jnp.zeros_like(squared))


def segment_lengths_jax(start: Array, end: Array) -> Array:
    """Lengths of segments.

    Parameters
    ----------
    start : jax.Array
        Segment start points, shape ``(..., D)``.
    end : jax.Array
        Segment end points, shape ``(..., D)``.

    Returns
    -------
    jax.Array
        Segment lengths, shape ``(...)``.
    """
    return safe_norm_jax(end - start)


def segment_tangents_jax(start: Array, end: Array, eps: float = DEFAULT_EPS) -> Array:
    """Unit tangents of segments, pointing from start to end.

    Parameters
    ----------
    start : jax.Array
        Segment start points, shape ``(..., D)``.
    end : jax.Array
        Segment end points, shape ``(..., D)``.
    eps : float
        Floor on the segment length. Degenerate (zero-length) segments get a
        zero tangent instead of NaN.

    Returns
    -------
    jax.Array
        Unit tangents, shape ``(..., D)``.
    """
    vectors = end - start
    lengths = clamp_min(safe_norm_jax(vectors), eps)
    return vectors / lengths[..., None]


def points_on_segments_jax(start: Array, end: Array, t: Array) -> Array:
    """Evaluate segments at parameter values.

    Parameters
    ----------
    start : jax.Array
        Segment start points, shape ``(..., D)``.
    end : jax.Array
        Segment end points, shape ``(..., D)``.
    t : jax.Array
        Parameter values, shape ``(...)``. ``0`` gives the start and ``1`` the
        end. Values outside ``[0, 1]`` extrapolate along the line.

    Returns
    -------
    jax.Array
        Points, shape ``(..., D)``.
    """
    return start + t[..., None] * (end - start)


def closest_points_on_segments_jax(
    points: Array, start: Array, end: Array, eps: float = DEFAULT_EPS
) -> ClosestPoints:
    """Find the closest point on each segment to each query point.

    The closest point is the orthogonal projection onto the segment's line,
    clamped to the segment. When the projection is clamped, the gradient with
    respect to ``t`` is zero, and gradients reach the endpoints through the
    closest point itself.

    The arithmetic is that of the torch version, but done one coordinate at a
    time. With inputs that broadcast against each other, such as points
    ``(P, 1, D)`` and segments ``(1, K, D)``, every intermediate then has the
    pair shape ``(P, K)`` rather than ``(P, K, D)``, and each dot product is a
    sum of D arrays rather than a reduction over a short last axis. XLA fuses
    that into a few elementwise passes, several times faster on a CPU than
    with ``(P, K, D)`` intermediates. Only the returned ``point`` has a
    coordinate axis, and XLA drops it when it is unused.

    Parameters
    ----------
    points : jax.Array
        Query points, shape ``(..., D)``.
    start : jax.Array
        Segment start points, shape ``(..., D)``, broadcastable with
        ``points``.
    end : jax.Array
        Segment end points, shape ``(..., D)``, broadcastable with ``points``.
    eps : float
        Floor on squared segment lengths. A degenerate segment behaves like
        the point ``start``.

    Returns
    -------
    ClosestPoints
        ``t``, ``point`` and ``distance``, broadcast over the leading
        dimensions of the inputs.
    """
    x, a, b = (_coordinates(values) for values in (points, start, end))
    vectors = [bd - ad for ad, bd in zip(a, b, strict=True)]
    squared_lengths = _dot(vectors, vectors)
    offsets = [xd - ad for xd, ad in zip(x, a, strict=True)]
    projections = _dot(offsets, vectors)
    t = clamp(projections / clamp_min(squared_lengths, eps), 0.0, 1.0)
    # as points_on_segments_jax: start + t (end - start)
    point = [ad + t * vd for ad, vd in zip(a, vectors, strict=True)]
    differences = [xd - pd for xd, pd in zip(x, point, strict=True)]
    distance = _safe_sqrt(_dot(differences, differences))
    return ClosestPoints(t=t, point=jnp.stack(point, axis=-1), distance=distance)


def _coordinates(values: Array) -> list[Array]:
    """The coordinates of points, one array per axis of the last dimension."""
    return [values[..., axis] for axis in range(values.shape[-1])]


def _dot(u: list[Array], v: list[Array]) -> Array:
    """Dot product of vectors given coordinate by coordinate, summed in order."""
    return functools.reduce(
        operator.add, (ud * vd for ud, vd in zip(u, v, strict=True))
    )


def pairwise_closest_points_jax(
    points: Array, start: Array, end: Array, eps: float = DEFAULT_EPS
) -> ClosestPoints:
    """Closest points for every combination of query point and segment.

    This evaluates all ``M * N`` pairs and is intended for small problems.
    Renderers on large volumes should only evaluate nearby pairs.

    Parameters
    ----------
    points : jax.Array
        Query points, shape ``(M, D)``.
    start : jax.Array
        Segment start points, shape ``(N, D)``.
    end : jax.Array
        Segment end points, shape ``(N, D)``.
    eps : float
        Floor on squared segment lengths.

    Returns
    -------
    ClosestPoints
        ``t`` and ``distance`` of shape ``(M, N)``, and ``point`` of shape
        ``(M, N, D)``.
    """
    return closest_points_on_segments_jax(
        points[:, None, :], start[None, :, :], end[None, :, :], eps=eps
    )


def clamp(values: Array, lower: float, upper: float) -> Array:
    """``values`` clamped to ``[lower, upper]``, with torch's gradient.

    The gradient is passed where ``lower < values < upper`` and is zero
    elsewhere, including at the bounds themselves, as for ``torch.clamp``. A
    point whose projection lands exactly on a segment's end therefore pulls
    on that end only through the closest point, not through ``t``.
    """
    return jnp.where(values <= lower, lower, jnp.where(values >= upper, upper, values))


def clamp_min(values: Array, lower: float) -> Array:
    """``values`` clamped from below, with ``torch.clamp_min``'s gradient.

    The gradient is passed only where ``values > lower``.
    """
    return jnp.where(values <= lower, lower, values)


def lerp(start: Array, end: Array, weight: Array) -> Array:
    """Linear interpolation, computed as ``torch.lerp`` computes it.

    ``start + weight * (end - start)`` for ``|weight| < 0.5`` and
    ``end - (end - start) * (1 - weight)`` otherwise, which is exact at both
    ends.
    """
    difference = end - start
    return jnp.where(
        jnp.abs(weight) < 0.5,
        start + weight * difference,
        end - difference * (1 - weight),
    )
