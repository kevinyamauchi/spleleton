"""Differentiable geometry utilities for line segments, in PyTorch.

See :mod:`spleleton.geometry` for the conventions (broadcasting, coordinates).
"""

import torch

from spleleton.geometry._common import DEFAULT_EPS, ClosestPoints


def safe_norm_torch(vectors: torch.Tensor) -> torch.Tensor:
    """Euclidean norm along the last dimension, with a zero gradient at zero.

    ``torch.linalg.norm`` has an undefined (NaN) gradient at the zero vector.
    Points that lie exactly on a segment are common (e.g. voxel centres on an
    axis-aligned centreline), so the distance must stay differentiable there.

    Parameters
    ----------
    vectors : torch.Tensor
        Tensor with a trailing dimension holding the vector components.

    Returns
    -------
    torch.Tensor
        The norms, with the last dimension removed.
    """
    squared = (vectors * vectors).sum(dim=-1)
    nonzero = squared > 0
    safe_squared = torch.where(nonzero, squared, torch.ones_like(squared))
    return torch.where(nonzero, torch.sqrt(safe_squared), torch.zeros_like(squared))


def segment_lengths_torch(start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    """Lengths of segments.

    Parameters
    ----------
    start : torch.Tensor
        Segment start points, shape ``(..., D)``.
    end : torch.Tensor
        Segment end points, shape ``(..., D)``.

    Returns
    -------
    torch.Tensor
        Segment lengths, shape ``(...)``.
    """
    return safe_norm_torch(end - start)


def segment_tangents_torch(
    start: torch.Tensor, end: torch.Tensor, eps: float = DEFAULT_EPS
) -> torch.Tensor:
    """Unit tangents of segments, pointing from start to end.

    Parameters
    ----------
    start : torch.Tensor
        Segment start points, shape ``(..., D)``.
    end : torch.Tensor
        Segment end points, shape ``(..., D)``.
    eps : float
        Floor on the segment length. Degenerate (zero-length) segments get a
        zero tangent instead of NaN.

    Returns
    -------
    torch.Tensor
        Unit tangents, shape ``(..., D)``.
    """
    vectors = end - start
    lengths = safe_norm_torch(vectors).clamp_min(eps)
    return vectors / lengths.unsqueeze(-1)


def points_on_segments_torch(
    start: torch.Tensor, end: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """Evaluate segments at parameter values.

    Parameters
    ----------
    start : torch.Tensor
        Segment start points, shape ``(..., D)``.
    end : torch.Tensor
        Segment end points, shape ``(..., D)``.
    t : torch.Tensor
        Parameter values, shape ``(...)``. ``0`` gives the start and ``1`` the
        end. Values outside ``[0, 1]`` extrapolate along the line.

    Returns
    -------
    torch.Tensor
        Points, shape ``(..., D)``.
    """
    return start + t.unsqueeze(-1) * (end - start)


def closest_points_on_segments_torch(
    points: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
    eps: float = DEFAULT_EPS,
) -> ClosestPoints:
    """Find the closest point on each segment to each query point.

    The closest point is the orthogonal projection onto the segment's line,
    clamped to the segment. When the projection is clamped, the gradient with
    respect to ``t`` is zero, and gradients reach the endpoints through the
    closest point itself.

    Parameters
    ----------
    points : torch.Tensor
        Query points, shape ``(..., D)``.
    start : torch.Tensor
        Segment start points, shape ``(..., D)``, broadcastable with
        ``points``.
    end : torch.Tensor
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
    vectors = end - start
    squared_lengths = (vectors * vectors).sum(dim=-1)
    projections = ((points - start) * vectors).sum(dim=-1)
    t = (projections / squared_lengths.clamp_min(eps)).clamp(0.0, 1.0)
    point = points_on_segments_torch(start, end, t)
    distance = safe_norm_torch(points - point)
    return ClosestPoints(t=t, point=point, distance=distance)


def pairwise_closest_points_torch(
    points: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
    eps: float = DEFAULT_EPS,
) -> ClosestPoints:
    """Closest points for every combination of query point and segment.

    This evaluates all ``M * N`` pairs and is intended for small problems.
    Renderers on large volumes should only evaluate nearby pairs.

    Parameters
    ----------
    points : torch.Tensor
        Query points, shape ``(M, D)``.
    start : torch.Tensor
        Segment start points, shape ``(N, D)``.
    end : torch.Tensor
        Segment end points, shape ``(N, D)``.
    eps : float
        Floor on squared segment lengths.

    Returns
    -------
    ClosestPoints
        ``t`` and ``distance`` of shape ``(M, N)``, and ``point`` of shape
        ``(M, N, D)``.
    """
    return closest_points_on_segments_torch(
        points[:, None, :], start[None, :, :], end[None, :, :], eps=eps
    )
