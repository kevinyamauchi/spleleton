"""Differentiable skeleton primitives, in PyTorch.

See :mod:`spleleton.primitives` for the interface renderers rely on.
"""

from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch
from torch import nn

from spleleton.geometry._common import DEFAULT_EPS
from spleleton.geometry._geometry_torch import (
    closest_points_on_segments_torch,
    pairwise_closest_points_torch,
    segment_lengths_torch,
    segment_tangents_torch,
)
from spleleton.primitives._arrays import PrimitiveQuery, SegmentArrays

if TYPE_CHECKING:
    from spleleton.graph import SkeletonGraph


class TorchLineSegments(nn.Module):
    """A set of line segments in 2D or 3D with a radius profile and an opacity.

    The radius is defined at both endpoints and interpolated linearly along
    the segment. Radius and opacity are stored unconstrained (log-radius and
    opacity logit) so that gradient steps keep them valid.

    Parameters
    ----------
    start : torch.Tensor
        Segment start points, shape ``(N, D)``.
    end : torch.Tensor
        Segment end points, shape ``(N, D)``.
    radius : float or torch.Tensor
        Tube radius. A scalar, shape ``(N,)`` for a constant radius per
        segment, or shape ``(N, 2)`` for the radius at the start and end.
        Must be positive.
    opacity : float or torch.Tensor
        Opacity, a scalar or shape ``(N,)``, in the open interval ``(0, 1)``.
    """

    #: Name of the array backend.
    backend: ClassVar[str] = "torch"

    def __init__(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        radius: float | torch.Tensor = 1.0,
        opacity: float | torch.Tensor = 0.5,
    ) -> None:
        super().__init__()
        start = _as_float_tensor(start)
        end = _as_float_tensor(end, like=start)
        if start.ndim != 2 or start.shape[-1] not in (2, 3):
            raise ValueError(
                "start must have shape (N, D) with D = 2 or 3, "
                f"got {tuple(start.shape)}"
            )
        if end.shape != start.shape:
            raise ValueError(
                f"end must have the same shape as start {tuple(start.shape)}, "
                f"got {tuple(end.shape)}"
            )
        n_segments = start.shape[0]

        radius = _as_float_tensor(radius, like=start)
        if radius.ndim == 1:
            radius = radius[:, None]
        try:
            radius = radius.expand(n_segments, 2)
        except RuntimeError as error:
            raise ValueError(
                "radius must be a scalar or have shape (N,) or (N, 2), "
                f"got {tuple(radius.shape)}"
            ) from error
        if not bool((radius > 0).all()):
            raise ValueError("radius must be positive")

        opacity = _as_float_tensor(opacity, like=start)
        try:
            opacity = opacity.expand(n_segments)
        except RuntimeError as error:
            raise ValueError(
                "opacity must be a scalar or have shape (N,), "
                f"got {tuple(opacity.shape)}"
            ) from error
        if not bool(((opacity > 0) & (opacity < 1)).all()):
            raise ValueError("opacity must lie in the open interval (0, 1)")

        self.start = nn.Parameter(start.clone())
        self.end = nn.Parameter(end.clone())
        self.log_radius = nn.Parameter(torch.log(radius).clone())
        self.opacity_logit = nn.Parameter(torch.logit(opacity).clone())

    @classmethod
    def from_graph(
        cls,
        graph: "SkeletonGraph",
        radius: float = 1.0,
        opacity: float = 0.99,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> "TorchLineSegments":
        """One segment per polyline piece of every edge of a graph.

        Parameters
        ----------
        graph : SkeletonGraph
            The graph, in physical coordinates.
        radius : float
            Radius used if the graph has no radii.
        opacity : float
            Opacity of every segment.
        dtype : torch.dtype, optional
            Floating-point type. Defaults to torch's default dtype.
        device : torch.device or str, optional
            Device of the parameters.

        Returns
        -------
        TorchLineSegments
            The segments, ordered edge by edge.
        """
        return cls.from_arrays(
            SegmentArrays.from_graph(graph, radius=radius, opacity=opacity),
            dtype=dtype,
            device=device,
        )

    @classmethod
    def from_arrays(
        cls,
        arrays: SegmentArrays,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> "TorchLineSegments":
        """Segments with the given (NumPy) parameters.

        Parameters
        ----------
        arrays : SegmentArrays
            The parameters.
        dtype : torch.dtype, optional
            Floating-point type. Defaults to torch's default dtype.
        device : torch.device or str, optional
            Device of the parameters.

        Returns
        -------
        TorchLineSegments
            The segments.
        """
        dtype = dtype or torch.get_default_dtype()

        def tensor(array: np.ndarray) -> torch.Tensor:
            return torch.as_tensor(array, dtype=dtype, device=device)

        return cls(
            tensor(arrays.start),
            tensor(arrays.end),
            radius=tensor(arrays.radius),
            opacity=tensor(arrays.opacity),
        )

    def to_arrays(self) -> SegmentArrays:
        """The parameters, detached and copied to NumPy (float64)."""
        with torch.no_grad():

            def array(tensor: torch.Tensor) -> np.ndarray:
                # a copy: for a float64 tensor on the CPU, .double().numpy()
                # would share the parameter's memory
                return tensor.detach().cpu().double().numpy().copy()

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
    def radius(self) -> torch.Tensor:
        """Radius at the start and end of each segment, shape ``(N, 2)``."""
        return torch.exp(self.log_radius)

    @property
    def opacity(self) -> torch.Tensor:
        """Opacity of each segment, shape ``(N,)``."""
        return torch.sigmoid(self.opacity_logit)

    @property
    def lengths(self) -> torch.Tensor:
        """Length of each segment, shape ``(N,)``."""
        return segment_lengths_torch(self.start, self.end)

    @property
    def tangents(self) -> torch.Tensor:
        """Unit tangent of each segment, from start to end, shape ``(N, D)``."""
        return segment_tangents_torch(self.start, self.end)

    def tube_volumes(self) -> torch.Tensor:
        """Volume (area in 2D) of each segment's tube, a capsule, shape ``(N,)``.

        Uses the segment's mean radius ``r`` and length ``L``: ``2 r L + pi
        r^2`` in 2D and ``pi r^2 L + 4/3 pi r^3`` in 3D, in physical units.
        """
        r = self.radius.mean(dim=-1)
        length = self.lengths
        if self.start.shape[-1] == 2:
            return 2 * r * length + torch.pi * r**2
        return torch.pi * r**2 * length + 4 / 3 * torch.pi * r**3

    @property
    def midpoints(self) -> torch.Tensor:
        """Midpoint of each segment, shape ``(N, D)``."""
        return 0.5 * (self.start + self.end)

    def radius_at(self, t: torch.Tensor) -> torch.Tensor:
        """Radius at parameter values along each segment.

        Parameters
        ----------
        t : torch.Tensor
            Parameter values, shape ``(..., N)``. Each column refers to the
            segment with the same index.

        Returns
        -------
        torch.Tensor
            Radii, shape ``(..., N)``.
        """
        radius = self.radius
        return torch.lerp(radius[:, 0], radius[:, 1], t)

    def bounds(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Axis-aligned bounding box of each segment, without its radius.

        Returns
        -------
        lower, upper : torch.Tensor
            Corners of the boxes, each of shape ``(N, D)``.
        """
        return torch.minimum(self.start, self.end), torch.maximum(self.start, self.end)

    def query_indexed(
        self, points: torch.Tensor, index: torch.Tensor, eps: float = DEFAULT_EPS
    ) -> PrimitiveQuery:
        """Closest point, distance and radius for groups of points and segments.

        Each group of points is paired with its own list of segments, which
        is how block-binned renderers evaluate only nearby pairs.

        Parameters
        ----------
        points : torch.Tensor
            Query points, shape ``(..., P, D)``.
        index : torch.Tensor
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
        start = self.start[index].unsqueeze(-3)
        end = self.end[index].unsqueeze(-3)
        closest = closest_points_on_segments_torch(
            points.unsqueeze(-2), start, end, eps=eps
        )
        radius = self.radius[index].unsqueeze(-3)
        return PrimitiveQuery(
            t=closest.t,
            point=closest.point,
            distance=closest.distance,
            radius=torch.lerp(radius[..., 0], radius[..., 1], closest.t),
        )

    def query(self, points: torch.Tensor, eps: float = DEFAULT_EPS) -> PrimitiveQuery:
        """Closest point, distance and radius for every (point, segment) pair.

        Parameters
        ----------
        points : torch.Tensor
            Query points, shape ``(M, D)``.
        eps : float
            Floor on squared segment lengths.

        Returns
        -------
        PrimitiveQuery
            Fields of shape ``(M, N)``, and ``(M, N, D)`` for ``point``.
        """
        closest = pairwise_closest_points_torch(points, self.start, self.end, eps=eps)
        return PrimitiveQuery(
            t=closest.t,
            point=closest.point,
            distance=closest.distance,
            radius=self.radius_at(closest.t),
        )

    def extra_repr(self) -> str:
        """Summary shown by ``repr``."""
        return f"n_segments={len(self)}"


def _as_float_tensor(
    value: float | torch.Tensor, like: torch.Tensor | None = None
) -> torch.Tensor:
    """Convert a value to a floating-point tensor.

    If ``like`` is given, the result takes its dtype and device. Otherwise
    floating-point inputs keep their dtype, and anything else gets the default
    dtype.
    """
    if like is not None:
        return torch.as_tensor(value, dtype=like.dtype, device=like.device)
    tensor = torch.as_tensor(value)
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.get_default_dtype())
    return tensor
