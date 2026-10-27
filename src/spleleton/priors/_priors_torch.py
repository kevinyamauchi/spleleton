"""Priors (extra loss terms) for fitting skeletons, in PyTorch.

See :mod:`spleleton.priors` for the units of the costs.
"""

import torch

from spleleton.initialise import SegmentLayout
from spleleton.primitives._primitives_torch import TorchLineSegments
from spleleton.priors._leaves import leaf_chains


class TorchFreeEndCost:
    """A cost for every leaf edge, so that small spurs are pruned.

    A bump on a tube is part of the mask, so the spur covering it genuinely
    lowers the rendering loss; only a prior removes it. Each leaf edge (an
    edge with a degree-1 node) pays

        weight * r^D * mean opacity of its segments / n_voxels

    where ``r`` is the edge's mean radius (not differentiated) and ``D`` the
    dimension: a leaf survives only if it explains more than about
    ``weight * r^D`` voxels. Scaling with ``r^D`` makes the threshold relative
    to the tube's size. A bump covers roughly 1-3 r^2 voxels in 2D, a true leaf
    at least 4 radii long about 5 r^2 or more beyond its parent's surface.

    Parameters
    ----------
    layout : SegmentLayout
        Which segments make up which edge; leaves are found from it.
    n_voxels : int
        Number of voxels in the target image (the BCE's normalisation).
    weight : float
        Cost of a leaf, in units of ``r^D`` voxels.
    """

    def __init__(
        self, layout: SegmentLayout, n_voxels: int, weight: float = 3.0
    ) -> None:
        self.weight = weight
        self.n_voxels = n_voxels
        self.leaf_chains = [
            torch.as_tensor(chain, dtype=torch.long) for chain in leaf_chains(layout)
        ]

    def __call__(
        self, segments: TorchLineSegments, iteration: int
    ) -> dict[str, torch.Tensor]:
        """The weighted free-end cost."""
        if not self.leaf_chains or self.weight == 0:
            return {"free_end": segments.opacity.sum() * 0.0}
        ndim = segments.start.shape[-1]
        opacity = segments.opacity
        radius = segments.radius.detach().mean(dim=-1)
        cost = sum(
            radius[chain].mean() ** ndim * opacity[chain].mean()
            for chain in self.leaf_chains
        )
        return {"free_end": self.weight * cost / self.n_voxels}


class TorchConnectionLoss:
    """Pull nearby segment endpoints together, so segments form chains.

    Every ``update_every`` iterations, the pairs of endpoints (of different
    segments) closer than ``reach`` times the larger of their radii are
    found; until the next update, the loss is

        weight * sum over pairs of |p_a - p_b|^2 / n_voxels

    i.e. a gap of one voxel between two endpoints costs ``weight`` voxels.

    Parameters
    ----------
    weight : float
        Cost of a one-voxel gap, in voxels.
    n_voxels : int
        Number of voxels in the target image.
    reach : float
        Pairs closer than ``reach * max(radius)`` are pulled together. The
        larger radius matters at junctions: the endpoints of thin children
        spread over about the parent's radius, and with the smaller radius
        noisy junctions fall apart into separate components.
    update_every : int
        How often (in iterations) the pairs are recomputed.
    """

    def __init__(
        self,
        n_voxels: int,
        weight: float = 1.0,
        reach: float = 1.5,
        update_every: int = 20,
    ) -> None:
        self.weight = weight
        self.n_voxels = n_voxels
        self.reach = reach
        self.update_every = update_every
        self._pairs: torch.Tensor | None = None

    def __call__(
        self, segments: TorchLineSegments, iteration: int
    ) -> dict[str, torch.Tensor]:
        """The weighted connection loss."""
        endpoints = torch.cat([segments.start, segments.end])
        if self._pairs is None or iteration % self.update_every == 0:
            self._pairs = endpoint_pairs_torch(segments, self.reach)
        if len(self._pairs) == 0:
            return {"connection": endpoints.sum() * 0.0}
        a, b = self._pairs.T
        squared = ((endpoints[a] - endpoints[b]) ** 2).sum(dim=-1)
        return {"connection": self.weight * squared.sum() / self.n_voxels}


def endpoint_pairs_torch(segments: TorchLineSegments, reach: float) -> torch.Tensor:
    """Pairs of endpoints of different segments within ``reach`` radii.

    The radius of a pair is the larger of its two endpoints' radii.

    Endpoints are numbered ``0..N-1`` for starts and ``N..2N-1`` for ends.

    Returns
    -------
    torch.Tensor
        Index pairs ``(a, b)`` with ``a < b``, shape ``(P, 2)``.
    """
    with torch.no_grad():
        n = len(segments)
        endpoints = torch.cat([segments.start, segments.end])
        radii = torch.cat([segments.radius[:, 0], segments.radius[:, 1]])
        owner = torch.cat([torch.arange(n), torch.arange(n)]).to(endpoints.device)
        distance = torch.cdist(endpoints, endpoints)
        limit = reach * torch.maximum(radii[:, None], radii[None, :])
        close = (distance < limit) & (owner[:, None] != owner[None, :])
        close = torch.triu(close, diagonal=1)
        return torch.nonzero(close)
