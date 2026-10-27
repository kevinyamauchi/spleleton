"""Priors (extra loss terms) for fitting skeletons, in JAX.

See :mod:`spleleton.priors` for the units of the costs, and
:class:`spleleton.fit.JaxPrior` for how a JAX prior separates its settings
(the hashable prior object, static in the jitted fit step) from its data (a
pytree of arrays, the *state*).
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from spleleton.initialise import SegmentLayout
from spleleton.primitives._primitives_jax import JaxLineSegments
from spleleton.priors._leaves import leaf_chains

#: Room for endpoint pairs, as a multiple of the number found. During a fit
#: the number grows 2-3x as segments gather, and each growth of the table
#: compiles the fit step again; the table itself is tiny (two integers per
#: pair), so it is sized generously.
_PAIR_ROOM = 2.0


@dataclasses.dataclass(frozen=True)
class JaxFreeEndCost:
    """A cost for every leaf edge, so that small spurs are pruned.

    As :class:`spleleton.priors.TorchFreeEndCost`: each leaf edge pays

        weight * r^D * mean opacity of its segments / n_voxels

    where ``r`` is the edge's mean radius (not differentiated) and ``D`` the
    dimension.

    The state is, per segment, the index of its leaf edge (or the number of
    leaves, for a segment on no leaf edge), and the number of segments per
    leaf. It is fixed for the whole fit.

    Parameters
    ----------
    layout : SegmentLayout
        Which segments make up which edge; leaves are found from it. It is
        not part of the prior's identity (hash and equality), since it only
        determines the state.
    n_voxels : int
        Number of voxels in the target image (the BCE's normalisation).
    weight : float
        Cost of a leaf, in units of ``r^D`` voxels.
    """

    layout: SegmentLayout = dataclasses.field(compare=False, repr=False)
    n_voxels: int
    weight: float = 3.0

    @property
    def leaf_chains(self) -> list[np.ndarray]:
        """The segment indices of each leaf edge."""
        return leaf_chains(self.layout)

    def init(self, segments: JaxLineSegments) -> tuple[Array, Array]:
        """Leaf index per segment, and number of segments per leaf."""
        chains = self.leaf_chains
        leaf = np.full(len(segments), len(chains), dtype=np.int32)
        for index, chain in enumerate(chains):
            leaf[chain] = index
        sizes = np.array([len(chain) for chain in chains], dtype=np.int32)
        return jnp.asarray(leaf), jnp.asarray(sizes)

    def update(
        self, segments: JaxLineSegments, state: tuple[Array, Array], iteration: int
    ) -> tuple[Array, Array]:
        """The state does not change during a fit."""
        return state

    def __call__(
        self, segments: JaxLineSegments, state: tuple[Array, Array]
    ) -> dict[str, Array]:
        """The weighted free-end cost."""
        leaf, sizes = state
        n_leaves = sizes.shape[0]
        if n_leaves == 0 or self.weight == 0:
            return {"free_end": segments.opacity.sum() * 0.0}
        ndim = segments.start.shape[-1]
        radius = jax.lax.stop_gradient(segments.radius).mean(axis=-1)
        # segments on no leaf have index n_leaves, which segment_sum drops
        mean_radius = jax.ops.segment_sum(radius, leaf, num_segments=n_leaves) / sizes
        mean_opacity = (
            jax.ops.segment_sum(segments.opacity, leaf, num_segments=n_leaves) / sizes
        )
        cost = (mean_radius**ndim * mean_opacity).sum()
        return {"free_end": self.weight * cost / self.n_voxels}


@dataclasses.dataclass(frozen=True)
class JaxConnectionLoss:
    """Pull nearby segment endpoints together, so segments form chains.

    As :class:`spleleton.priors.TorchConnectionLoss`: every ``update_every``
    iterations, the pairs of endpoints (of different segments) closer than
    ``reach`` times the larger of their radii are found; until the next
    update, the loss is

        weight * sum over pairs of |p_a - p_b|^2 / n_voxels

    The state is the pairs, padded to a fixed number with a validity mask
    (see :func:`endpoint_pairs_jax`), with room for twice the pairs found. If
    an update finds more pairs than fit, the room grows to twice the new
    number (and the fit step compiles again).

    Parameters
    ----------
    n_voxels : int
        Number of voxels in the target image.
    weight : float
        Cost of a one-voxel gap, in voxels.
    reach : float
        Pairs closer than ``reach * max(radius)`` are pulled together.
    update_every : int
        How often (in iterations) the pairs are recomputed.
    """

    n_voxels: int
    weight: float = 1.0
    reach: float = 1.5
    update_every: int = 20

    def init(self, segments: JaxLineSegments) -> tuple[Array, Array]:
        """The pairs of the given segments, with room for more."""
        count = _count_pairs(segments, self.reach)
        size = _pair_capacity(count, len(segments))
        return endpoint_pairs_jax(segments, self.reach, size=size)

    def update(
        self, segments: JaxLineSegments, state: tuple[Array, Array], iteration: int
    ) -> tuple[Array, Array]:
        """The pairs, recomputed every ``update_every`` iterations."""
        if iteration % self.update_every:
            return state
        size = state[0].shape[0]
        count = _count_pairs(segments, self.reach)
        if count > size:
            size = _pair_capacity(count, len(segments))
        return endpoint_pairs_jax(segments, self.reach, size=size)

    def __call__(
        self, segments: JaxLineSegments, state: tuple[Array, Array]
    ) -> dict[str, Array]:
        """The weighted connection loss."""
        pairs, valid = state
        endpoints = jnp.concatenate([segments.start, segments.end])
        difference = endpoints[pairs[:, 0]] - endpoints[pairs[:, 1]]
        squared = jnp.where(valid, (difference**2).sum(axis=-1), 0.0)
        return {"connection": self.weight * squared.sum() / self.n_voxels}


def endpoint_pairs_jax(
    segments: JaxLineSegments, reach: float, size: int | None = None
) -> tuple[Array, Array]:
    """Pairs of endpoints of different segments within ``reach`` radii.

    The radius of a pair is the larger of its two endpoints' radii.
    Endpoints are numbered ``0..N-1`` for starts and ``N..2N-1`` for ends.

    Parameters
    ----------
    segments : JaxLineSegments
        The segments.
    reach : float
        Pairs closer than ``reach * max(radius)`` are kept.
    size : int, optional
        Number of pairs returned. By default, exactly the number found, which
        needs concrete segments; with a size, the function is jittable.

    Returns
    -------
    pairs : jax.Array
        Index pairs ``(a, b)`` with ``a < b``, in row-major order (as
        ``torch.nonzero`` gives them), shape ``(size, 2)``. Unused rows are
        ``(0, 0)``; with too small a size, the last pairs are missing.
    valid : jax.Array
        Which rows are pairs, shape ``(size,)``.
    """
    if size is None:
        size = _count_pairs(segments, reach)
    return _pairs_jit(segments, reach, size=size)


def _close(segments: JaxLineSegments, reach: float) -> Array:
    """Which endpoint pairs ``a < b`` are close, shape ``(2N, 2N)``."""
    segments = jax.lax.stop_gradient(segments)
    n = len(segments)
    endpoints = jnp.concatenate([segments.start, segments.end])
    radii = jnp.concatenate([segments.radius[:, 0], segments.radius[:, 1]])
    owner = jnp.concatenate([jnp.arange(n), jnp.arange(n)])
    distance = jnp.sqrt(
        ((endpoints[:, None, :] - endpoints[None, :, :]) ** 2).sum(axis=-1)
    )
    limit = reach * jnp.maximum(radii[:, None], radii[None, :])
    close = (distance < limit) & (owner[:, None] != owner[None, :])
    return jnp.triu(close, k=1)


@jax.jit
def _count_jit(segments: JaxLineSegments, reach: float) -> Array:
    return _close(segments, reach).sum()


def _count_pairs(segments: JaxLineSegments, reach: float) -> int:
    """Number of close endpoint pairs (needs concrete segments)."""
    return int(_count_jit(segments, reach))


def _pairs(segments: JaxLineSegments, reach: float, size: int) -> tuple[Array, Array]:
    close = _close(segments, reach)
    a, b = jnp.nonzero(close, size=size, fill_value=0)
    valid = jnp.arange(size) < close.sum()
    return jnp.stack([a, b], axis=-1), valid


_pairs_jit = jax.jit(_pairs, static_argnames=("size",))


def _pair_capacity(count: int, n_segments: int) -> int:
    """Room for ``count`` pairs and more, as a multiple of 8.

    Never more than the number of possible pairs of ``2 n_segments``
    endpoints (but at least 1).
    """
    room = 8 * math.ceil(max(count, 1) * _PAIR_ROOM / 8)
    n_endpoints = 2 * n_segments
    return max(1, min(room, n_endpoints * (n_endpoints - 1) // 2))
