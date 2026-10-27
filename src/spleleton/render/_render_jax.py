"""Differentiable rendering of skeleton primitives into a voxel grid, in JAX.

See :mod:`spleleton.render` for the kernels, unions and block binning.

Under ``jit`` every shape must be known when a function is traced, so the
binning uses fixed capacities (:class:`BinCapacity`): at most ``blocks``
active blocks, each with at most ``per_block`` primitives. The binning
reports whether they were enough (``JaxBlockBins.overflow``) and what would
have been (``JaxBlockBins.needed``); a caller that sees an overflow grows the
capacity and renders again, so no result is ever computed from a truncated
table. :func:`render_jax` measures the capacity itself when none is given.

The blocks are evaluated in batches by ``lax.map``. When there is more than
one batch, each batch's kernel is rematerialised (``jax.checkpoint``): the
backward pass recomputes its per-pair values instead of saving them for every
batch, which would take memory proportional to all pairs (gigabytes at 128^3)
and, on a CPU, more time than the recomputation. A single batch saves them,
since they are then bounded by ``max_pairs`` and recomputing would only cost
time.

The grid shape, the spacing, the :class:`RenderConfig` and the capacity are
static (a new value compiles again). The softmax temperature is traced, so a
temperature schedule compiles once; the ``temperature`` field of the config
is ignored in favour of the ``temperature`` argument.
"""

import dataclasses
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from spleleton.geometry._geometry_jax import clamp_min
from spleleton.primitives._primitives_jax import JaxLineSegments
from spleleton.render._config import RenderConfig

#: Extra room given to measured capacities, so that small changes during a fit
#: do not overflow them.
HEADROOM = 0.25


class BinCapacity(NamedTuple):
    """Fixed sizes of the block table.

    Attributes
    ----------
    blocks : int
        Number of active blocks evaluated (B).
    per_block : int
        Number of primitives per block (K).
    """

    blocks: int
    per_block: int

    @classmethod
    def for_need(
        cls,
        blocks: int,
        per_block: int,
        n_blocks: int,
        n_primitives: int,
        headroom: float = HEADROOM,
    ) -> "BinCapacity":
        """A capacity for a measured need, with headroom.

        Each size is the need plus ``headroom``, rounded up to a multiple of
        8. No more blocks than the grid has, and no more primitives per block
        than there are primitives, are ever needed, so the sizes are capped
        there.

        Parameters
        ----------
        blocks, per_block : int
            The needed sizes, e.g. from :attr:`JaxBlockBins.needed`.
        n_blocks : int
            Number of blocks in the grid.
        n_primitives : int
            Number of primitives.
        headroom : float
            Fraction added to each size.

        Returns
        -------
        BinCapacity
            The capacity, with both sizes at least 1.
        """

        def size(need: int, cap: int) -> int:
            padded = 8 * math.ceil(max(need, 1) * (1 + headroom) / 8)
            return max(1, min(padded, cap))

        return cls(size(blocks, n_blocks), size(per_block, n_primitives))


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class JaxBlockBins:
    """Assignment of primitives to the blocks of a grid, with fixed sizes.

    Attributes
    ----------
    grid_blocks : tuple of int
        Number of blocks per axis (static).
    active_blocks : jax.Array
        Linear indices of the active blocks, shape ``(B,)``. Unused entries
        hold the number of blocks (one past the last).
    primitives : jax.Array
        Primitive indices per active block, in increasing order, shape
        ``(B, K)``, padded with -1.
    overflow : jax.Array
        Whether the capacity was too small (a boolean scalar). If so, the
        table is truncated.
    needed : jax.Array
        The capacity that would have been enough, ``(blocks, per_block)``.
    """

    grid_blocks: tuple[int, ...] = dataclasses.field(metadata={"static": True})
    active_blocks: Array
    primitives: Array
    overflow: Array
    needed: Array

    @property
    def n_pairs(self) -> int:
        """Number of (block, primitive) assignments. Not jittable."""
        return int((self.primitives >= 0).sum())

    @property
    def needed_capacity(self) -> BinCapacity:
        """:attr:`needed` as a :class:`BinCapacity`, without headroom. Not jittable."""
        blocks, per_block = (int(n) for n in np.asarray(self.needed))
        return BinCapacity(blocks, per_block)


def render_jax(
    primitives: JaxLineSegments,
    shape: tuple[int, ...],
    spacing: tuple[float, ...] | None = None,
    config: RenderConfig | None = None,
    capacity: BinCapacity | None = None,
    temperature: float | Array | None = None,
) -> Array:
    """Render primitives into a grid with a soft union.

    Parameters
    ----------
    primitives : JaxLineSegments
        The primitives, in physical coordinates.
    shape : tuple of int
        Shape of the output grid.
    spacing : tuple of float, optional
        Physical voxel size per axis. Defaults to 1 on every axis.
    config : RenderConfig, optional
        Kernel and binning parameters.
    capacity : BinCapacity or tuple of int, optional
        Sizes of the block table. By default they are measured from the
        primitives, which therefore must not be traced: inside ``jit``,
        ``grad`` or another transformation, pass a capacity (e.g. from
        :func:`bin_capacity_jax`, measured beforehand) and check
        :func:`bin_primitives_jax` for overflow.
    temperature : float or jax.Array, optional
        Temperature of the ``"softmax"`` union. Defaults to
        ``config.temperature``. It may be traced; changing it does not
        compile again.

    Returns
    -------
    jax.Array
        The rendered occupancy in ``[0, 1)``, of the given shape and with the
        primitives' dtype. Differentiable with respect to the primitives'
        parameters. If ``capacity`` is too small, primitives are missing from
        some blocks.
    """
    config = config or RenderConfig()
    shape, spacing = _check(primitives, shape, spacing)
    if capacity is None:
        capacity = bin_capacity_jax(primitives, shape, spacing, config)
    capacity = BinCapacity(*capacity)
    temperature = config.temperature if temperature is None else temperature
    image, _ = _render_jit(
        primitives,
        _temperature(temperature, primitives),
        shape=shape,
        spacing=spacing,
        config=_static(config),
        capacity=capacity,
    )
    return image


def render_snapshot(
    primitives: JaxLineSegments,
    shape: tuple[int, ...],
    spacing: tuple[float, ...] | None,
    config: RenderConfig,
) -> np.ndarray:
    """Render as NumPy (for :mod:`spleleton.recording`).

    Snapshots of one fit share a capacity, kept per number of primitives,
    grid and settings: it only grows, when a snapshot overflows it, so the
    render compiles once per fit rather than once per measured capacity.
    """
    shape, spacing = _check(primitives, shape, spacing)
    static = _static(config)
    key = (len(primitives), primitives.start.dtype, shape, spacing, static)
    capacity = _snapshot_capacities.get(key) or bin_capacity_jax(
        primitives, shape, spacing, config
    )
    n_blocks = math.prod(_grid_blocks(shape, config.block_size))
    temperature = _temperature(config.temperature, primitives)
    while True:
        image, bins = _render_jit(
            primitives,
            temperature,
            shape=shape,
            spacing=spacing,
            config=static,
            capacity=capacity,
        )
        if not bool(bins.overflow):
            break
        blocks, per_block = (int(n) for n in np.asarray(bins.needed))
        capacity = BinCapacity.for_need(
            max(blocks, capacity.blocks),
            max(per_block, capacity.per_block),
            n_blocks,
            len(primitives),
        )
    if len(_snapshot_capacities) >= _MAX_SNAPSHOT_KEYS:
        _snapshot_capacities.clear()
    _snapshot_capacities[key] = capacity
    return np.asarray(image)


#: Capacities of recent snapshot renders, by (N, dtype, shape, spacing, config).
_snapshot_capacities: dict[tuple, BinCapacity] = {}
_MAX_SNAPSHOT_KEYS = 64


def bin_primitives_jax(
    primitives: JaxLineSegments,
    shape: tuple[int, ...],
    spacing: tuple[float, ...] | None = None,
    config: RenderConfig | None = None,
    capacity: BinCapacity | None = None,
) -> JaxBlockBins:
    """Assign primitives to the blocks their padded bounding boxes overlap.

    Parameters
    ----------
    primitives : JaxLineSegments
        The primitives.
    shape : tuple of int
        Shape of the grid.
    spacing : tuple of float, optional
        Physical voxel size per axis. Defaults to 1 on every axis.
    config : RenderConfig, optional
        Kernel, cutoff and block size.
    capacity : BinCapacity or tuple of int, optional
        Sizes of the table. By default they are measured (with headroom),
        which needs concrete primitives.

    Returns
    -------
    JaxBlockBins
        The assignment. Computed without gradients. Jittable for a given
        capacity.
    """
    config = config or RenderConfig()
    shape, spacing = _check(primitives, shape, spacing)
    if capacity is None:
        capacity = bin_capacity_jax(primitives, shape, spacing, config)
    capacity = BinCapacity(*capacity)
    return _bin_jit(
        primitives,
        shape=shape,
        spacing=spacing,
        config=_static(config),
        capacity=capacity,
    )


def bin_capacity_jax(
    primitives: JaxLineSegments,
    shape: tuple[int, ...],
    spacing: tuple[float, ...] | None = None,
    config: RenderConfig | None = None,
    headroom: float = HEADROOM,
) -> BinCapacity:
    """Measure the block table needed for some primitives, plus headroom.

    Parameters
    ----------
    primitives : JaxLineSegments
        The primitives. They must be concrete (not traced inside ``jit``,
        ``grad`` or another transformation).
    shape : tuple of int
        Shape of the grid.
    spacing : tuple of float, optional
        Physical voxel size per axis. Defaults to 1 on every axis.
    config : RenderConfig, optional
        Kernel, cutoff and block size.
    headroom : float
        Fraction added to each size (see :meth:`BinCapacity.for_need`).

    Returns
    -------
    BinCapacity
        The capacity.
    """
    config = config or RenderConfig()
    shape, spacing = _check(primitives, shape, spacing)
    if isinstance(primitives.start, jax.core.Tracer):
        # a TypeError, as JAX's own errors for traced values are
        raise TypeError(
            "the bin capacity cannot be measured from traced primitives; "
            "inside jit or grad, pass a capacity (e.g. from bin_capacity_jax)"
        )
    grid_blocks = _grid_blocks(shape, config.block_size)
    n_blocks = math.prod(grid_blocks)
    if len(primitives) == 0:
        return BinCapacity.for_need(0, 0, n_blocks, 0, headroom)
    counts = _count_jit(
        primitives, shape=shape, spacing=spacing, config=_static(config)
    )
    blocks, per_block = (int(n) for n in np.asarray(counts))
    return BinCapacity.for_need(blocks, per_block, n_blocks, len(primitives), headroom)


def render_dense_jax(
    primitives: JaxLineSegments,
    shape: tuple[int, ...],
    spacing: tuple[float, ...] | None = None,
    config: RenderConfig | None = None,
    temperature: float | Array | None = None,
) -> Array:
    """Render without binning or cutoff, evaluating every voxel-primitive pair.

    This is the reference that :func:`render_jax` approximates. It needs memory
    proportional to voxels times primitives, so it is meant for tests.
    """
    config = config or RenderConfig()
    shape, spacing = _check(primitives, shape, spacing)
    temperature = config.temperature if temperature is None else temperature
    return _dense_jit(
        primitives,
        _temperature(temperature, primitives),
        shape=shape,
        spacing=spacing,
        config=_static(config),
    )


def _render_binned(
    primitives: JaxLineSegments,
    temperature: Array,
    shape: tuple[int, ...],
    spacing: tuple[float, ...],
    config: RenderConfig,
    capacity: BinCapacity,
    width: Array | None = None,
) -> tuple[Array, JaxBlockBins]:
    """The binned render and its bins (traceable; the fit step calls this).

    ``width``, if given, replaces the kernel's width (``wall_width`` in tube
    mode, ``sigma`` in centreline mode) in the profile, as a traced value, so
    that the render can be differentiated with respect to it; the binning
    still uses the config's.
    """
    ndim = len(shape)
    dtype = primitives.start.dtype
    block = config.block_size
    grid_blocks = _grid_blocks(shape, block)
    n_blocks = math.prod(grid_blocks)
    voxels_per_block = block**ndim

    bins = _bin(primitives, shape, spacing, config, capacity)
    blocks = jnp.zeros((n_blocks, voxels_per_block), dtype=dtype)
    if len(primitives):
        local = jnp.asarray(_local_offsets(block, ndim))
        strides = jnp.asarray(_strides(grid_blocks))
        sizes = jnp.asarray(grid_blocks)
        spacing_array = jnp.asarray(spacing, dtype=dtype)
        n_per_batch = max(
            1, config.max_pairs // (voxels_per_block * capacity.per_block)
        )
        batch_size = min(n_per_batch, capacity.blocks)
        kernel = _render_blocks
        if batch_size < capacity.blocks:
            # recomputed in the backward pass rather than saved per batch
            kernel = jax.checkpoint(_render_blocks, static_argnums=(3,))

        def render_block(item: tuple[Array, Array]) -> Array:
            active, index = item
            # voxel centres of the block, shape (voxels_per_block, D)
            corner = (active // strides) % sizes * block
            points = (corner + local).astype(dtype) * spacing_array
            return kernel(primitives, points, index, config, temperature, width)

        values = jax.lax.map(
            render_block, (bins.active_blocks, bins.primitives), batch_size=batch_size
        )
        # unused slots point one past the last block and are dropped
        blocks = blocks.at[bins.active_blocks].set(values, mode="drop")

    # (blocks per axis..., voxels per axis...) -> interleaved -> grid
    grid = blocks.reshape(*grid_blocks, *([block] * ndim))
    order = [
        axis
        for pair in zip(range(ndim), range(ndim, 2 * ndim), strict=True)
        for axis in pair
    ]
    grid = grid.transpose(order).reshape([n * block for n in grid_blocks])
    return grid[tuple(slice(0, n) for n in shape)], bins


def _bin(
    primitives: JaxLineSegments,
    shape: tuple[int, ...],
    spacing: tuple[float, ...],
    config: RenderConfig,
    capacity: BinCapacity,
) -> JaxBlockBins:
    """Fixed-size block table (traceable)."""
    grid_blocks = _grid_blocks(shape, config.block_size)
    n_blocks = math.prod(grid_blocks)
    n_primitives = len(primitives)
    overlap = _overlap(primitives, shape, spacing, config)
    counts = overlap.sum(axis=0)
    needed = jnp.stack([(counts > 0).sum(), counts.max()]).astype(jnp.int32)
    overflow = (needed[0] > capacity.blocks) | (needed[1] > capacity.per_block)

    # the fullest blocks; empty ones among them are unused slots
    filled, chosen = jax.lax.top_k(counts, capacity.blocks)
    active = jnp.where(filled > 0, chosen, n_blocks).astype(jnp.int32)

    # compact each chosen block's row of the overlap into its first K slots
    rows = overlap.T[chosen]
    slot = jnp.cumsum(rows, axis=1) - 1
    slot = jnp.where(rows & (slot < capacity.per_block), slot, capacity.per_block)
    table = jnp.full((capacity.blocks, capacity.per_block), -1, dtype=jnp.int32)
    table = table.at[jnp.arange(capacity.blocks)[:, None], slot].set(
        jnp.broadcast_to(
            jnp.arange(n_primitives, dtype=jnp.int32), (capacity.blocks, n_primitives)
        ),
        mode="drop",
    )
    return JaxBlockBins(
        grid_blocks=grid_blocks,
        active_blocks=active,
        primitives=table,
        overflow=overflow,
        needed=needed,
    )


def _count(
    primitives: JaxLineSegments,
    shape: tuple[int, ...],
    spacing: tuple[float, ...],
    config: RenderConfig,
) -> Array:
    """Active blocks and the most primitives in one block (traceable)."""
    counts = _overlap(primitives, shape, spacing, config).sum(axis=0)
    return jnp.stack([(counts > 0).sum(), counts.max()])


def _overlap(
    primitives: JaxLineSegments,
    shape: tuple[int, ...],
    spacing: tuple[float, ...],
    config: RenderConfig,
) -> Array:
    """Whether each primitive's padded box overlaps each block, ``(N, n_blocks)``."""
    block = config.block_size
    grid_blocks = _grid_blocks(shape, block)
    primitives = jax.lax.stop_gradient(primitives)
    spacing_array = jnp.asarray(spacing, dtype=primitives.start.dtype)
    lower, upper = primitives.bounds()
    if config.mode == "tube":
        reach = primitives.radius.max(axis=-1) + config.cutoff * config.wall_width
    else:
        reach = jnp.full_like(lower[:, 0], config.cutoff * config.sigma)
    # range of voxel indices inside each padded box, clipped to the grid; the
    # clipping happens before the cast to an integer so that far-away
    # primitives cannot overflow it (and stays outside the grid when torch's
    # unclipped bound would)
    size = jnp.asarray(shape, dtype=lower.dtype)
    first = jnp.clip(jnp.ceil((lower - reach[:, None]) / spacing_array), 0, size)
    last = jnp.clip(jnp.floor((upper + reach[:, None]) / spacing_array), -1, size - 1)
    first, last = first.astype(jnp.int32), last.astype(jnp.int32)
    inside = jnp.all(first <= last, axis=-1)
    # floor division, so that last = -1 maps to block -1
    first_block, last_block = first // block, last // block
    block_index = jnp.asarray(
        np.stack(np.unravel_index(np.arange(math.prod(grid_blocks)), grid_blocks), -1)
    )
    within = (first_block[:, None, :] <= block_index[None]) & (
        block_index[None] <= last_block[:, None, :]
    )
    return inside[:, None] & jnp.all(within, axis=-1)


def _render_dense(
    primitives: JaxLineSegments,
    temperature: Array,
    shape: tuple[int, ...],
    spacing: tuple[float, ...],
    config: RenderConfig,
) -> Array:
    dtype = primitives.start.dtype
    if len(primitives) == 0:
        return jnp.zeros(shape, dtype=dtype)
    axes = [
        jnp.arange(n, dtype=dtype) * jnp.asarray(step, dtype=dtype)
        for n, step in zip(shape, spacing, strict=True)
    ]
    points = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1).reshape(
        -1, len(shape)
    )
    index = jnp.arange(len(primitives))
    return _render_blocks(primitives, points, index, config, temperature).reshape(shape)


def _render_blocks(
    primitives: JaxLineSegments,
    points: Array,
    index: Array,
    config: RenderConfig,
    temperature: Array,
    width: Array | None = None,
) -> Array:
    """Combine each block's primitives at each of its voxels.

    Parameters
    ----------
    primitives : JaxLineSegments
        The primitives.
    points : jax.Array
        Voxel centres, shape ``(..., P, D)``.
    index : jax.Array
        Primitive indices per block, shape ``(..., K)``, padded with -1.
    config : RenderConfig
        Kernel and union (its temperature is ignored).
    temperature : jax.Array
        Temperature of the ``"softmax"`` union.
    width : jax.Array, optional
        The kernel's width in the profile (see :func:`_render_binned`).

    Returns
    -------
    jax.Array
        Occupancy, shape ``(..., P)``.
    """
    valid = index >= 0
    index = jnp.maximum(index, 0)
    query = primitives.query_indexed(points, index)
    if config.mode == "tube":
        wall = config.wall_width if width is None else width
        profile = jax.nn.sigmoid((query.radius - query.distance) / wall)
    elif config.mode == "centreline":
        sigma = config.sigma if width is None else width
        profile = jnp.exp(-0.5 * (query.distance / sigma) ** 2)
    else:
        raise ValueError(f"unknown render mode {config.mode!r}")
    opacity = jnp.where(valid, primitives.opacity[index], 0.0)
    contribution = profile * opacity[..., None, :]
    if config.union == "max":
        return contribution.max(axis=-1)
    if config.union == "soft":
        # 1 - prod(1 - g), via logs so that many small contributions stay accurate
        return -jnp.expm1(jnp.log1p(-contribution).sum(axis=-1))
    if config.union == "softmax":
        # a temperature of 0 gives the max; the softmax is then evaluated at a
        # harmless temperature and discarded, so it cannot spoil the gradient
        positive = temperature > 0
        safe_temperature = jnp.where(positive, temperature, 1.0)
        # only primitives within reach take part, as in the binning
        if config.mode == "tube":
            reach = (
                primitives.radius.max(axis=-1)[index]
                + config.cutoff * config.wall_width
            )
        else:
            reach = jnp.full(index.shape, config.cutoff * config.sigma, points.dtype)
        in_reach = valid[..., None, :] & (query.distance <= reach[..., None, :])
        logits = jnp.where(in_reach, contribution / safe_temperature, -jnp.inf)
        shift = jax.lax.stop_gradient(logits.max(axis=-1, keepdims=True))
        shift = jnp.where(jnp.isfinite(shift), shift, 0.0)
        weights = jnp.exp(logits - shift)
        total = weights.sum(axis=-1)
        weighted = (weights * contribution).sum(axis=-1)
        softmax = jnp.where(total > 0, weighted / clamp_min(total, 1e-30), 0.0)
        return jnp.where(positive, softmax, contribution.max(axis=-1))
    raise ValueError(f"unknown union {config.union!r}")


_STATIC = ("shape", "spacing", "config", "capacity")
_render_jit = jax.jit(_render_binned, static_argnames=_STATIC)
_bin_jit = jax.jit(_bin, static_argnames=_STATIC)
_count_jit = jax.jit(_count, static_argnames=_STATIC[:3])
_dense_jit = jax.jit(_render_dense, static_argnames=_STATIC[:3])


def _check(
    primitives: JaxLineSegments,
    shape: tuple[int, ...],
    spacing: tuple[float, ...] | None,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Shape and spacing as hashable tuples, checked against the primitives."""
    shape = tuple(int(n) for n in shape)
    ndim = len(shape)
    if primitives.start.shape[-1] != ndim:
        raise ValueError(
            f"shape has {ndim} axes but the primitives are "
            f"{primitives.start.shape[-1]}D"
        )
    if spacing is None:
        return shape, (1.0,) * ndim
    spacing = tuple(float(step) for step in np.asarray(spacing).reshape(-1))
    if len(spacing) != ndim:
        raise ValueError(f"spacing must have {ndim} entries, got {len(spacing)}")
    return shape, spacing


def _static(config: RenderConfig) -> RenderConfig:
    """The config with its temperature fixed, so that it never recompiles."""
    return dataclasses.replace(config, temperature=0.0)


def _temperature(value: float | Array, primitives: JaxLineSegments) -> Array:
    return jnp.asarray(value, dtype=primitives.start.dtype)


def _grid_blocks(shape: tuple[int, ...], block: int) -> tuple[int, ...]:
    return tuple(math.ceil(n / block) for n in shape)


def _strides(sizes: tuple[int, ...]) -> np.ndarray:
    """Row-major strides of a grid with the given sizes."""
    strides = [1]
    for size in reversed(sizes[1:]):
        strides.insert(0, strides[0] * size)
    return np.array(strides)


def _local_offsets(block: int, ndim: int) -> np.ndarray:
    """Voxel offsets within a block, row-major, shape ``(block**ndim, ndim)``."""
    axes = [np.arange(block)] * ndim
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, ndim)
