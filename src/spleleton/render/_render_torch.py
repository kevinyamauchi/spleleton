"""Differentiable rendering of skeleton primitives into a voxel grid, in PyTorch.

See :mod:`spleleton.render` for the kernels, unions and block binning.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from spleleton.primitives._primitives_torch import TorchLineSegments
from spleleton.render._config import RenderConfig


@dataclass(frozen=True)
class TorchBlockBins:
    """Assignment of primitives to the blocks of a grid.

    Attributes
    ----------
    grid_blocks : tuple of int
        Number of blocks per axis.
    active_blocks : torch.Tensor
        Linear indices of blocks with at least one primitive, shape ``(B,)``.
    primitives : torch.Tensor
        Primitive indices per active block, shape ``(B, K)``, padded with -1.
    """

    grid_blocks: tuple[int, ...]
    active_blocks: torch.Tensor
    primitives: torch.Tensor

    @property
    def n_pairs(self) -> int:
        """Number of (block, primitive) assignments."""
        return int((self.primitives >= 0).sum())


def render_torch(
    primitives: TorchLineSegments,
    shape: tuple[int, ...],
    spacing: tuple[float, ...] | None = None,
    config: RenderConfig | None = None,
) -> torch.Tensor:
    """Render primitives into a grid with a soft union.

    Parameters
    ----------
    primitives : TorchLineSegments
        The primitives, in physical coordinates.
    shape : tuple of int
        Shape of the output grid.
    spacing : tuple of float, optional
        Physical voxel size per axis. Defaults to 1 on every axis.
    config : RenderConfig, optional
        Kernel and binning parameters.

    Returns
    -------
    torch.Tensor
        The rendered occupancy in ``[0, 1)``, of the given shape, on the
        primitives' device and with their dtype. Differentiable with respect
        to the primitives' parameters.
    """
    config = config or RenderConfig()
    ndim = len(shape)
    reference = primitives.start
    if reference.shape[-1] != ndim:
        raise ValueError(
            f"shape has {ndim} axes but the primitives are {reference.shape[-1]}D"
        )
    spacing_tensor = _spacing(spacing, ndim, reference)
    block = config.block_size
    grid_blocks = tuple(math.ceil(n / block) for n in shape)
    n_blocks = math.prod(grid_blocks)
    voxels_per_block = block**ndim

    bins = bin_primitives_torch(primitives, shape, spacing_tensor, config)
    blocks = torch.zeros(
        (n_blocks, voxels_per_block), dtype=reference.dtype, device=reference.device
    )

    if len(bins.active_blocks):
        local = _local_offsets(block, ndim, reference.device)
        grid_strides = _strides(grid_blocks, reference.device)
        n_per_batch = max(
            1, config.max_pairs // (voxels_per_block * bins.primitives.shape[1])
        )
        values = []
        for first in range(0, len(bins.active_blocks), n_per_batch):
            active = bins.active_blocks[first : first + n_per_batch]
            index = bins.primitives[first : first + n_per_batch]
            # voxel centres of each block, shape (b, voxels_per_block, D)
            corner = _unravel(active, grid_strides) * block
            points = (corner[:, None, :] + local[None]).to(
                reference.dtype
            ) * spacing_tensor
            values.append(_render_blocks(primitives, points, index, config))
        blocks = blocks.index_copy(0, bins.active_blocks, torch.cat(values))

    # (blocks per axis..., voxels per axis...) -> interleaved -> grid
    grid = blocks.reshape(*grid_blocks, *([block] * ndim))
    order = [
        axis
        for pair in zip(range(ndim), range(ndim, 2 * ndim), strict=True)
        for axis in pair
    ]
    grid = grid.permute(order).reshape([n * block for n in grid_blocks])
    return grid[tuple(slice(0, n) for n in shape)]


def render_snapshot(
    primitives: TorchLineSegments,
    shape: tuple[int, ...],
    spacing: tuple[float, ...] | None,
    config: RenderConfig,
) -> np.ndarray:
    """Render without gradients, as NumPy (for :mod:`spleleton.recording`)."""
    with torch.no_grad():
        return render_torch(primitives, shape, spacing, config).cpu().numpy()


def bin_primitives_torch(
    primitives: TorchLineSegments,
    shape: tuple[int, ...],
    spacing: torch.Tensor,
    config: RenderConfig,
) -> TorchBlockBins:
    """Assign primitives to the blocks their padded bounding boxes overlap.

    Parameters
    ----------
    primitives : TorchLineSegments
        The primitives.
    shape : tuple of int
        Shape of the grid.
    spacing : torch.Tensor
        Physical voxel size per axis, shape ``(D,)``.
    config : RenderConfig
        Kernel, cutoff and block size.

    Returns
    -------
    TorchBlockBins
        The assignment. Computed without gradients.
    """
    block = config.block_size
    ndim = len(shape)
    device = primitives.start.device
    grid_blocks = tuple(math.ceil(n / block) for n in shape)
    with torch.no_grad():
        lower, upper = primitives.bounds()
        if config.mode == "tube":
            reach = (
                primitives.radius.max(dim=-1).values + config.cutoff * config.wall_width
            )
        else:
            reach = torch.full_like(lower[:, 0], config.cutoff * config.sigma)
        # range of voxel indices inside each padded box, clipped to the grid
        last_voxel = torch.tensor(shape, device=device) - 1
        first = torch.ceil((lower - reach[:, None]) / spacing).long().clamp(min=0)
        last = torch.floor((upper + reach[:, None]) / spacing).long()
        last = torch.minimum(last, last_voxel)
        inside = torch.all(first <= last, dim=-1)
        primitive_ids = torch.nonzero(inside).squeeze(-1)
        first_block = first[inside] // block
        extent = last[inside] // block - first_block + 1

        # enumerate every (primitive, block) pair inside each primitive's range
        counts = extent.prod(dim=-1)
        pair_primitive = torch.repeat_interleave(primitive_ids, counts)
        pair_owner = torch.repeat_interleave(
            torch.arange(len(counts), device=device), counts
        )
        offsets = torch.cumsum(counts, 0) - counts
        local = torch.arange(int(counts.sum()), device=device) - offsets[pair_owner]
        multi_index = torch.empty((len(local), ndim), dtype=torch.long, device=device)
        for axis in reversed(range(ndim)):
            size = extent[pair_owner, axis]
            multi_index[:, axis] = local % size
            local = local // size
        pair_block = (
            (first_block[pair_owner] + multi_index) * _strides(grid_blocks, device)
        ).sum(dim=-1)

        # group primitives by block into a padded table
        order = torch.argsort(pair_block, stable=True)
        pair_block, pair_primitive = pair_block[order], pair_primitive[order]
        active_blocks, per_block = torch.unique_consecutive(
            pair_block, return_counts=True
        )
        width = int(per_block.max()) if len(per_block) else 0
        row = torch.repeat_interleave(
            torch.arange(len(active_blocks), device=device), per_block
        )
        column = (
            torch.arange(len(pair_block), device=device)
            - (torch.cumsum(per_block, 0) - per_block)[row]
        )
        table = torch.full(
            (len(active_blocks), width), -1, dtype=torch.long, device=device
        )
        table[row, column] = pair_primitive
    return TorchBlockBins(
        grid_blocks=grid_blocks, active_blocks=active_blocks, primitives=table
    )


def render_dense_torch(
    primitives: TorchLineSegments,
    shape: tuple[int, ...],
    spacing: tuple[float, ...] | None = None,
    config: RenderConfig | None = None,
) -> torch.Tensor:
    """Render without binning or cutoff, evaluating every voxel-primitive pair.

    This is the reference that :func:`render_torch` approximates. It needs memory
    proportional to voxels times primitives, so it is meant for tests.
    """
    config = config or RenderConfig()
    reference = primitives.start
    spacing_tensor = _spacing(spacing, len(shape), reference)
    axes = [
        torch.arange(n, device=reference.device, dtype=reference.dtype) * step
        for n, step in zip(shape, spacing_tensor, strict=True)
    ]
    points = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(
        1, -1, len(shape)
    )
    index = torch.arange(len(primitives), device=reference.device)[None]
    return _render_blocks(primitives, points, index, config).reshape(shape)


def _render_blocks(
    primitives: TorchLineSegments,
    points: torch.Tensor,
    index: torch.Tensor,
    config: RenderConfig,
) -> torch.Tensor:
    """Combine each block's primitives at each of its voxels.

    Parameters
    ----------
    primitives : TorchLineSegments
        The primitives.
    points : torch.Tensor
        Voxel centres, shape ``(b, P, D)``.
    index : torch.Tensor
        Primitive indices per block, shape ``(b, K)``, padded with -1.
    config : RenderConfig
        Kernel and union.

    Returns
    -------
    torch.Tensor
        Occupancy, shape ``(b, P)``.
    """
    valid = index >= 0
    query = primitives.query_indexed(points, index.clamp(min=0))
    if config.mode == "tube":
        profile = torch.sigmoid((query.radius - query.distance) / config.wall_width)
    elif config.mode == "centreline":
        profile = torch.exp(-0.5 * (query.distance / config.sigma) ** 2)
    else:
        raise ValueError(f"unknown render mode {config.mode!r}")
    opacity = torch.where(valid, primitives.opacity[index.clamp(min=0)], 0.0)
    contribution = profile * opacity[:, None, :]
    if config.union == "max":
        return contribution.amax(dim=-1)
    if config.union == "soft":
        # 1 - prod(1 - g), via logs so that many small contributions stay accurate
        return -torch.expm1(torch.log1p(-contribution).sum(dim=-1))
    if config.union == "softmax":
        if config.temperature <= 0:
            return contribution.amax(dim=-1)
        # only primitives within reach take part, as in the binning
        index = index.clamp(min=0)
        if config.mode == "tube":
            reach = (
                primitives.radius.max(dim=-1).values[index]
                + config.cutoff * config.wall_width
            )
        else:
            reach = torch.full(
                index.shape,
                config.cutoff * config.sigma,
                dtype=points.dtype,
                device=points.device,
            )
        in_reach = valid[:, None, :] & (query.distance <= reach[:, None, :])
        logits = torch.where(in_reach, contribution / config.temperature, -torch.inf)
        shift = logits.amax(dim=-1, keepdim=True).detach()
        shift = torch.where(torch.isfinite(shift), shift, 0.0)
        weights = torch.exp(logits - shift)
        total = weights.sum(dim=-1)
        weighted = (weights * contribution).sum(dim=-1)
        return torch.where(total > 0, weighted / total.clamp_min(1e-30), 0.0)
    raise ValueError(f"unknown union {config.union!r}")


def _spacing(
    spacing: tuple[float, ...] | None, ndim: int, like: torch.Tensor
) -> torch.Tensor:
    if spacing is None:
        return torch.ones(ndim, dtype=like.dtype, device=like.device)
    tensor = torch.as_tensor(spacing, dtype=like.dtype, device=like.device)
    if tensor.shape != (ndim,):
        raise ValueError(f"spacing must have {ndim} entries, got {tuple(tensor.shape)}")
    return tensor


def _strides(sizes: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """Row-major strides of a grid with the given sizes."""
    strides = [1]
    for size in reversed(sizes[1:]):
        strides.insert(0, strides[0] * size)
    return torch.tensor(strides, dtype=torch.long, device=device)


def _unravel(index: torch.Tensor, strides: torch.Tensor) -> torch.Tensor:
    """Multi-indices of linear indices, shape ``(n, D)``."""
    multi = []
    for stride in strides:
        multi.append(index // stride)
        index = index % stride
    return torch.stack(multi, dim=-1)


def _local_offsets(block: int, ndim: int, device: torch.device) -> torch.Tensor:
    """Voxel offsets within a block, row-major, shape ``(block**ndim, ndim)``."""
    axes = [torch.arange(block, device=device)] * ndim
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, ndim)
