"""Exact crops and their compiled-shape buckets (plan, Section 10.4).

The binned renderer puts a segment in every voxel of every 16^3 block its
padded box touches. So the crop of an (event, family) is the union of the
blocks that its free segments touch in any configuration (plus 1 voxel of
drift for the relaxations), and on it the library's own kernel renders the
free segments, maxed with the background: the render of every other segment
on the full volume, made once. Outside the crop nothing changes.

**Buckets.** So that most events reuse compiled programs, each crop axis
grows (by real voxels of the volume, so nothing needs masking) to 32, 48,
64, 80, 96 or 128 voxels, per axis (cubes cost up to 3x the time); the free
segments are padded to 8, 16, 24, 32, 48 or 64 slots with dead ones; the
frozen endpoints to multiples of 64; and the bin capacity to a few sizes. A
halo region (``cldice``) is the crop plus ``2h`` on each side, clipped to the
volume, so its shape follows the crop's bucket.
"""

import dataclasses
import math
import time
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from spleleton.bench.probe._setup import Reference, Setup, bin_need, block_ranges
from spleleton.bench.probe._terms import (
    FAR,
    FIELDS,
    HALO_TERMS,
    VOXEL_TERMS,
    Crop,
    Static,
    background_jit,
    halo_width,
    segments_of,
    soft_skeleton_jit,
    sparsity_sum,
)
from spleleton.render._render_jax import BinCapacity

CROP_BUCKETS = (32, 48, 64, 80, 96, 128)
SLOT_BUCKETS = (8, 16, 24, 32, 48, 64)
BLOCK_BUCKETS = (8, 16, 32, 64, 128, 256, 512)
PER_BLOCK_BUCKETS = (4, 8, 12, 16, 24, 32, 48, 64)
ENDPOINT_BUCKET = 64
#: Extra padding of the crop, in voxels, for segments moving in a relaxation.
DRIFT = 1.0
#: Headroom of measured capacities.
HEADROOM = 0.25
#: The crop's render batches (how ``lax.map`` batches, not the values).
PROBE_MAX_PAIRS = 2**18


def bucket(value: float, buckets: tuple[int, ...], cap: int) -> int:
    """The smallest bucket at least ``value`` (at most ``cap``)."""
    for size in buckets:
        if size >= value:
            return min(size, cap)
    return cap


def crop_blocks(configs, shape, spacing, render) -> tuple[np.ndarray, np.ndarray]:
    """Blocks touched by any live slot of any configuration (inclusive range)."""
    lo = np.full(len(shape), np.iinfo(int).max)
    hi = np.full(len(shape), -1)
    for c in configs:
        live = c["live"]
        first, last, inside = block_ranges(
            c["start"][live],
            c["end"][live],
            np.exp(c["log_radius"][live]),
            shape,
            spacing,
            render,
            margin=DRIFT,
        )
        if inside.any():
            lo = np.minimum(lo, first[inside].min(axis=0))
            hi = np.maximum(hi, last[inside].max(axis=0))
    return lo, hi


def place(lo, hi, volume_shape, block: int, bucketed: bool):
    """Origin (voxels) and shape of the crop covering blocks ``lo..hi``."""
    volume_blocks = np.array([math.ceil(n / block) for n in volume_shape])
    extent = hi - lo + 1
    if bucketed:
        edge = np.array(
            [bucket(e * block, CROP_BUCKETS, 10**9) // block for e in extent]
        )
        size = np.minimum(np.maximum(edge, extent), volume_blocks)
    else:
        size = extent
    origin = np.minimum(lo, volume_blocks - size)
    if not (np.all(origin <= lo) and np.all(origin + size - 1 >= hi)):
        raise RuntimeError("the crop does not cover its blocks")
    shape = tuple(
        int(min(s * block, n - o * block))
        for s, n, o in zip(size, volume_shape, origin, strict=True)
    )
    return origin * block, shape


def crop_capacity(
    configs, origin_physical, crop_shape, spacing, render, slots, bucketed
):
    """The bin capacity every configuration needs on the crop."""
    blocks = per_block = 0
    for c in configs:
        live = {k: c[k][c["live"]] for k in FIELDS}
        live["start"] = live["start"] - origin_physical
        live["end"] = live["end"] - origin_physical
        b, k = bin_need(live, crop_shape, spacing, render)
        blocks, per_block = max(blocks, b), max(per_block, k)
    n_blocks = math.prod(math.ceil(n / render.block_size) for n in crop_shape)
    if not bucketed:
        return BinCapacity.for_need(blocks, per_block, n_blocks, slots)
    return BinCapacity(
        bucket(blocks * (1 + HEADROOM), BLOCK_BUCKETS, n_blocks),
        bucket(per_block * (1 + HEADROOM), PER_BLOCK_BUCKETS, slots),
    )


@dataclass
class Background:
    """The frozen segments of one set-up, rendered on the full volume once.

    Attributes
    ----------
    occupancy : np.ndarray
        Their render.
    stats : dict of str to tuple of np.ndarray
        Per per-voxel term, its statistics of that render.
    skeleton, target_skeleton : np.ndarray, optional
        For ``cldice``: the soft skeletons of the render and of the target.
    sparsity : float
        The frozen segments' opacity times volume, in voxels, summed.
    points, radii, owners : np.ndarray
        The frozen endpoints.
    time_s : float
        How long it took.
    """

    occupancy: np.ndarray
    stats: dict[str, tuple[np.ndarray, ...]]
    skeleton: np.ndarray | None
    target_skeleton: np.ndarray | None
    sparsity: float
    points: np.ndarray
    radii: np.ndarray
    owners: np.ndarray
    time_s: float


def background(reference: Reference, setup: Setup) -> Background:
    """Render every segment but the free ones on the full volume."""
    started = time.perf_counter()
    st = reference.static
    raw = {k: v.copy() for k, v in reference.raw.items()}
    raw["start"][setup.free] = FAR
    raw["end"][setup.free] = FAR
    target = jnp.asarray(reference.target)
    occupancy, stats, overflow = background_jit(segments_of(raw), target, st)
    if bool(overflow):
        raise RuntimeError("the background's render overflowed its capacity")
    occupancy = np.asarray(occupancy)
    stats = {
        name: tuple(np.asarray(a) for a in arrays) for name, arrays in stats.items()
    }
    skeleton = target_skeleton = None
    for term in st.losses:
        if term.name in HALO_TERMS:
            k = dict(term.params).get("k", 10)
            skeleton = np.asarray(soft_skeleton_jit(jnp.asarray(occupancy), k))
            target_skeleton = np.asarray(soft_skeleton_jit(target, k))
    n = len(reference.raw["start"])
    keep = np.setdiff1d(np.arange(n), setup.free)
    segments = segments_of({k: v[keep] for k, v in reference.raw.items()})
    sparsity = float(
        sparsity_sum(segments, jnp.ones(len(keep), bool), st.render, st.voxel_volume)
    )
    points = np.concatenate([reference.raw["start"][keep], reference.raw["end"][keep]])
    radii = np.exp(
        np.concatenate(
            [reference.raw["log_radius"][keep, 0], reference.raw["log_radius"][keep, 1]]
        )
    )
    return Background(
        occupancy,
        stats,
        skeleton,
        target_skeleton,
        sparsity,
        points,
        radii,
        np.concatenate([keep, keep]),
        time.perf_counter() - started,
    )


def frozen_pairs(points, radii, owners, reach: float) -> float:
    """The frozen endpoints' close pairs among themselves (NumPy)."""
    squared = ((points[:, None] - points[None]) ** 2).sum(axis=-1)
    limit = reach * np.maximum(radii[:, None], radii[None])
    close = (np.sqrt(squared) < limit) & (owners[:, None] != owners[None])
    return float(np.where(np.triu(close, k=1), squared, 0.0).sum())


def _window(origin, shape) -> tuple[slice, ...]:
    return tuple(slice(int(o), int(o) + s) for o, s in zip(origin, shape, strict=True))


def _outside(values: np.ndarray, window) -> float:
    values = values.copy()
    values[window] = 0.0
    return float(values.sum())


def halo_region(origin, crop_shape, volume_shape, h: int):
    """Origin and shape of a crop's halo region, and its kept box (region coords)."""
    origin = np.asarray(origin)
    size = np.minimum(np.asarray(crop_shape) + 4 * h, np.asarray(volume_shape))
    region_origin = np.clip(origin - 2 * h, 0, np.asarray(volume_shape) - size)
    keep_lo = np.maximum(origin - h, 0) - region_origin
    keep_hi = (
        np.minimum(origin + np.asarray(crop_shape) + h, np.asarray(volume_shape))
        - region_origin
    )
    return region_origin, tuple(int(s) for s in size), keep_lo, keep_hi


def make_crop(
    reference: Reference,
    back: Background,
    origin,
    crop_shape,
    reach: float,
    bucketed: bool,
) -> tuple[Crop, tuple[int, ...] | None]:
    """The crop's data and constants, and its halo region's shape (or ``None``)."""
    window = _window(origin, crop_shape)
    outside = {
        name: jnp.asarray([_outside(a, window) for a in arrays])
        for name, arrays in back.stats.items()
    }
    m = len(back.points)
    size = ENDPOINT_BUCKET * math.ceil(max(m, 1) / ENDPOINT_BUCKET) if bucketed else m
    ndim = back.points.shape[1] if m else len(crop_shape)
    points = np.full((size, ndim), FAR)
    radii = np.ones(size)
    alive = np.zeros(size, bool)
    points[:m], radii[:m], alive[:m] = back.points, back.radii, True
    halo, region = None, None
    h = halo_width(reference.static.losses)
    if h:
        region_origin, region, keep_lo, keep_hi = halo_region(
            origin, crop_shape, reference.shape, h
        )
        rwindow = _window(region_origin, region)
        keep = np.zeros(region, bool)
        keep[_window(keep_lo, tuple(keep_hi - keep_lo))] = True
        kept_global = np.zeros(reference.shape, bool)
        kept_global[rwindow] = keep
        target = reference.target
        skeleton, target_skeleton = back.skeleton, back.target_skeleton
        halo = {
            "background": jnp.asarray(back.occupancy[rwindow]),
            "target": jnp.asarray(target[rwindow]),
            "keep": jnp.asarray(keep.astype(np.float64)),
            "offset": jnp.asarray(np.asarray(origin) - region_origin, dtype=jnp.int32),
            "target_skeleton": jnp.asarray(target_skeleton[window]),
            "outside": jnp.asarray(
                [
                    float((skeleton * target)[~kept_global].sum()),
                    float(skeleton[~kept_global].sum()),
                    _outside(target_skeleton * back.occupancy, window),
                    float(target_skeleton.sum()),
                ]
            ),
        }
    crop = Crop(
        origin=jnp.asarray(np.asarray(origin, float) * np.asarray(reference.spacing)),
        target=jnp.asarray(reference.target[window]),
        background=jnp.asarray(back.occupancy[window]),
        outside=outside,
        frozen_points=jnp.asarray(points),
        frozen_radii=jnp.asarray(radii),
        frozen_alive=jnp.asarray(alive),
        frozen_sparsity=jnp.asarray(back.sparsity),
        frozen_connection=jnp.asarray(
            frozen_pairs(back.points, back.radii, back.owners, reach)
        ),
        halo=halo,
    )
    return crop, region


def crop_static(reference: Reference, crop_shape, capacity, region) -> Static:
    """The settings of a crop's evaluation."""
    full = reference.static
    render = full.render
    return full._replace(
        shape=tuple(int(n) for n in crop_shape),
        render=dataclasses.replace(render, max_pairs=PROBE_MAX_PAIRS),
        capacity=capacity,
        region=region,
    )


def inside_crop(params, live, origin, crop_shape, reference: Reference) -> bool:
    """Whether every live slot's blocks still lie in the crop."""
    render = reference.static.render
    block = render.block_size
    first, last, inside = block_ranges(
        params["start"][live],
        params["end"][live],
        np.exp(params["log_radius"][live]),
        reference.shape,
        reference.spacing,
        render,
    )
    lo = np.asarray(origin) // block
    hi = (np.asarray(origin) + np.asarray(crop_shape) - 1) // block
    return bool(np.all(first[inside] >= lo) and np.all(last[inside] <= hi))


#: Names of the per-voxel terms (re-exported for checks).
PER_VOXEL = tuple(VOXEL_TERMS)
