"""Loss terms on the full volume and on exact crops (JAX, float64).

**The full volume** (:func:`full_terms`) renders every segment with the
library's binned renderer and computes each loss term with the library's own
function (:data:`spleleton.fit._fit_jax.LOSSES_JAX`), so it is the fit's loss
(with the connection pairs recomputed for every configuration, plan Section
10.4). It relaxes the reference and checks the crops.

**A crop** (:func:`crop_terms`) renders only the free segments, on the 16^3
blocks their padded boxes touch, and maxes the result with the rest of the
volume's render (the *background*). Under the max union that is the full
render on the crop, and outside the crop nothing changes, so every term is
its value on the crop combined with constants from outside it:

- a term that sums per-voxel statistics (:data:`VOXEL_TERMS`: ``bce``,
  ``dice``, ``weighted_l2``, ``focal``, ``tversky``) adds the crop's sums to
  the sums outside it, then finishes as the library does (Dice and Tversky
  keep their global sums, so the same error moves them less in a larger
  tree);
- ``sparsity`` adds the free segments' opacity times volume to the frozen
  segments';
- ``cldice`` is not per-voxel: its soft skeleton at a voxel depends on the
  occupancy up to ``h = k + 2`` voxels away. Its precision is computed on a
  *region* that extends the crop by ``2h`` (clipped to the volume), and only
  the voxels within ``h`` of the crop are taken from it; beyond them the
  skeleton is the background's. Its sensitivity is per-voxel;
- the connection prior adds the free endpoints' pairs (among themselves and
  with the frozen endpoints) to the frozen pairs.

The fit does not differentiate the sparsity term's volumes. ``fit_gradient``
selects that convention (the force the fit feels) or differentiates them too
(the derivative of the margin itself).
"""

import math
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from spleleton.bench.probe._spec import CONNECTION
from spleleton.fit._fit_jax import (
    LOSSES_JAX,
    LossContext,
    _optimizer,
    _Settings,
    kernel_volumes_jax,
    soft_skeleton_jax,
)
from spleleton.fit._losses import LossTerm
from spleleton.geometry._geometry_jax import clamp
from spleleton.primitives._primitives_jax import JaxLineSegments
from spleleton.render._config import RenderConfig
from spleleton.render._render_jax import BinCapacity, _render_binned

#: Dead slots sit here: outside every grid, so never binned or rendered.
FAR = -1.0e4
DEAD_LOGIT = -30.0
FIELDS = ("start", "end", "log_radius", "opacity_logit")


class Static(NamedTuple):
    """Static settings of an evaluation (a new value compiles again).

    Attributes
    ----------
    shape : tuple of int
        The grid rendered on: the crop's, or the full volume's.
    spacing : tuple of float
        Physical voxel size.
    render : RenderConfig
        The renderer (max union).
    capacity : BinCapacity
        The bin capacity of the render.
    n_voxels : int
        The full volume's number of voxels (every term's normalisation).
    losses : tuple of LossTerm
        The loss terms.
    reach : float
        The connection prior's reach, in radii.
    weights : tuple of float
        The weight of each term (losses, then the connection prior).
    lrs : tuple of float
        Adam's learning rates: positions, log-radii, opacity logits.
    region : tuple of int, optional
        The shape of the halo region of a non-local term (``cldice``).
    """

    shape: tuple[int, ...]
    spacing: tuple[float, ...]
    render: RenderConfig
    capacity: BinCapacity
    n_voxels: int
    losses: tuple[LossTerm, ...]
    reach: float
    weights: tuple[float, ...]
    lrs: tuple[float, float, float]
    region: tuple[int, ...] | None = None

    @property
    def names(self) -> tuple[str, ...]:
        """The terms' names (losses, then the connection prior)."""
        return (*(term.name for term in self.losses), CONNECTION)

    @property
    def voxel_volume(self) -> float:
        """Physical volume of a voxel."""
        return float(np.prod(self.spacing))


class Crop(NamedTuple):
    """What a crop's terms need besides the free segments (traced).

    Attributes
    ----------
    origin : jax.Array
        Physical position of the crop's first voxel, ``(D,)``.
    target : jax.Array
        The target on the crop.
    background : jax.Array
        The render of the frozen segments on the crop.
    outside : dict of str to jax.Array
        Per per-voxel term, its statistics summed outside the crop.
    frozen_points, frozen_radii, frozen_alive : jax.Array
        The frozen segments' endpoints (padded; ``frozen_alive`` is false for
        padding).
    frozen_sparsity : jax.Array
        The frozen segments' opacity times volume, in voxels, summed.
    frozen_connection : jax.Array
        The frozen endpoints' pairs among themselves, summed.
    halo : dict of str to jax.Array, optional
        For ``cldice``: ``background`` and ``target`` on the region, ``keep``
        (the region's voxels within ``h`` of the crop), ``offset`` (the
        crop's first voxel in the region), ``target_skeleton`` on the crop,
        and ``outside``, the sums from beyond the region's kept voxels.
    """

    origin: Array
    target: Array
    background: Array
    outside: dict[str, Array]
    frozen_points: Array
    frozen_radii: Array
    frozen_alive: Array
    frozen_sparsity: Array
    frozen_connection: Array
    halo: dict[str, Array] | None = None


# -- per-voxel terms -------------------------------------------------------------------


def _voxel_bce(occupancy, target, eps=1e-6):
    """Per-voxel BCE, clamped as the library's ``bce`` (same gradient rule)."""
    occupancy = clamp(occupancy, eps, 1 - eps)
    return -(target * jnp.log(occupancy) + (1 - target) * jnp.log1p(-occupancy))


def _bce_stats(occupancy, target, eps=1e-6):
    return (_voxel_bce(occupancy, target, eps),)


def _mean_finish(sums, n_voxels, **params):
    return sums[0] / n_voxels


def _dice_stats(occupancy, target, eps=1e-6):
    return occupancy * target, occupancy, target


def _dice_finish(sums, n_voxels, eps=1e-6):
    overlap, occupied, target = sums
    return 1 - (2 * overlap + eps) / (occupied + target + eps)


def _weighted_l2_stats(occupancy, target, pos_weight=10.0):
    return ((1 + (pos_weight - 1) * target) * (occupancy - target) ** 2,)


def _focal_stats(occupancy, target, gamma=2.0, eps=1e-6):
    occupancy = clamp(occupancy, eps, 1 - eps)
    return (
        -(
            target * (1 - occupancy) ** gamma * jnp.log(occupancy)
            + (1 - target) * occupancy**gamma * jnp.log1p(-occupancy)
        ),
    )


def _tversky_stats(occupancy, target, alpha=0.3, beta=0.7, eps=1e-6):
    return occupancy * target, occupancy * (1 - target), (1 - occupancy) * target


def _tversky_finish(sums, n_voxels, alpha=0.3, beta=0.7, eps=1e-6):
    overlap, false_positive, false_negative = sums
    return 1 - (overlap + eps) / (
        overlap + alpha * false_positive + beta * false_negative + eps
    )


class VoxelTerm(NamedTuple):
    """A term that is a function of sums of per-voxel statistics.

    ``stats(occupancy, target, **params)`` returns per-voxel arrays;
    ``finish(sums, n_voxels, **params)`` returns the term from their sums.
    """

    stats: Callable[..., tuple[Array, ...]]
    finish: Callable[..., Array]


#: Per-voxel terms, by the name of the library's loss term.
VOXEL_TERMS: dict[str, VoxelTerm] = {
    "bce": VoxelTerm(_bce_stats, _mean_finish),
    "dice": VoxelTerm(_dice_stats, _dice_finish),
    "weighted_l2": VoxelTerm(_weighted_l2_stats, _mean_finish),
    "focal": VoxelTerm(_focal_stats, _mean_finish),
    "tversky": VoxelTerm(_tversky_stats, _tversky_finish),
}
#: Terms with a halo, and its width in voxels from their parameters.
HALO_TERMS: dict[str, Callable[..., int]] = {"cldice": lambda k=10, eps=1e-6: k + 2}
#: Terms computed from the segments rather than the render.
SEGMENT_TERMS = ("sparsity",)


def check_losses(losses: tuple[LossTerm, ...]) -> None:
    """Refuse loss terms the probe cannot compute on a crop."""
    known = {*VOXEL_TERMS, *HALO_TERMS, *SEGMENT_TERMS}
    unknown = [term.name for term in losses if term.name not in known]
    if unknown:
        raise ValueError(
            f"the probe has no crop form of the loss terms {unknown}; "
            f"known: {sorted(known)}"
        )
    if sum(term.name in HALO_TERMS for term in losses) > 1:
        raise ValueError("the probe supports one term with a halo at most")


def halo_width(losses: tuple[LossTerm, ...]) -> int:
    """The halo of the loss's non-local term, in voxels (0 if none)."""
    for term in losses:
        if term.name in HALO_TERMS:
            return HALO_TERMS[term.name](**dict(term.params))
    return 0


# -- the connection prior and sparsity -------------------------------------------------


def endpoints(segments: JaxLineSegments) -> tuple[Array, Array]:
    """Every start then every end, and their radii."""
    points = jnp.concatenate([segments.start, segments.end])
    radii = jnp.concatenate([segments.radius[:, 0], segments.radius[:, 1]])
    return points, radii


def pair_sum(a, b, reach: float, same: bool = False) -> Array:
    """Sum of squared distances of the close endpoint pairs between two sets.

    Each set is ``(points, radii, owner, alive)``. Pairs are selected as
    :func:`spleleton.priors.endpoint_pairs_jax` selects them (closer than
    ``reach`` times the larger radius, of different segments), but for every
    configuration. With ``same``, the sets are one and each pair counts once.
    """
    points_a, radii_a, owner_a, alive_a = a
    points_b, radii_b, owner_b, alive_b = b
    squared = ((points_a[:, None, :] - points_b[None, :, :]) ** 2).sum(axis=-1)
    distance = jnp.sqrt(jax.lax.stop_gradient(squared))
    limit = reach * jnp.maximum(
        jax.lax.stop_gradient(radii_a)[:, None], jax.lax.stop_gradient(radii_b)[None]
    )
    close = (
        (distance < limit)
        & (owner_a[:, None] != owner_b[None, :])
        & alive_a[:, None]
        & alive_b[None, :]
    )
    if same:
        close = jnp.triu(close, k=1)
    return jnp.where(close, squared, 0.0).sum()


def _endpoint_set(segments: JaxLineSegments, live: Array):
    n = len(segments)
    points, radii = endpoints(segments)
    owner = jnp.concatenate([jnp.arange(n), jnp.arange(n)])
    return points, radii, owner, jnp.concatenate([live, live])


def sparsity_sum(segments, live, render, voxel_volume, fit_gradient=True) -> Array:
    """Opacity times rendered volume, in voxels, summed over live segments."""
    shape_source = jax.lax.stop_gradient(segments) if fit_gradient else segments
    volumes = kernel_volumes_jax(shape_source, render) / voxel_volume
    return jnp.where(live, segments.opacity * volumes, 0.0).sum()


def _with_total(terms: dict[str, Array], st: Static) -> dict[str, Array]:
    total = sum(w * terms[name] for w, name in zip(st.weights, st.names, strict=True))
    return {**terms, "total": total}


# -- the full volume -------------------------------------------------------------------


def full_terms(segments, live, target, st: Static, fit_gradient=True):
    """The loss terms (unweighted) and their weighted total on the full volume.

    Returns the terms and the render's overflow flag.
    """
    dtype = segments.start.dtype
    occupancy, bins = _render_binned(
        segments, jnp.zeros((), dtype), st.shape, st.spacing, st.render, st.capacity
    )
    shape_source = jax.lax.stop_gradient(segments) if fit_gradient else segments
    volumes = kernel_volumes_jax(shape_source, st.render) / st.voxel_volume
    volumes = jnp.where(live, volumes, 0.0)
    context = LossContext(occupancy, target, segments, volumes)
    terms = {
        term.name: LOSSES_JAX[term.name](context, **dict(term.params))
        for term in st.losses
    }
    own = _endpoint_set(segments, live)
    terms[CONNECTION] = pair_sum(own, own, st.reach, same=True) / st.n_voxels
    return _with_total(terms, st), bins.overflow


def optimizer(st: Static) -> optax.GradientTransformation:
    """Adam per parameter group, as the fit's."""
    position, radius, opacity = st.lrs
    return _optimizer(_Settings(position, radius, opacity, (), 1.0))


def _masked(grad, frozen):
    return jax.tree.map(lambda g, m: jnp.where(m, 0.0, g), grad, frozen)


def _full_step_impl(segments, state, live, frozen, target, st):
    """One Adam step on the full volume; the terms and gradient before it."""

    def loss(params):
        terms, overflow = full_terms(params, live, target, st)
        return terms["total"], (terms, overflow)

    (_, (terms, overflow)), grad = jax.value_and_grad(loss, has_aux=True)(segments)
    updates, state = optimizer(st).update(_masked(grad, frozen), state, segments)
    return optax.apply_updates(segments, updates), state, terms, grad, overflow


full_step = jax.jit(_full_step_impl, static_argnames=("st",))
full_eval = jax.jit(full_terms, static_argnames=("st", "fit_gradient"))


def _background_impl(segments, target, st):
    occupancy, bins = _render_binned(
        segments,
        jnp.zeros((), segments.start.dtype),
        st.shape,
        st.spacing,
        st.render,
        st.capacity,
    )
    stats = {
        term.name: VOXEL_TERMS[term.name].stats(occupancy, target, **dict(term.params))
        for term in st.losses
        if term.name in VOXEL_TERMS
    }
    return occupancy, stats, bins.overflow


background_jit = jax.jit(_background_impl, static_argnames=("st",))
soft_skeleton_jit = jax.jit(soft_skeleton_jax, static_argnames=("k",))


# -- crops -----------------------------------------------------------------------------


def crop_render(free, crop: Crop, st: Static) -> tuple[Array, Array]:
    """The occupancy on the crop (free segments maxed with the background)."""
    dtype = free.start.dtype
    shifted = JaxLineSegments(
        start=free.start - crop.origin,
        end=free.end - crop.origin,
        log_radius=free.log_radius,
        opacity_logit=free.opacity_logit,
    )
    rendered, bins = _render_binned(
        shifted, jnp.zeros((), dtype), st.shape, st.spacing, st.render, st.capacity
    )
    return jnp.maximum(crop.background, rendered), bins.overflow


def crop_terms_at(occupancy, free, live, crop: Crop, st: Static, fit_gradient=True):
    """The terms from a crop's occupancy and the free segments, plus constants."""
    terms: dict[str, Array] = {}
    for term in st.losses:
        params = dict(term.params)
        if term.name in VOXEL_TERMS:
            voxel = VOXEL_TERMS[term.name]
            stats = voxel.stats(occupancy, crop.target, **params)
            sums = tuple(
                outside + value.sum()
                for outside, value in zip(crop.outside[term.name], stats, strict=True)
            )
            terms[term.name] = voxel.finish(sums, st.n_voxels, **params)
        elif term.name == "sparsity":
            free_part = sparsity_sum(
                free, live, st.render, st.voxel_volume, fit_gradient
            )
            terms[term.name] = (crop.frozen_sparsity + free_part) / st.n_voxels
        elif term.name == "cldice":
            terms[term.name] = _cldice_crop(occupancy, crop, **params)
        else:
            raise ValueError(f"no crop form of the loss term {term.name!r}")
    own = _endpoint_set(free, live)
    frozen = (
        crop.frozen_points,
        crop.frozen_radii,
        jnp.full(crop.frozen_radii.shape, -1),
        crop.frozen_alive,
    )
    pairs = pair_sum(own, own, st.reach, same=True) + pair_sum(own, frozen, st.reach)
    terms[CONNECTION] = (crop.frozen_connection + pairs) / st.n_voxels
    return _with_total(terms, st)


def crop_terms(free, live, crop: Crop, st: Static, fit_gradient=True):
    """The terms of a configuration of the free segments on a crop.

    Returns the terms (unweighted, plus ``total``) and ``overflow``.
    """
    occupancy, overflow = crop_render(free, crop, st)
    return {
        **crop_terms_at(occupancy, free, live, crop, st, fit_gradient),
        "overflow": overflow,
    }


def _cldice_crop(occupancy, crop: Crop, k=10, eps=1e-6) -> Array:
    """ClDice from the crop, its halo region and the constants beyond it."""
    halo = crop.halo
    region = jax.lax.dynamic_update_slice(
        halo["background"], occupancy, tuple(halo["offset"])
    )
    skeleton = soft_skeleton_jax(region, k)
    keep = halo["keep"]
    outside = halo["outside"]
    overlap = outside[0] + (skeleton * halo["target"] * keep).sum()
    skeleton_sum = outside[1] + (skeleton * keep).sum()
    precision = (overlap + eps) / (skeleton_sum + eps)
    covered = outside[2] + (halo["target_skeleton"] * occupancy).sum()
    sensitivity = (covered + eps) / (outside[3] + eps)
    return 1 - 2 * precision * sensitivity / (precision + sensitivity)


def _floats(terms: dict[str, Array]) -> dict[str, Array]:
    return {name: value for name, value in terms.items() if name != "overflow"}


def _terms_batch_impl(free, live, crop, st):
    return jax.vmap(lambda f, alive: crop_terms(f, alive, crop, st))(free, live)


def _forces_batch_impl(free, live, tangent, crop, st):
    """Terms and their derivatives along ``tangent`` (forward mode).

    Returns the terms, the force the fit feels (its gradient convention) and
    the derivative of the terms themselves (the volumes differentiated too).
    """

    def one(f, alive, t):
        values, force = jax.jvp(
            lambda x: _floats(crop_terms(x, alive, crop, st)), (f,), (t,)
        )
        _, derivative = jax.jvp(
            lambda x: _floats(crop_terms(x, alive, crop, st, fit_gradient=False)),
            (f,),
            (t,),
        )
        return values, force, derivative

    return jax.vmap(one)(free, live, tangent)


def _relax_batch_impl(free, live, frozen, crop, st, n_iterations):
    """Adam on the free parameters, ``frozen`` held; the terms at every step.

    Returns per row the terms before each step, the parameters after it, and
    the terms after the last step.
    """
    adam = optimizer(st)

    def one(f, alive):
        def loss(params):
            terms = crop_terms(params, alive, crop, st)
            return terms["total"], terms

        def body(carry, _):
            params, state = carry
            (_, terms), grad = jax.value_and_grad(loss, has_aux=True)(params)
            updates, state = adam.update(_masked(grad, frozen), state, params)
            params = optax.apply_updates(params, updates)
            return (params, state), (terms, params)

        (params, _), (history, path) = jax.lax.scan(
            body, (f, adam.init(f)), None, length=n_iterations
        )
        return history, path, crop_terms(params, alive, crop, st)

    return jax.vmap(one)(free, live)


def _attribution_impl(free, live, crop, st):
    """The total's gradient with respect to the crop's occupancy."""
    occupancy, _ = crop_render(free, crop, st)
    return jax.grad(lambda o: crop_terms_at(o, free, live, crop, st)["total"])(
        occupancy
    )


terms_batch = jax.jit(_terms_batch_impl, static_argnames=("st",))
forces_batch = jax.jit(_forces_batch_impl, static_argnames=("st",))
relax_batch = jax.jit(_relax_batch_impl, static_argnames=("st", "n_iterations"))
attribution = jax.jit(_attribution_impl, static_argnames=("st",))


def _segment_gradients_impl(free, live, crop, st):
    """Each term's gradient with respect to the free segments."""
    jacobian = jax.jacrev(lambda f: _floats(crop_terms(f, live, crop, st)))
    return jacobian(free)


segment_gradients = jax.jit(_segment_gradients_impl, static_argnames=("st",))


# -- host-side helpers -----------------------------------------------------------------


def segments_of(raw: dict[str, np.ndarray]) -> JaxLineSegments:
    """Segments from raw float64 parameters."""
    return JaxLineSegments(
        **{k: jnp.asarray(np.asarray(raw[k], dtype=np.float64)) for k in FIELDS}
    )


def raw_of(segments: JaxLineSegments) -> dict[str, np.ndarray]:
    """Raw float64 parameters of segments."""
    return {k: np.asarray(getattr(segments, k), dtype=np.float64) for k in FIELDS}


def stack(configs: list[dict[str, Any]]) -> tuple[JaxLineSegments, Array]:
    """A batch of configurations: segments with a leading axis, and live masks."""
    segments = JaxLineSegments(
        **{k: jnp.asarray(np.stack([c[k] for c in configs])) for k in FIELDS}
    )
    return segments, jnp.asarray(np.stack([c["live"] for c in configs]))


def mask_tree(frozen: dict[str, np.ndarray]) -> JaxLineSegments:
    """Boolean masks as a segments pytree."""
    return JaxLineSegments(**{k: jnp.asarray(frozen[k]) for k in FIELDS})


def n_blocks(shape: tuple[int, ...], block: int) -> int:
    """Number of blocks of a grid."""
    return math.prod(math.ceil(n / block) for n in shape)
