"""Fit primitives to a target image by differentiable rendering, in JAX.

See :mod:`spleleton.fit` for the loss.

The fit is a Python loop around one jitted step (loss, gradient and Adam
update), so that callbacks and the per-iteration history work as in
:func:`spleleton.fit.fit_torch`. JAX is functional: :func:`fit_jax` returns
new segments instead of updating them in place.

What the step compiles for (a new value compiles again):

- the shapes of the segments, the target and the priors' states;
- the spacing, the loss weights and learning rates, and the renderer's
  settings, except its temperature, which is traced: a temperature schedule
  compiles once, and switching the union to the max (``switch_to_max``)
  compiles once more;
- the renderer's bin capacity. If a step overflows it, the step's result is
  discarded, the capacity grows and the same iteration is redone, so no
  iteration is ever computed with a truncated block table.
"""

from collections.abc import Callable, Sequence
from typing import Any, NamedTuple, Protocol

import jax
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from spleleton.fit._config import FitConfig
from spleleton.fit._losses import LossTerm
from spleleton.geometry._geometry_jax import clamp
from spleleton.primitives._primitives_jax import JaxLineSegments
from spleleton.render._config import RenderConfig
from spleleton.render._render_jax import (
    BinCapacity,
    _check,
    _grid_blocks,
    _render_binned,
    _static,
    bin_capacity_jax,
)

PyTree = Any


class JaxPrior(Protocol):
    """An extra loss term for :func:`fit_jax`.

    A JAX prior separates its settings from its data. The prior object holds
    only settings and must be hashable (e.g. a frozen dataclass), because the
    jitted step takes it as a static argument. Its data, which may change
    during a fit, is a pytree of arrays (the *state*), traced by the step.
    """

    def init(self, segments: JaxLineSegments) -> PyTree:
        """The state for a fit of these segments."""
        ...

    def update(
        self, segments: JaxLineSegments, state: PyTree, iteration: int
    ) -> PyTree:
        """The state for an iteration (called on the host before each step)."""
        ...

    def __call__(self, segments: JaxLineSegments, state: PyTree) -> dict[str, Array]:
        """The weighted terms (called inside the jitted step)."""
        ...


def bce_loss_jax(occupancy: Array, target: Array, eps: float = 1e-6) -> Array:
    """Mean binary cross-entropy, with the occupancy clamped away from 0 and 1."""
    occupancy = clamp(occupancy, eps, 1 - eps)
    return -(target * jnp.log(occupancy) + (1 - target) * jnp.log1p(-occupancy)).mean()


def dice_loss_jax(occupancy: Array, target: Array, eps: float = 1e-6) -> Array:
    """One minus the soft Dice coefficient."""
    overlap = (occupancy * target).sum()
    return 1 - (2 * overlap + eps) / (occupancy.sum() + target.sum() + eps)


class LossContext(NamedTuple):
    """What a loss term is computed from.

    Attributes
    ----------
    occupancy : jax.Array
        The rendering.
    target : jax.Array
        The target field.
    segments : JaxLineSegments
        The segments.
    volumes : jax.Array
        Each segment's rendered volume in voxels, not differentiated: a
        capsule for the ``"tube"`` kernel, a Gaussian line for
        ``"centreline"``.
    """

    occupancy: Array
    target: Array
    segments: JaxLineSegments
    volumes: Array


#: ``name -> function(context, **params) -> scalar``, see :func:`register_loss_jax`.
LOSSES_JAX: dict[str, Callable[..., Array]] = {}


def register_loss_jax(name: str) -> Callable[[Callable], Callable]:
    """Register a loss term's JAX function under a name (a decorator).

    The function is called inside the jitted step as ``function(context,
    **params)``, with a :class:`LossContext` and the term's parameters, and
    returns an unweighted scalar.
    """

    def register(function: Callable) -> Callable:
        if name in LOSSES_JAX:
            raise ValueError(f"loss {name!r} is already registered")
        LOSSES_JAX[name] = function
        return function

    return register


@register_loss_jax("bce")
def _bce(context: LossContext, eps: float = 1e-6) -> Array:
    return bce_loss_jax(context.occupancy, context.target, eps)


@register_loss_jax("dice")
def _dice(context: LossContext, eps: float = 1e-6) -> Array:
    return dice_loss_jax(context.occupancy, context.target, eps)


@register_loss_jax("sparsity")
def _sparsity(context: LossContext) -> Array:
    return (context.segments.opacity * context.volumes).sum() / context.target.size


@register_loss_jax("weighted_l2")
def _weighted_l2(context: LossContext, pos_weight: float = 10.0) -> Array:
    """Mean squared error; voxels are weighted by ``1 + (pos_weight - 1) t``."""
    target = context.target
    weight = 1 + (pos_weight - 1) * target
    return (weight * (context.occupancy - target) ** 2).mean()


@register_loss_jax("focal")
def _focal(context: LossContext, gamma: float = 2.0, eps: float = 1e-6) -> Array:
    """Focal loss: BCE with each voxel weighted by ``(1 - p_t)^gamma``."""
    occupancy = clamp(context.occupancy, eps, 1 - eps)
    target = context.target
    return -(
        target * (1 - occupancy) ** gamma * jnp.log(occupancy)
        + (1 - target) * occupancy**gamma * jnp.log1p(-occupancy)
    ).mean()


@register_loss_jax("tversky")
def _tversky(
    context: LossContext, alpha: float = 0.3, beta: float = 0.7, eps: float = 1e-6
) -> Array:
    """One minus the Tversky index (``alpha``: false positives, ``beta``: misses)."""
    occupancy, target = context.occupancy, context.target
    overlap = (occupancy * target).sum()
    false_positive = (occupancy * (1 - target)).sum()
    false_negative = ((1 - occupancy) * target).sum()
    return 1 - (overlap + eps) / (
        overlap + alpha * false_positive + beta * false_negative + eps
    )


@register_loss_jax("cldice")
def _cldice(context: LossContext, k: int = 10, eps: float = 1e-6) -> Array:
    """One minus clDice (Shit et al., 2021) of the rendering and the target.

    Topology precision is the share of the rendering's soft skeleton inside
    the target, topology sensitivity the share of the target's soft skeleton
    inside the rendering; clDice is their harmonic mean.
    """
    occupancy, target = context.occupancy, context.target
    skeleton = soft_skeleton_jax(occupancy, k)
    target_skeleton = jax.lax.stop_gradient(soft_skeleton_jax(target, k))
    precision = ((skeleton * target).sum() + eps) / (skeleton.sum() + eps)
    sensitivity = ((target_skeleton * occupancy).sum() + eps) / (
        target_skeleton.sum() + eps
    )
    return 1 - 2 * precision * sensitivity / (precision + sensitivity)


def soft_skeleton_jax(image: Array, k: int = 10) -> Array:
    """The soft skeleton of Shit et al. (2021), by iterated min and max pooling.

    Erosion is the minimum over 3-voxel windows along each axis, dilation the
    maximum over a 3^D window, and opening an erosion then a dilation. Each
    of ``k`` iterations erodes the image and adds what opening removes.
    """

    def pool(x: Array, window: tuple[int, ...]) -> Array:
        return jax.lax.reduce_window(
            x, -jnp.inf, jax.lax.max, window, (1,) * x.ndim, "SAME"
        )

    ndim = image.ndim
    axes = [tuple(3 if a == b else 1 for b in range(ndim)) for a in range(ndim)]

    def erode(x: Array) -> Array:
        return jnp.min(jnp.stack([-pool(-x, window) for window in axes]), axis=0)

    def open_(x: Array) -> Array:
        return pool(erode(x), (3,) * ndim)

    skeleton = jax.nn.relu(image - open_(image))
    for _ in range(k):
        image = erode(image)
        delta = jax.nn.relu(image - open_(image))
        skeleton = skeleton + jax.nn.relu(delta - skeleton * delta)
    return skeleton


def kernel_volumes_jax(segments: JaxLineSegments, render_config: RenderConfig) -> Array:
    """Each segment's rendered volume at opacity 1, in physical units.

    A capsule (:meth:`JaxLineSegments.tube_volumes`) for the ``"tube"``
    kernel. For ``"centreline"``, the integral of the Gaussian line: per
    unit length ``2 pi sigma^2`` in 3D (``sqrt(2 pi) sigma`` in 2D), plus
    one full Gaussian for the two half-Gaussian caps.
    """
    if render_config.mode == "tube":
        return segments.tube_volumes()
    sigma = render_config.sigma
    if segments.start.shape[-1] == 2:
        return np.sqrt(2 * np.pi) * sigma * segments.lengths + 2 * np.pi * sigma**2
    return 2 * np.pi * sigma**2 * segments.lengths + (2 * np.pi) ** 1.5 * sigma**3


class FitState(NamedTuple):
    """What a fit's last iteration depended on, besides the segments.

    With these, :func:`loss_terms_jax` reproduces the fit's last loss
    (``history[-1]``, at :attr:`pre_last`) and the loss at the final segments
    bit for bit, in any process, because it runs the fit's own compiled step.

    Attributes
    ----------
    pre_last : JaxLineSegments or None
        The segments before the last Adam step, which ``history[-1]`` belongs
        to. ``None`` if the fit ran no iteration.
    prior_states : tuple
        Each regulariser's state at the last iteration (e.g. the connection
        loss's pair table).
    capacity : BinCapacity
        The renderer's bin capacity of the last step.
    render_config : RenderConfig
        The renderer's settings at the last iteration (its union and
        temperature).
    """

    pre_last: JaxLineSegments | None
    prior_states: tuple[PyTree, ...]
    capacity: BinCapacity
    render_config: RenderConfig


def fit_jax(
    segments: JaxLineSegments,
    target: Array | np.ndarray,
    spacing: tuple[float, ...] | None = None,
    config: FitConfig | None = None,
    callback: Callable[[int, JaxLineSegments], None] | None = None,
    regularisers: Sequence[JaxPrior] = (),
    capacity: BinCapacity | None = None,
    device: jax.Device | None = None,
    return_state: bool = False,
) -> (
    tuple[JaxLineSegments, list[dict[str, float]]]
    | tuple[JaxLineSegments, list[dict[str, float]], FitState]
):
    """Optimise primitives so that their rendering matches a target.

    Parameters
    ----------
    segments : JaxLineSegments
        Initial primitives.
    target : array_like
        Target image in ``[0, 1]`` (e.g. a binary mask), NumPy or JAX.
    spacing : tuple of float, optional
        Physical voxel size per axis.
    config : FitConfig, optional
        Optimiser, loss and schedule settings.
    callback : callable, optional
        Called as ``callback(iteration, segments)`` before every step and
        once after the last one (with ``iteration == n_iterations``).
    regularisers : sequence of JaxPrior
        Extra loss terms (see :class:`JaxPrior`), e.g. the priors in
        :mod:`spleleton.priors`. Their terms are already weighted.
    capacity : BinCapacity, optional
        Initial bin capacity of the renderer. By default it is measured from
        the initial segments, with headroom. It grows when a step overflows.
    device : jax.Device, optional
        Device of the fit. Defaults to JAX's default device.
    return_state : bool
        Also return the :class:`FitState` of the last iteration.

    Returns
    -------
    segments : JaxLineSegments
        The optimised primitives.
    history : list of dict
        Per iteration: the total loss and each of its terms.
    state : FitState
        Only with ``return_state``: what the last iteration depended on.
    """
    config = config or FitConfig()
    if device is not None:
        segments = jax.device_put(segments, device)
    dtype = segments.start.dtype
    target = jnp.asarray(target, dtype=dtype)
    if device is not None:
        target = jax.device_put(target, device)
    shape, spacing = _check(segments, target.shape, spacing)
    settings = _settings(config, spacing)
    regularisers = tuple(regularisers)
    n_blocks = int(np.prod(_grid_blocks(shape, config.render.block_size)))
    if capacity is None:
        capacity = bin_capacity_jax(segments, shape, spacing, config.render_config(0))
    capacity = BinCapacity(*capacity)

    opt_state = _optimizer(settings).init(segments)
    states = tuple(prior.init(segments) for prior in regularisers)
    history = []
    pre_last = None
    for iteration in range(config.n_iterations):
        if callback is not None:
            callback(iteration, segments)
        states = tuple(
            prior.update(segments, state, iteration)
            for prior, state in zip(regularisers, states, strict=True)
        )
        render_config = config.render_config(iteration)
        temperature = jnp.asarray(render_config.temperature, dtype=dtype)
        while True:
            result = _step(
                segments,
                opt_state,
                states,
                target,
                temperature,
                shape=shape,
                spacing=spacing,
                render_config=_static(render_config),
                capacity=capacity,
                settings=settings,
                regularisers=regularisers,
            )
            terms, overflow, needed = jax.device_get(result[2:])
            if not overflow:
                break
            # the step is pure, so the state before it is still at hand
            capacity = BinCapacity.for_need(
                max(int(needed[0]), capacity.blocks),
                max(int(needed[1]), capacity.per_block),
                n_blocks,
                len(segments),
            )
        pre_last = segments
        segments, opt_state = result[:2]
        history.append({name: float(value) for name, value in terms.items()})
    if callback is not None:
        callback(config.n_iterations, segments)
    if not return_state:
        return segments, history
    last = config.render_config(max(config.n_iterations - 1, 0))
    return segments, history, FitState(pre_last, states, capacity, last)


def loss_terms_jax(
    segments: JaxLineSegments,
    target: Array | np.ndarray,
    state: FitState,
    spacing: tuple[float, ...] | None = None,
    config: FitConfig | None = None,
    regularisers: Sequence[JaxPrior] = (),
) -> tuple[dict[str, float], bool]:
    """The loss terms of a fit's last step, at other segments.

    This runs the fit's own step with the state of its last iteration and
    discards the update, so it compiles nothing new in the fitting process,
    and its terms equal the fit's to the bit: at ``state.pre_last`` they are
    ``history[-1]``.

    Parameters
    ----------
    segments : JaxLineSegments
        Where to evaluate the loss.
    target : array_like
        The fit's target.
    state : FitState
        The fit's state (from ``fit_jax(..., return_state=True)``).
    spacing : tuple of float, optional
        The fit's spacing.
    config : FitConfig, optional
        The fit's settings.
    regularisers : sequence of JaxPrior
        The fit's regularisers (their settings; the states are in ``state``).

    Returns
    -------
    terms : dict of str to float
        The total loss and each term, as in the fit's history.
    overflow : bool
        Whether the bin capacity overflowed (then the terms are not exact).
    """
    config = config or FitConfig()
    dtype = segments.start.dtype
    target = jnp.asarray(target, dtype=dtype)
    shape, spacing = _check(segments, target.shape, spacing)
    settings = _settings(config, spacing)
    temperature = jnp.asarray(state.render_config.temperature, dtype=dtype)
    result = _step(
        segments,
        _optimizer(settings).init(segments),
        tuple(state.prior_states),
        target,
        temperature,
        shape=shape,
        spacing=spacing,
        render_config=_static(state.render_config),
        capacity=BinCapacity(*state.capacity),
        settings=settings,
        regularisers=tuple(regularisers),
    )
    terms, overflow, _ = jax.device_get(result[2:])
    return {name: float(value) for name, value in terms.items()}, bool(overflow)


def loss_gradient_jax(
    segments: JaxLineSegments,
    target: Array | np.ndarray,
    state: FitState,
    spacing: tuple[float, ...] | None = None,
    config: FitConfig | None = None,
    regularisers: Sequence[JaxPrior] = (),
) -> tuple[JaxLineSegments, bool]:
    """The gradient of a fit's last-step loss, at other segments.

    As :func:`loss_terms_jax`, with the state of the fit's last iteration,
    but returning the gradient the fit's step would take there (with its
    convention: the sparsity term's volumes are not differentiated).

    Returns
    -------
    gradient : JaxLineSegments
        The gradient with respect to every parameter.
    overflow : bool
        Whether the bin capacity overflowed (then it is not exact).
    """
    config = config or FitConfig()
    dtype = segments.start.dtype
    target = jnp.asarray(target, dtype=dtype)
    shape, spacing = _check(segments, target.shape, spacing)
    grads, overflow = _gradient(
        segments,
        tuple(state.prior_states),
        target,
        jnp.asarray(state.render_config.temperature, dtype=dtype),
        shape=shape,
        spacing=spacing,
        render_config=_static(state.render_config),
        capacity=BinCapacity(*state.capacity),
        settings=_settings(config, spacing),
        regularisers=tuple(regularisers),
    )
    return grads, bool(overflow)


class _Settings(NamedTuple):
    """The static settings of a fit step."""

    position_lr: float
    radius_lr: float
    opacity_lr: float
    losses: tuple[LossTerm, ...]
    voxel_volume: float


def _settings(config: FitConfig, spacing: tuple[float, ...]) -> _Settings:
    """The static settings of a fit's step."""
    unknown = [term.name for term in config.losses if term.name not in LOSSES_JAX]
    if unknown:
        raise ValueError(
            f"unknown loss terms {unknown}; registered: {', '.join(LOSSES_JAX)}"
        )
    return _Settings(
        position_lr=config.position_lr,
        radius_lr=config.radius_lr,
        opacity_lr=config.opacity_lr,
        losses=tuple(config.losses),
        voxel_volume=float(np.prod(spacing)),
    )


def _optimizer(settings: _Settings) -> optax.GradientTransformation:
    """Adam per parameter group, with ``torch.optim.Adam``'s defaults.

    optax adds ``eps`` outside the square root, as torch does.
    """
    labels = JaxLineSegments(
        start="position", end="position", log_radius="radius", opacity_logit="opacity"
    )
    return optax.multi_transform(
        {
            "position": optax.adam(settings.position_lr),
            "radius": optax.adam(settings.radius_lr),
            "opacity": optax.adam(settings.opacity_lr),
        },
        labels,
    )


def _loss(
    segments: JaxLineSegments,
    states: tuple[PyTree, ...],
    target: Array,
    temperature: Array,
    shape: tuple[int, ...],
    spacing: tuple[float, ...],
    render_config: RenderConfig,
    capacity: BinCapacity,
    settings: _Settings,
    regularisers: tuple[JaxPrior, ...],
    width: Array | None = None,
) -> tuple[Array, tuple[dict[str, Array], Any]]:
    """The fit's loss, its terms and the render's bins (traceable).

    ``width`` overrides the render kernel's width as a traced value (see
    :func:`~spleleton.render._render_jax._render_binned`); the fit never
    passes it.
    """
    occupancy, bins = _render_binned(
        segments, temperature, shape, spacing, render_config, capacity, width
    )
    # the volumes are not differentiated (torch computes them under no_grad)
    volumes = (
        kernel_volumes_jax(jax.lax.stop_gradient(segments), render_config)
        / settings.voxel_volume
    )
    context = LossContext(occupancy, target, segments, volumes)
    terms = {
        term.name: LOSSES_JAX[term.name](context, **dict(term.params))
        for term in settings.losses
    }
    # summed in the order of the terms, after all of them are computed
    loss = jnp.zeros((), occupancy.dtype)
    for index, term in enumerate(settings.losses):
        weighted = term.weight * terms[term.name]
        loss = weighted if index == 0 else loss + weighted
    for prior, state in zip(regularisers, states, strict=True):
        for name, term in prior(segments, state).items():
            terms[name] = term
            loss = loss + term
    return loss, (terms, bins)


def _gradient_impl(
    segments, states, target, temperature, shape, spacing, render_config,
    capacity, settings, regularisers,
):  # fmt: skip
    """The loss's gradient (as the fit step takes it) and the overflow flag."""
    (_, (_, bins)), grads = jax.value_and_grad(_loss, has_aux=True)(
        segments, states, target, temperature, shape, spacing, render_config,
        capacity, settings, regularisers,
    )  # fmt: skip
    return grads, bins.overflow


_gradient = jax.jit(
    _gradient_impl,
    static_argnames=(
        "shape",
        "spacing",
        "render_config",
        "capacity",
        "settings",
        "regularisers",
    ),
)


def _step_impl(
    segments: JaxLineSegments,
    opt_state: PyTree,
    states: tuple[PyTree, ...],
    target: Array,
    temperature: Array,
    shape: tuple[int, ...],
    spacing: tuple[float, ...],
    render_config: RenderConfig,
    capacity: BinCapacity,
    settings: _Settings,
    regularisers: tuple[JaxPrior, ...],
) -> tuple[JaxLineSegments, PyTree, dict[str, Array], Array, Array]:
    """One iteration: loss, gradient and Adam update.

    Returns the new segments and optimiser state, the loss terms, and the
    renderer's overflow flag and needed capacity.
    """

    def loss_fn(segments: JaxLineSegments):
        return _loss(
            segments,
            states,
            target,
            temperature,
            shape,
            spacing,
            render_config,
            capacity,
            settings,
            regularisers,
        )

    (loss, (terms, bins)), grads = jax.value_and_grad(loss_fn, has_aux=True)(segments)
    updates, opt_state = _optimizer(settings).update(grads, opt_state, segments)
    segments = optax.apply_updates(segments, updates)
    return segments, opt_state, {"loss": loss, **terms}, bins.overflow, bins.needed


_step = jax.jit(
    _step_impl,
    static_argnames=(
        "shape",
        "spacing",
        "render_config",
        "capacity",
        "settings",
        "regularisers",
    ),
)
