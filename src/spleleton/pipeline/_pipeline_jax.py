"""End-to-end skeletonisation pipelines, fitting with JAX.

See :mod:`spleleton.pipeline` for the pipelines. They do what the torch
versions do, step for step; only the fit, the priors and the coverage render
between rounds use JAX. Segments are fitted in float32, as in torch.

Each pipeline is its fit stage (``*_fit_stage_jax``, which returns a
:class:`~spleleton.pipeline.FitOutput`) followed by the graph stage
(:func:`~spleleton.pipeline.graph_stage`).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from spleleton.baseline import skeletonize_to_graph
from spleleton.fit._fit_jax import fit_jax, loss_gradient_jax
from spleleton.initialise import segments_from_graph
from spleleton.pipeline._common import initial_seeds, spawn_seeds
from spleleton.pipeline._config import (
    PipelineResult,
    ScratchFitConfig,
    SkeletonFitConfig,
)
from spleleton.pipeline._stages import FitOutput, GraphConfig, graph_stage
from spleleton.primitives import SegmentArrays
from spleleton.primitives._arrays import as_segment_arrays
from spleleton.primitives._primitives_jax import JaxLineSegments
from spleleton.priors._priors_jax import JaxConnectionLoss, JaxFreeEndCost
from spleleton.render._render_jax import render_jax
from spleleton.targets import TargetConfig, target_field, target_render_mode


def skeleton_fit_stage_jax(
    mask: np.ndarray,
    spacing: tuple[float, ...] | None = None,
    config: SkeletonFitConfig | None = None,
    callback=None,
    device: jax.Device | None = None,
    target: TargetConfig | None = None,
    init_jitter: float = 0.0,
    jitter_seed: int = 0,
    diagnostics: bool = False,
) -> FitOutput:
    """Phase 4's fit stage: fit segments initialised from ``skeletonize``.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask, 2D or 3D.
    spacing : tuple of float, optional
        Physical voxel size per axis.
    config : SkeletonFitConfig, optional
        Fit settings.
    callback : callable, optional
        Passed to :func:`spleleton.fit.fit_jax`, e.g. a
        :class:`~spleleton.recording.FitRecorder`.
    device : jax.Device, optional
        Device to fit on. Defaults to JAX's default device.
    target : TargetConfig, optional
        The field the rendering is fitted to (see :mod:`spleleton.targets`).
        Defaults to the mask.
    init_jitter, jitter_seed, diagnostics
        See :func:`scratch_fit_stage_jax`.

    Returns
    -------
    FitOutput
        The fitted segments (``JaxLineSegments``), the layout, the loss and
        the final fit state.
    """
    config = config or SkeletonFitConfig()
    mask = np.asarray(mask, dtype=bool)
    field = _target(mask, spacing, target, config)
    arrays, layout = fit_start(mask, spacing, config, init_jitter, jitter_seed)
    segments = JaxLineSegments.from_arrays(arrays, dtype=jnp.float32, device=device)
    regularisers = []
    if config.free_end_weight > 0:
        regularisers.append(JaxFreeEndCost(layout, mask.size, config.free_end_weight))
    n_segments = len(segments)
    segments, history, state = fit_jax(
        segments,
        field,
        spacing=spacing,
        config=config.fit,
        callback=callback,
        regularisers=regularisers,
        device=device,
        return_state=True,
    )
    rounds = [
        {
            "n_segments": n_segments,
            "added": n_segments,
            "n_iterations": config.fit.n_iterations,
        }
    ]
    if diagnostics:
        rounds[0].update(
            _convergence(history, state, field, spacing, config.fit, regularisers)
        )
    return FitOutput(segments, layout, history, state, rounds)


def scratch_fit_stage_jax(
    mask: np.ndarray,
    spacing: tuple[float, ...] | None = None,
    config: ScratchFitConfig | None = None,
    callback=None,
    device: jax.Device | None = None,
    target: TargetConfig | None = None,
    init_jitter: float = 0.0,
    jitter_seed: int = 0,
    diagnostics: bool = False,
    initial: SegmentArrays | None = None,
) -> FitOutput:
    """Phase 5's fit stage: fit segments seeded from the distance transform.

    As the fit of :func:`spleleton.pipeline.scratch_fit_torch`. Each round
    has a new number of segments, so the fit step compiles again every round.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask, 2D or 3D.
    spacing : tuple of float, optional
        Physical voxel size per axis.
    config : ScratchFitConfig, optional
        Fit settings.
    callback : callable, optional
        Passed to every round of :func:`spleleton.fit.fit_jax`.
    device : jax.Device, optional
        Device to fit on. Defaults to JAX's default device.
    target : TargetConfig, optional
        The field the rendering is fitted to (see :mod:`spleleton.targets`).
        Defaults to the mask. Seeding and spawning use the mask.
    init_jitter : float
        Moves each endpoint of the initial segments by at most this much, in
        a random direction (physical units), for the benchmark's noise floor.
        0 moves nothing.
    jitter_seed : int
        Seeds the jitter.
    diagnostics : bool
        Add each round's convergence to its record: ``loss_change`` (the
        relative change of the total loss over the round's last 50
        iterations) and ``grad_norm`` (the norm of the gradient at its last
        iteration). This compiles one more program per round.
    initial : SegmentArrays, optional
        Start from these segments instead of the distance transform's seeds
        (e.g. the true graph, for the benchmark's ground-truth start); every
        round and spawn runs as usual. The jitter applies to them too.

    Returns
    -------
    FitOutput
        The fitted segments (``JaxLineSegments``, no layout), the loss of all
        rounds and the final fit state of the last round.
    """
    config = config or ScratchFitConfig()
    mask = np.asarray(mask, dtype=bool)
    field = _target(mask, spacing, target, config)

    def to_jax(arrays):
        return JaxLineSegments.from_arrays(arrays, dtype=jnp.float32, device=device)

    if initial is None:
        start = fit_start(mask, spacing, config, init_jitter, jitter_seed)[0]
    else:
        start = jitter_segments(as_segment_arrays(initial), init_jitter, jitter_seed)
    segments = to_jax(start)
    history: list[dict[str, float]] = []
    rounds: list[dict[str, int]] = []
    added = len(segments)
    state = None
    for round_index in range(config.n_rounds):
        connection = JaxConnectionLoss(
            mask.size, config.connection_weight, config.connection_reach
        )
        rounds.append(
            {
                "n_segments": len(segments),
                "added": added,
                "n_iterations": config.fit.n_iterations,
            }
        )
        segments, round_history, state = fit_jax(
            segments,
            field,
            spacing=spacing,
            config=config.fit,
            callback=callback,
            regularisers=[connection],
            device=device,
            return_state=True,
        )
        history += round_history
        if diagnostics:
            rounds[-1].update(
                _convergence(
                    round_history, state, field, spacing, config.fit, [connection]
                )
            )
        if round_index == config.n_rounds - 1:
            break
        covered = np.asarray(render_jax(segments, mask.shape, spacing)) >= 0.5
        new = spawn_seeds(mask, covered, spacing, config)
        added = len(new)
        if added:
            segments = to_jax(segments.to_arrays().concatenate(new))
    return FitOutput(segments, None, history, state, rounds)


def skeleton_fit_jax(
    mask: np.ndarray,
    spacing: tuple[float, ...] | None = None,
    config: SkeletonFitConfig | None = None,
    callback=None,
    device: jax.Device | None = None,
    graph: GraphConfig | None = None,
    target: TargetConfig | None = None,
) -> PipelineResult:
    """Phase 4: fit segments initialised from ``skeletonize``, then clean up.

    :func:`skeleton_fit_stage_jax` followed by the graph stage.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask, 2D or 3D.
    spacing : tuple of float, optional
        Physical voxel size per axis.
    config : SkeletonFitConfig, optional
        Fit settings.
    callback : callable, optional
        Passed to :func:`spleleton.fit.fit_jax`, e.g. a
        :class:`~spleleton.recording.FitRecorder`.
    device : jax.Device, optional
        Device to fit on. Defaults to JAX's default device.
    graph : GraphConfig, optional
        Graph stage settings. Defaults to ``GraphConfig(method="layout")``.
    target : TargetConfig, optional
        The field the rendering is fitted to. Defaults to the mask.

    Returns
    -------
    PipelineResult
        The cleaned graph, the fitted segments (``JaxLineSegments``), the
        layout and the loss.
    """
    fit = skeleton_fit_stage_jax(mask, spacing, config, callback, device, target)
    graph = graph or GraphConfig(method="layout")
    raw, clean = graph_stage(fit.segments, fit.layout, mask, spacing, graph)
    return PipelineResult(
        graph=clean,
        segments=fit.segments,
        layout=fit.layout,
        history=fit.history,
        raw_graph=raw,
    )


def scratch_fit_jax(
    mask: np.ndarray,
    spacing: tuple[float, ...] | None = None,
    config: ScratchFitConfig | None = None,
    callback=None,
    device: jax.Device | None = None,
    graph: GraphConfig | None = None,
    target: TargetConfig | None = None,
) -> PipelineResult:
    """Phase 5: fit segments seeded from the distance transform, no skeleton.

    :func:`scratch_fit_stage_jax` followed by the graph stage.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask, 2D or 3D.
    spacing : tuple of float, optional
        Physical voxel size per axis.
    config : ScratchFitConfig, optional
        Fit settings.
    callback : callable, optional
        Passed to every round of :func:`spleleton.fit.fit_jax`.
    device : jax.Device, optional
        Device to fit on. Defaults to JAX's default device.
    graph : GraphConfig, optional
        Graph stage settings. Defaults to ``GraphConfig(method="assemble")``.
    target : TargetConfig, optional
        The field the rendering is fitted to. Defaults to the mask.

    Returns
    -------
    PipelineResult
        The cleaned graph, the fitted segments (``JaxLineSegments``, no
        layout) and the loss of all rounds.
    """
    fit = scratch_fit_stage_jax(mask, spacing, config, callback, device, target)
    graph = graph or GraphConfig(method="assemble")
    raw, clean = graph_stage(fit.segments, None, mask, spacing, graph)
    return PipelineResult(
        graph=clean,
        segments=fit.segments,
        layout=None,
        history=fit.history,
        raw_graph=raw,
    )


def _target(
    mask: np.ndarray,
    spacing: tuple[float, ...] | None,
    target: TargetConfig | None,
    config: SkeletonFitConfig | ScratchFitConfig,
) -> np.ndarray:
    """The target field, after checking that the fit renders in its mode."""
    target = target or TargetConfig()
    mode = target_render_mode(target)
    if config.fit.render.mode != mode:
        raise ValueError(
            f"target {target.name!r} is for the {mode!r} render mode, but the "
            f"fit renders in {config.fit.render.mode!r} (set fit.render.mode)"
        )
    return target_field(target, mask, spacing)


#: Convergence is the relative loss change over this many last iterations.
CONVERGENCE_WINDOW = 50


def fit_start(
    mask: np.ndarray,
    spacing: tuple[float, ...] | None,
    config: SkeletonFitConfig | ScratchFitConfig,
    init_jitter: float = 0.0,
    jitter_seed: int = 0,
) -> tuple[SegmentArrays, object]:
    """The segments a fit stage starts from, and their layout.

    Phase 4 (a :class:`SkeletonFitConfig`) starts from ``skeletonize``'s
    graph, cut into segments, with their layout; Phase 5 (a
    :class:`ScratchFitConfig`) from the distance transform's seeds, with no
    layout. Either is then jittered (:func:`jitter_segments`).
    """
    mask = np.asarray(mask, dtype=bool)
    if isinstance(config, SkeletonFitConfig):
        skeleton = skeletonize_to_graph(mask, spacing=spacing)
        arrays, layout = segments_from_graph(
            skeleton, tolerance=config.simplify_tolerance
        )
    else:
        arrays, layout = initial_seeds(mask, spacing, config), None
    return jitter_segments(arrays, init_jitter, jitter_seed), layout


def jitter_segments(arrays: SegmentArrays, jitter: float, seed: int) -> SegmentArrays:
    """Move each endpoint by at most ``jitter``, in a random direction.

    Each endpoint moves by a length drawn uniformly from ``[jitter / 2,
    jitter]``, so that no endpoint stays put. A jitter of 0 returns the
    arrays unchanged.
    """
    if jitter <= 0:
        return arrays
    rng = np.random.default_rng(seed)

    def moved(points: np.ndarray) -> np.ndarray:
        direction = rng.normal(size=points.shape)
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        length = rng.uniform(0.5 * jitter, jitter, size=(len(points), 1))
        return points + length * direction

    return dataclasses.replace(arrays, start=moved(arrays.start), end=moved(arrays.end))


def _convergence(history, state, field, spacing, config, regularisers) -> dict:
    """A round's relative loss change over its last iterations, and its gradient."""
    losses = np.array([row["loss"] for row in history])
    window = min(CONVERGENCE_WINDOW, len(losses) - 1)
    change = (
        float((losses[-1 - window] - losses[-1]) / abs(losses[-1]))
        if window > 0
        else float("nan")
    )
    grad_norm = float("nan")
    if state.pre_last is not None:
        grads, _ = loss_gradient_jax(
            state.pre_last, field, state, spacing, config, regularisers
        )
        leaves = [np.asarray(g, dtype=np.float64) for g in jax.tree.leaves(grads)]
        grad_norm = float(np.sqrt(sum(float((g**2).sum()) for g in leaves)))
    return {"loss_change": change, "grad_norm": grad_norm}
