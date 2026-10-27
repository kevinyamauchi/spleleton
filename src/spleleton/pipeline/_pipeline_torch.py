"""End-to-end skeletonisation pipelines, fitting with PyTorch.

See :mod:`spleleton.pipeline` for the pipelines.
"""

import numpy as np
import torch

from spleleton.assemble import assemble_graph
from spleleton.baseline import skeletonize_to_graph
from spleleton.fit._fit_torch import fit_torch
from spleleton.initialise import segments_from_graph
from spleleton.pipeline._common import initial_seeds, spawn_seeds
from spleleton.pipeline._config import (
    PipelineResult,
    ScratchFitConfig,
    SkeletonFitConfig,
)
from spleleton.primitives._primitives_torch import TorchLineSegments
from spleleton.priors._priors_torch import TorchConnectionLoss, TorchFreeEndCost
from spleleton.render._render_torch import render_torch


def skeleton_fit_torch(
    mask: np.ndarray,
    spacing: tuple[float, ...] | None = None,
    config: SkeletonFitConfig | None = None,
    callback=None,
    device: torch.device | str | None = None,
) -> PipelineResult:
    """Phase 4: fit segments initialised from ``skeletonize``, then clean up.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask, 2D or 3D.
    spacing : tuple of float, optional
        Physical voxel size per axis.
    config : SkeletonFitConfig, optional
        Pipeline settings.
    callback : callable, optional
        Passed to :func:`spleleton.fit.fit_torch`, e.g. a
        :class:`~spleleton.recording.FitRecorder`.
    device : torch.device or str, optional
        Device to fit on.

    Returns
    -------
    PipelineResult
        The cleaned graph, the fitted segments, the layout and the loss.
    """
    config = config or SkeletonFitConfig()
    mask = np.asarray(mask, dtype=bool)
    skeleton = skeletonize_to_graph(mask, spacing=spacing)
    arrays, layout = segments_from_graph(skeleton, tolerance=config.simplify_tolerance)
    segments = TorchLineSegments.from_arrays(arrays, dtype=torch.float32, device=device)
    target = torch.as_tensor(mask, dtype=torch.float32, device=device)
    regularisers = []
    if config.free_end_weight > 0:
        regularisers.append(TorchFreeEndCost(layout, mask.size, config.free_end_weight))
    history = fit_torch(
        segments,
        target,
        spacing=spacing,
        config=config.fit,
        callback=callback,
        regularisers=regularisers,
    )
    graph = config.clean.apply(layout.to_graph(segments, config.prune_threshold))
    return PipelineResult(
        graph=graph, segments=segments, layout=layout, history=history
    )


def scratch_fit_torch(
    mask: np.ndarray,
    spacing: tuple[float, ...] | None = None,
    config: ScratchFitConfig | None = None,
    callback=None,
    device: torch.device | str | None = None,
) -> PipelineResult:
    """Phase 5: fit segments seeded from the distance transform, no skeleton.

    Seeds are placed on the ridge of the distance transform
    (:func:`spleleton.seeding.seed_segments`) and fitted with the endpoint
    connection loss, which pulls nearby endpoints together so the segments
    form chains. After each round except the last, parts of the mask the
    rendering does not cover are seeded too. Finally the segments are
    assembled into a graph (:func:`spleleton.assemble.assemble_graph`) and
    cleaned up.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask, 2D or 3D.
    spacing : tuple of float, optional
        Physical voxel size per axis.
    config : ScratchFitConfig, optional
        Pipeline settings.
    callback : callable, optional
        Passed to every round of :func:`spleleton.fit.fit_torch`.
    device : torch.device or str, optional
        Device to fit on.

    Returns
    -------
    PipelineResult
        The cleaned graph, the fitted segments (no layout) and the loss of all
        rounds.
    """
    config = config or ScratchFitConfig()
    mask = np.asarray(mask, dtype=bool)
    target = torch.as_tensor(mask, dtype=torch.float32, device=device)

    def to_torch(arrays):
        return TorchLineSegments.from_arrays(arrays, dtype=torch.float32, device=device)

    segments = to_torch(initial_seeds(mask, spacing, config))
    history: list[dict[str, float]] = []
    for round_index in range(config.n_rounds):
        connection = TorchConnectionLoss(
            mask.size, config.connection_weight, config.connection_reach
        )
        history += fit_torch(
            segments,
            target,
            spacing=spacing,
            config=config.fit,
            callback=callback,
            regularisers=[connection],
        )
        if round_index == config.n_rounds - 1:
            break
        with torch.no_grad():
            covered = render_torch(segments, mask.shape, spacing).cpu().numpy() >= 0.5
        new = spawn_seeds(mask, covered, spacing, config)
        if len(new):
            segments = to_torch(segments.to_arrays().concatenate(new))

    graph = assemble_graph(segments, config.prune_threshold, join=config.join)
    graph = config.clean.apply(graph)
    return PipelineResult(graph=graph, segments=segments, layout=None, history=history)
