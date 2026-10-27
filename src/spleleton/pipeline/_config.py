"""Settings and results of the pipelines (shared by all backends)."""

from dataclasses import dataclass, field
from typing import Any

from spleleton.fit._config import FitConfig
from spleleton.fit._losses import default_losses
from spleleton.graph import SkeletonGraph
from spleleton.initialise import SegmentLayout
from spleleton.postprocess import clean_graph


@dataclass(frozen=True)
class CleanConfig:
    """Graph clean-up settings (see :func:`spleleton.postprocess.clean_graph`).

    Parameters
    ----------
    enabled : bool
        Run the clean-up at all.
    min_protrusion : float
        Leaves protruding less than this many radii are pruned as spurs.
    bridge_reach : float
        Largest bridged gap, in radii.
    bridge_angle : float
        Largest angle between a free end's tangent and its bridge, in degrees.
    min_island : float
        Components shorter than this many radii (in total) are dropped.
    """

    enabled: bool = True
    min_protrusion: float = 1.5
    bridge_reach: float = 2.0
    bridge_angle: float = 30.0
    min_island: float = 3.0

    def apply(self, graph: SkeletonGraph) -> SkeletonGraph:
        """Clean a graph, or return it unchanged if disabled."""
        if not self.enabled:
            return graph
        return clean_graph(
            graph,
            self.min_protrusion,
            self.bridge_reach,
            self.bridge_angle,
            self.min_island,
        )


@dataclass(frozen=True)
class SkeletonFitConfig:
    """Settings of ``skeleton_fit_torch`` and ``skeleton_fit_jax``.

    Parameters
    ----------
    fit : FitConfig
        Optimiser and loss settings. The sparsity weight defaults to 0.25: in
        3D, 0.5 already prunes real segments of clean trees (junction F1 0.91
        vs 0.96), while in 2D the difference is small.
    free_end_weight : float
        Cost of a leaf edge in units of ``r^D`` voxels (see
        :class:`spleleton.priors.TorchFreeEndCost`); 0 disables it.
    simplify_tolerance : float
        Douglas-Peucker tolerance for turning skeleton edges into segments.

    The graph is built from the fit by the graph stage, whose settings
    (pruning and clean-up) are a separate
    :class:`~spleleton.pipeline.GraphConfig`.
    """

    fit: FitConfig = field(
        default_factory=lambda: FitConfig(
            n_iterations=600, losses=default_losses(sparsity=0.25)
        )
    )
    free_end_weight: float = 3.0
    simplify_tolerance: float = 1.0


@dataclass
class PipelineResult:
    """What a pipeline produced.

    Attributes
    ----------
    graph : SkeletonGraph
        The final skeleton graph.
    segments : TorchLineSegments or JaxLineSegments
        The fitted segments, of the pipeline's backend.
    layout : SegmentLayout, optional
        How segments make up edges (for pipelines with a fixed topology).
    history : list of dict
        Loss per iteration.
    raw_graph : SkeletonGraph, optional
        The graph before clean-up.
    """

    graph: SkeletonGraph
    segments: Any
    layout: SegmentLayout | None
    history: list[dict[str, float]]
    raw_graph: SkeletonGraph | None = None


@dataclass(frozen=True)
class ScratchFitConfig:
    """Settings of ``scratch_fit_torch`` and ``scratch_fit_jax``.

    Parameters
    ----------
    fit : FitConfig
        Optimiser and loss settings of each round.
    n_rounds : int
        Fitting rounds. Between rounds, new seeds are spawned in parts of the
        mask that the rendering does not cover yet.
    connection_weight : float
        Weight of the endpoint-connection loss (cost of a one-voxel gap
        between two nearby endpoints, in voxels).
    connection_reach : float
        Endpoints closer than this many radii (the larger of the two) are
        pulled together. Using the larger radius matters at junctions, where
        thin children's endpoints spread over about the parent's radius.
    spacing_factor : float
        Seeds are at least this many local radii apart.
    min_spawn_voxels : int
        Uncovered regions smaller than this are not seeded.
    min_spawn_thickness : float, optional
        Uncovered regions are only seeded if they are at least this thick
        somewhere (the maximum of their own distance transform), in physical
        units; ``0`` disables the check. ``None`` (default) picks by
        dimension: 0 in 2D and 1.5 in 3D. In 3D, surface noise and bumps
        leave many small uncovered blobs that otherwise spawn spurs and
        fragments (on corrupted 128^3 volumes: 0.83 vs 1.17 spur ends, 1.33 vs
        1.67 components, with half the segments); in 2D, the check slightly
        hurt (1.2 vs 1.0 components on noisy masks).

    The graph is assembled from the fit by the graph stage, whose settings
    (pruning, joining and clean-up) are a separate
    :class:`~spleleton.pipeline.GraphConfig`.
    """

    fit: FitConfig = field(
        default_factory=lambda: FitConfig(
            n_iterations=300, losses=default_losses(sparsity=0.5)
        )
    )
    n_rounds: int = 3
    connection_weight: float = 1.0
    connection_reach: float = 1.5
    spacing_factor: float = 2.0
    min_spawn_voxels: int = 4
    min_spawn_thickness: float | None = None
