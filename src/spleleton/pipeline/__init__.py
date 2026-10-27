"""End-to-end skeletonisation pipelines.

- ``skeleton_fit_*`` (Phase 4): skeletonise the mask, simplify the skeleton
  into line segments, fit them to the mask with the free-end prior, rebuild
  the graph with the skeleton's topology, and clean it up (bridge breaks,
  prune spurs).
- ``scratch_fit_*`` (Phase 5): seed short segments on the ridge of the
  distance transform, fit them with the endpoint-connection loss over several
  rounds (spawning seeds where the mask is not yet covered), assemble them
  into a graph and clean it up.

Each pipeline exists once per backend, with a ``_torch`` or ``_jax`` suffix;
the settings and the result type are shared. The JAX pipelines are split into
a fit stage (``*_fit_stage_jax``) and a graph stage (:func:`graph_stage`, with
its own :class:`GraphConfig`), so that one fit can be assembled in many ways.
The torch pipelines are not updated to the split yet.
"""

from typing import TYPE_CHECKING

from spleleton._backends import lazy_exports
from spleleton.pipeline._config import (
    CleanConfig,
    PipelineResult,
    ScratchFitConfig,
    SkeletonFitConfig,
)
from spleleton.pipeline._stages import (
    FitOutput,
    GraphConfig,
    graph_methods,
    graph_stage,
    register_graph_method,
)

if TYPE_CHECKING:
    from spleleton.pipeline._pipeline_jax import (
        scratch_fit_jax,
        scratch_fit_stage_jax,
        skeleton_fit_jax,
        skeleton_fit_stage_jax,
    )
    from spleleton.pipeline._pipeline_torch import (
        scratch_fit_torch,
        skeleton_fit_torch,
    )

__all__ = [
    "CleanConfig",
    "FitOutput",
    "GraphConfig",
    "PipelineResult",
    "ScratchFitConfig",
    "SkeletonFitConfig",
    "graph_methods",
    "graph_stage",
    "register_graph_method",
    "scratch_fit_jax",
    "scratch_fit_stage_jax",
    "scratch_fit_torch",
    "skeleton_fit_jax",
    "skeleton_fit_stage_jax",
    "skeleton_fit_torch",
]

__getattr__, __dir__ = lazy_exports(
    __name__,
    torch={"._pipeline_torch": ["scratch_fit_torch", "skeleton_fit_torch"]},
    jax={
        "._pipeline_jax": [
            "scratch_fit_jax",
            "scratch_fit_stage_jax",
            "skeleton_fit_jax",
            "skeleton_fit_stage_jax",
        ]
    },
)
