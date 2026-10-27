"""The two stages of a fitting pipeline: the fit, and the graph built from it.

A pipeline is a **fit stage**, which turns a mask into fitted segments (and,
for Phase 4, the layout that says which segments make up which edge), and a
**graph stage**, which turns those segments into a graph and cleans it up.
The stages are separate so that one fit can be assembled in many ways: the
fit takes minutes, the graph stage a fraction of a second.

The graph stage's settings are a :class:`GraphConfig`: the assembly method,
by name, its parameters, and the clean-up. Methods are registered with
:func:`register_graph_method`:

- ``"layout"`` snaps the segments back into the layout's topology
  (:meth:`spleleton.initialise.SegmentLayout.to_graph`, Phase 4);
- ``"assemble"`` assembles free segments by their endpoints
  (:func:`spleleton.assemble.assemble_graph`, Phase 5);
- ``"skeletonize"`` ignores the segments and skeletonises the mask
  (:func:`spleleton.baseline.skeletonize_to_graph`), for the baselines that
  fit nothing.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from spleleton.assemble import assemble_graph
from spleleton.baseline import skeletonize_to_graph
from spleleton.graph import SkeletonGraph
from spleleton.initialise import SegmentLayout
from spleleton.pipeline._config import CleanConfig


@dataclass(frozen=True)
class GraphConfig:
    """Settings of the graph stage: assembly and clean-up.

    Parameters
    ----------
    method : str
        The assembly method, by its registered name (``"layout"`` or
        ``"assemble"``, see :func:`register_graph_method`).
    prune_threshold : float
        Segments with a lower opacity count as pruned.
    join : float
        For ``"assemble"``: endpoints (and free ends near a segment's
        interior) closer than this many radii are joined.
    params : tuple of (str, value) pairs
        Further keyword arguments of the method, e.g.
        ``(("contained", 0.5),)`` for ``"assemble"``. Empty uses the method's
        defaults. Pairs, not a dict, so that the config stays hashable.
    clean : CleanConfig
        Graph clean-up after assembling.
    """

    method: str = "assemble"
    prune_threshold: float = 0.1
    join: float = 1.0
    params: tuple[tuple[str, Any], ...] = ()
    clean: CleanConfig = field(default_factory=CleanConfig)


#: ``method(segments, layout, mask, spacing, config) -> SkeletonGraph``.
GraphMethod = Callable[
    [Any, SegmentLayout | None, np.ndarray, tuple[float, ...] | None, GraphConfig],
    SkeletonGraph,
]

_GRAPH_METHODS: dict[str, GraphMethod] = {}


def register_graph_method(name: str) -> Callable[[GraphMethod], GraphMethod]:
    """Register an assembly method under a name (a decorator).

    A method is called as ``method(segments, layout, mask, spacing, config)``
    and returns the raw (not yet cleaned) graph. ``segments`` are the fitted
    segments of any backend, ``layout`` is the fit's layout or ``None``, and
    ``config`` is the :class:`GraphConfig`.
    """

    def register(method: GraphMethod) -> GraphMethod:
        if name in _GRAPH_METHODS:
            raise ValueError(f"graph method {name!r} is already registered")
        _GRAPH_METHODS[name] = method
        return method

    return register


def graph_methods() -> tuple[str, ...]:
    """The names of the registered assembly methods."""
    return tuple(_GRAPH_METHODS)


def graph_stage(
    segments: Any,
    layout: SegmentLayout | None,
    mask: np.ndarray,
    spacing: tuple[float, ...] | None,
    config: GraphConfig,
) -> tuple[SkeletonGraph, SkeletonGraph]:
    """Build a graph from fitted segments and clean it up.

    Parameters
    ----------
    segments : SegmentArrays, TorchLineSegments or JaxLineSegments
        The fitted segments.
    layout : SegmentLayout, optional
        The fit's layout (needed by ``"layout"``).
    mask : np.ndarray
        The binary mask that was fitted.
    spacing : tuple of float, optional
        Physical voxel size per axis.
    config : GraphConfig
        The method and its settings.

    Returns
    -------
    raw : SkeletonGraph
        The assembled graph.
    clean : SkeletonGraph
        The graph after clean-up (``raw`` itself if clean-up is disabled).
    """
    try:
        method = _GRAPH_METHODS[config.method]
    except KeyError:
        raise ValueError(
            f"unknown graph method {config.method!r}; "
            f"registered: {', '.join(_GRAPH_METHODS)}"
        ) from None
    raw = method(segments, layout, mask, spacing, config)
    return raw, config.clean.apply(raw)


@dataclass
class FitOutput:
    """What a fit stage produced.

    Attributes
    ----------
    segments : JaxLineSegments
        The fitted segments, as their raw parameters.
    layout : SegmentLayout, optional
        How segments make up edges (Phase 4), or ``None`` (Phase 5).
    history : list of dict
        Loss per iteration, over all rounds.
    state : FitState
        The state of the last round's last iteration (see
        :class:`spleleton.fit.FitState`).
    rounds : list of dict
        Per round: ``n_segments`` (fitted in the round), ``added`` (new in
        the round; all of them in the first) and ``n_iterations``.
    """

    segments: Any
    layout: SegmentLayout | None
    history: list[dict[str, float]]
    state: Any
    rounds: list[dict[str, int]]


@register_graph_method("layout")
def _layout(segments, layout, mask, spacing, config: GraphConfig) -> SkeletonGraph:
    if layout is None:
        raise ValueError("the 'layout' graph method needs the fit's layout")
    return layout.to_graph(segments, config.prune_threshold, **dict(config.params))


@register_graph_method("assemble")
def _assemble(segments, layout, mask, spacing, config: GraphConfig) -> SkeletonGraph:
    return assemble_graph(
        segments, config.prune_threshold, join=config.join, **dict(config.params)
    )


@register_graph_method("skeletonize")
def _skeletonize(segments, layout, mask, spacing, config: GraphConfig) -> SkeletonGraph:
    return skeletonize_to_graph(mask, spacing, **dict(config.params))
