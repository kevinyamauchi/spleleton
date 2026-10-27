"""Record a fit's intermediate states, to play it back (e.g. in napari).

A :class:`FitRecorder` is passed to a fit (``fit_torch`` or ``fit_jax``) as
its callback. Every ``every`` iterations it copies the segments' parameters
and, optionally, renders them. The final state is always recorded too. A
recorder can follow several consecutive fits (e.g. the rounds of
:func:`spleleton.pipeline.scratch_fit_torch`): iterations continue across
rounds, the last state of each round is kept, and the number of segments may
change between rounds (arrays are padded with NaN). Afterwards,
:meth:`FitRecorder.finish` bundles the snapshots, the per-iteration loss and
any context (target mask, segment layout, ground truth) into a
:class:`FitRecording`, which saves to and loads from a single ``.npz`` file.

Example
-------
>>> recorder = FitRecorder(target.shape, every=10)  # doctest: +SKIP
>>> history = fit_torch(segments, target, callback=recorder)  # doctest: +SKIP
>>> recorder.finish(history, target=target).save("fit.npz")  # doctest: +SKIP
"""

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from spleleton.graph import SkeletonGraph
from spleleton.initialise import SegmentLayout
from spleleton.primitives._arrays import SegmentArrays, as_segment_arrays
from spleleton.render._config import RenderConfig

# opacities are clamped to this distance from 0 and 1 when segments are rebuilt
_OPACITY_EPS = 1e-12


@dataclass
class FitRecording:
    """Snapshots of a fit, with its loss and context.

    Attributes
    ----------
    iterations : np.ndarray
        Iteration of each snapshot, shape ``(S,)``. A snapshot is taken
        before the step of its iteration; the last one is the final state.
    start, end : np.ndarray
        Segment endpoints, shape ``(S, N, D)``. If the number of segments
        changed during the fit, ``N`` is the largest number and missing
        segments are NaN.
    radius : np.ndarray
        Radii at both ends, shape ``(S, N, 2)``.
    opacity : np.ndarray
        Opacities, shape ``(S, N)``.
    loss : dict of str to np.ndarray
        Each loss term per iteration, shape ``(n_iterations,)``.
    renderings : np.ndarray, optional
        Rendered occupancy of each snapshot, shape ``(S, *image_shape)``,
        stored as float16.
    target : np.ndarray, optional
        The image the segments were fitted to.
    spacing : tuple of float, optional
        Physical voxel size per axis.
    layout : SegmentLayout, optional
        How segments make up graph edges, to rebuild a graph per snapshot.
    ground_truth : SkeletonGraph, optional
        The true graph, for comparison.
    final_graph : SkeletonGraph, optional
        The graph the pipeline produced in the end (e.g. after clean-up).
    """

    iterations: np.ndarray
    start: np.ndarray
    end: np.ndarray
    radius: np.ndarray
    opacity: np.ndarray
    loss: dict[str, np.ndarray]
    renderings: np.ndarray | None = None
    target: np.ndarray | None = None
    spacing: tuple[float, ...] | None = None
    layout: SegmentLayout | None = None
    ground_truth: SkeletonGraph | None = None
    final_graph: SkeletonGraph | None = None

    @property
    def n_snapshots(self) -> int:
        """Number of snapshots."""
        return len(self.iterations)

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return self.start.shape[-1]

    def segments(self, snapshot: int) -> SegmentArrays:
        """The segments at one snapshot."""
        present = ~np.isnan(self.opacity[snapshot])
        return SegmentArrays(
            *(
                getattr(self, key)[snapshot][present]
                for key in ("start", "end", "radius", "opacity")
            )
        )

    def graph(self, snapshot: int, prune_threshold: float = 0.1) -> SkeletonGraph:
        """The graph at one snapshot (requires a layout)."""
        if self.layout is None:
            raise ValueError("this recording has no segment layout")
        return self.layout.to_graph(self.segments(snapshot), prune_threshold)

    def save(self, path: str | Path) -> Path:
        """Save as a compressed ``.npz`` file; the suffix is added if missing."""
        path = Path(path)
        # append rather than replace, so a name like "fit_0.1" keeps its "0.1"
        if path.suffix != ".npz":
            path = path.with_name(path.name + ".npz")
        arrays = {
            "iterations": self.iterations,
            "start": self.start,
            "end": self.end,
            "radius": self.radius,
            "opacity": self.opacity,
            **{f"loss_{key}": value for key, value in self.loss.items()},
        }
        if self.renderings is not None:
            arrays["renderings"] = self.renderings
        if self.target is not None:
            arrays["target"] = self.target
        if self.layout is not None:
            arrays.update({f"layout_{k}": v for k, v in self.layout.to_dict().items()})
        if self.ground_truth is not None:
            arrays.update(
                {f"truth_{k}": v for k, v in self.ground_truth.to_dict().items()}
            )
        if self.final_graph is not None:
            arrays.update(
                {f"final_{k}": v for k, v in self.final_graph.to_dict().items()}
            )
        metadata = {"spacing": None if self.spacing is None else list(self.spacing)}
        np.savez_compressed(path, metadata=np.array(json.dumps(metadata)), **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "FitRecording":
        """Load a recording saved with :meth:`save`."""
        with np.load(path) as data:

            def prefixed(prefix: str) -> dict[str, np.ndarray]:
                return {
                    key.removeprefix(prefix): data[key]
                    for key in data.files
                    if key.startswith(prefix)
                }

            metadata = json.loads(str(data["metadata"]))
            layout = prefixed("layout_")
            truth = prefixed("truth_")
            final = prefixed("final_")
            return cls(
                iterations=data["iterations"],
                start=data["start"],
                end=data["end"],
                radius=data["radius"],
                opacity=data["opacity"],
                loss=prefixed("loss_"),
                renderings=data.get("renderings"),
                target=data.get("target"),
                spacing=None
                if metadata["spacing"] is None
                else tuple(metadata["spacing"]),
                layout=SegmentLayout.from_dict(layout) if layout else None,
                ground_truth=SkeletonGraph.from_dict(truth) if truth else None,
                final_graph=SkeletonGraph.from_dict(final) if final else None,
            )


class FitRecorder:
    """A fit callback (for ``fit_torch`` or ``fit_jax``) that takes snapshots.

    Snapshots are rendered with the renderer of the backend whose segments
    the fit passes in.

    Parameters
    ----------
    shape : tuple of int
        Shape of the target image, for rendering.
    every : int
        Take a snapshot every ``every`` iterations (and always of the final
        state).
    spacing : tuple of float, optional
        Physical voxel size per axis, for rendering.
    render_config : RenderConfig, optional
        Renderer settings for the snapshots' renderings. Defaults to the tube
        kernel with the max union, i.e. the geometry the segments describe.
    render_snapshots : bool
        Render every snapshot. Renderings are the largest part of a
        recording; switch them off for long 3D fits.
    """

    def __init__(
        self,
        shape: tuple[int, ...],
        every: int = 10,
        spacing: tuple[float, ...] | None = None,
        render_config: RenderConfig | None = None,
        render_snapshots: bool = True,
    ) -> None:
        if every < 1:
            raise ValueError(f"every must be at least 1, got {every}")
        self.shape = tuple(shape)
        self.every = every
        self.spacing = spacing
        self.render_config = render_config or RenderConfig()
        self.render_snapshots = render_snapshots
        self._snapshots: list[tuple[int, dict[str, np.ndarray]]] = []
        self._latest: tuple[int, dict[str, np.ndarray]] | None = None
        self._offset = 0
        self._last_iteration = -1
        self._segment_type: type | None = None

    def __call__(self, iteration: int, primitives: Any) -> None:
        """Remember the current state; keep it if it is due for a snapshot."""
        if iteration < self._last_iteration:
            # a new fit (round) started: keep the last state of the previous one
            self._commit_latest()
            self._offset += self._last_iteration
        self._last_iteration = iteration
        self._segment_type = type(primitives)
        iteration += self._offset
        arrays = as_segment_arrays(primitives)
        state = {
            "start": arrays.start,
            "end": arrays.end,
            "radius": arrays.radius,
            "opacity": arrays.opacity,
        }
        if iteration % self.every == 0:
            if self.render_snapshots:
                state["rendering"] = self._render(primitives)
            self._snapshots.append((iteration, state))
            self._latest = None
        else:
            self._latest = (iteration, state)

    def _commit_latest(self) -> None:
        if self._latest is None:
            return
        iteration, state = self._latest
        if self.render_snapshots:
            state["rendering"] = self._render(self._rebuild(state))
        self._snapshots.append((iteration, state))
        self._latest = None

    def finish(
        self,
        history: list[dict[str, float]],
        target: Any = None,
        layout: SegmentLayout | None = None,
        ground_truth: SkeletonGraph | None = None,
        final_graph: SkeletonGraph | None = None,
    ) -> FitRecording:
        """Bundle the snapshots into a recording.

        Parameters
        ----------
        history : list of dict
            The loss history returned by the fit.
        target : array, optional
            The image the segments were fitted to.
        layout : SegmentLayout, optional
            How segments make up graph edges.
        ground_truth : SkeletonGraph, optional
            The true graph.
        final_graph : SkeletonGraph, optional
            The graph the pipeline produced in the end.

        Returns
        -------
        FitRecording
            The recording.
        """
        self._commit_latest()
        snapshots = list(self._snapshots)
        if not snapshots:
            raise ValueError("nothing was recorded; was the recorder passed to fit?")
        n_max = max(len(state["opacity"]) for _, state in snapshots)

        def stack(key: str) -> np.ndarray:
            if key == "rendering":
                return np.stack([state[key] for _, state in snapshots])
            arrays = []
            for _, state in snapshots:
                array = state[key].astype(float)
                missing = n_max - len(array)
                if missing:
                    padding = np.full((missing, *array.shape[1:]), np.nan)
                    array = np.concatenate([array, padding])
                arrays.append(array)
            return np.stack(arrays)

        if target is not None and hasattr(target, "detach"):  # a torch tensor
            target = target.detach().cpu().numpy()
        return FitRecording(
            iterations=np.array([iteration for iteration, _ in snapshots]),
            start=stack("start"),
            end=stack("end"),
            radius=stack("radius"),
            opacity=stack("opacity"),
            loss={key: np.array([row[key] for row in history]) for key in history[0]}
            if history
            else {},
            renderings=stack("rendering") if self.render_snapshots else None,
            target=None if target is None else np.asarray(target),
            spacing=self.spacing,
            layout=layout,
            ground_truth=ground_truth,
            final_graph=final_graph,
        )

    def _render(self, primitives: Any) -> np.ndarray:
        backend = primitives.backend
        module = importlib.import_module(f"spleleton.render._render_{backend}")
        image = module.render_snapshot(
            primitives, self.shape, self.spacing, self.render_config
        )
        return image.astype(np.float16)

    def _rebuild(self, state: dict[str, np.ndarray]) -> Any:
        """Segments of the recorded backend from a recorded state.

        Torch segments are rebuilt in float64; JAX segments in its default
        precision.
        """
        opacity = np.clip(state["opacity"], _OPACITY_EPS, 1 - _OPACITY_EPS)
        arrays = SegmentArrays(state["start"], state["end"], state["radius"], opacity)
        if self._segment_type.backend == "torch":
            import torch

            return self._segment_type.from_arrays(arrays, dtype=torch.float64)
        return self._segment_type.from_arrays(arrays)
