"""napari helpers for looking at samples and skeleton graphs.

Requires the ``viz`` extra (``napari[all]``); napari is imported only when a
helper is called. Images are added with their voxel spacing as the layer
scale, so they line up with graphs, which are in physical coordinates.
"""

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import napari

    from spleleton.graph import SkeletonGraph
    from spleleton.recording import FitRecording
    from spleleton.synthetic.dataset import Sample

# colours of graph nodes by role
_NODE_COLORS = {"end": "yellow", "junction": "cyan", "other": "magenta"}

# colours of corruption markers by type
_CORRUPTION_COLORS = {
    "bump": "lime",
    "hole": "orange",
    "gap": "red",
    "contact": "magenta",
}


def add_graph(
    viewer: "napari.Viewer",
    graph: "SkeletonGraph",
    name: str = "graph",
    color: str = "red",
    width: float = 0.5,
) -> list:
    """Add a skeleton graph as a path layer (edges) and a points layer (nodes).

    Nodes are coloured by role: ends yellow, junctions cyan, anything else
    (degree 0 or 2) magenta.

    Parameters
    ----------
    viewer : napari.Viewer
        The viewer to add to.
    graph : SkeletonGraph
        The graph, in physical coordinates.
    name : str
        Prefix of the layer names.
    color : str
        Edge colour.
    width : float
        Edge width.

    Returns
    -------
    list
        The added layers.
    """
    layers = []
    if graph.n_edges:
        layers.append(
            viewer.add_shapes(
                list(graph.edge_paths),
                shape_type="path",
                edge_color=color,
                edge_width=width,
                name=f"{name} edges",
            )
        )
    degrees = graph.degrees
    roles = np.where(degrees == 1, "end", np.where(degrees >= 3, "junction", "other"))
    layers.append(
        viewer.add_points(
            graph.node_positions,
            features={"degree": degrees, "role": roles},
            face_color=[_NODE_COLORS[role] for role in roles],
            size=2.0,
            name=f"{name} nodes",
        )
    )
    return layers


def add_corruptions(
    viewer: "napari.Viewer", records: list[dict], ndim: int, name: str = "corruptions"
) -> list:
    """Mark the location of each corruption, coloured by type.

    Bumps are lime, holes orange, gaps red and contacts magenta (a contact is
    marked at its midpoint). Surface noise has no location and is skipped.

    Returns
    -------
    list
        The added layers (empty if there is nothing to mark).
    """
    positions, kinds = [], []
    for record in records:
        if "position" in record:
            positions.append(record["position"])
        elif record["type"] == "contact":
            positions.append(
                (np.asarray(record["start"]) + np.asarray(record["end"])) / 2
            )
        else:
            continue
        kinds.append(record["type"])
    if not positions:
        return []
    return [
        viewer.add_points(
            np.reshape(positions, (-1, ndim)),
            features={"type": kinds},
            face_color="transparent",
            border_color=[_CORRUPTION_COLORS.get(kind, "white") for kind in kinds],
            border_width=0.2,
            size=6.0,
            name=name,
        )
    ]


def view_sample(
    sample: "Sample",
    viewer: "napari.Viewer | None" = None,
    baseline: bool = True,
) -> "napari.Viewer":
    """Show a sample: both masks, the true graph, corruptions and a baseline.

    Parameters
    ----------
    sample : Sample
        The sample to show.
    viewer : napari.Viewer, optional
        Viewer to add to. A new one is created if omitted.
    baseline : bool
        Also show the graph from :func:`spleleton.baseline.skeletonize_to_graph`
        of the corrupted mask.

    Returns
    -------
    napari.Viewer
        The viewer.
    """
    import napari

    if viewer is None:
        viewer = napari.Viewer(ndisplay=sample.graph.ndim)
    scale = sample.spacing
    viewer.add_labels(
        sample.clean_mask.astype(np.uint8),
        name="clean mask",
        scale=scale,
        visible=False,
    )
    viewer.add_labels(
        sample.mask.astype(np.uint8), name="mask", scale=scale, opacity=0.4
    )
    add_graph(viewer, sample.graph, name="ground truth", color="red")
    add_corruptions(viewer, sample.corruptions, ndim=sample.graph.ndim)
    if baseline:
        from spleleton.baseline import skeletonize_to_graph

        add_graph(
            viewer,
            skeletonize_to_graph(sample.mask, spacing=sample.spacing),
            name="skeletonize",
            color="dodgerblue",
        )
    return viewer


def view_fit(
    recording: "FitRecording",
    viewer: "napari.Viewer | None" = None,
    graphs: bool = True,
    prune_threshold: float = 0.1,
) -> "napari.Viewer":
    """Play back a recorded fit, with one slider step per snapshot.

    The viewer gets a leading "snapshot" axis; moving its slider (or pressing
    play) steps through the fit. Layers:

    - **target**: the fitted mask (the same at every snapshot);
    - **rendering**: the rendered occupancy at each snapshot;
    - **segments**: each segment as a line, coloured by opacity (viridis:
      dark is pruned, yellow is opaque);
    - **fitted graph**: the graph the snapshot would give
      (:meth:`~spleleton.initialise.SegmentLayout.to_graph`), if the
      recording has a layout and ``graphs`` is true;
    - **final graph**: the pipeline's final graph (e.g. after clean-up), if
      the recording has one;
    - **ground truth**: the true graph, if the recording has one.

    A text overlay shows the iteration, each loss term, and how many
    segments are above the prune threshold.

    Parameters
    ----------
    recording : FitRecording
        The recording to show.
    viewer : napari.Viewer, optional
        Viewer to add to. A new one is created if omitted.
    graphs : bool
        Rebuild and show the graph at every snapshot.
    prune_threshold : float
        Opacity below which a segment counts as pruned.

    Returns
    -------
    napari.Viewer
        The viewer.
    """
    import napari

    ndim = recording.ndim
    if viewer is None:
        viewer = napari.Viewer(ndisplay=ndim)
    spacing = recording.spacing or (1.0,) * ndim
    n_snapshots = recording.n_snapshots
    snapshot_index = np.arange(n_snapshots, dtype=float)

    if recording.target is not None:
        viewer.add_labels(
            np.asarray(recording.target).astype(np.uint8),
            name="target",
            scale=spacing,
            opacity=0.3,
        )
    if recording.renderings is not None:
        viewer.add_image(
            recording.renderings.astype(np.float32),
            name="rendering",
            scale=(1.0, *spacing),
            colormap="magma",
            contrast_limits=(0.0, 1.0),
            blending="additive",
            opacity=0.7,
        )

    # every segment of every snapshot, as a line in (snapshot, *space);
    # NaN rows are segments that did not exist yet at that snapshot
    times = np.repeat(snapshot_index, recording.start.shape[1])[:, None]
    starts = np.hstack([times, recording.start.reshape(-1, ndim)])
    ends = np.hstack([times, recording.end.reshape(-1, ndim)])
    present = ~np.isnan(recording.opacity.reshape(-1))
    viewer.add_shapes(
        list(np.stack([starts, ends], axis=1)[present]),
        shape_type="line",
        features={
            "opacity": recording.opacity.reshape(-1)[present],
            "radius": recording.radius.mean(axis=-1).reshape(-1)[present],
        },
        edge_color="opacity",
        edge_colormap="viridis",
        edge_contrast_limits=(0.0, 1.0),
        edge_width=0.4,
        name="segments",
    )

    if graphs and recording.layout is not None:
        paths, node_positions, roles = [], [], []
        for snapshot in range(n_snapshots):
            graph = recording.graph(snapshot, prune_threshold)
            paths += [
                np.hstack([np.full((len(path), 1), float(snapshot)), path])
                for path in graph.edge_paths
            ]
            degrees = graph.degrees
            node_positions.append(
                np.hstack(
                    [np.full((graph.n_nodes, 1), float(snapshot)), graph.node_positions]
                )
            )
            roles += [
                "end" if degree == 1 else "junction" if degree >= 3 else "other"
                for degree in degrees
            ]
        if paths:
            viewer.add_shapes(
                paths,
                shape_type="path",
                edge_color="orange",
                edge_width=0.3,
                name="fitted graph",
                visible=False,
            )
            viewer.add_points(
                np.vstack(node_positions),
                features={"role": roles},
                face_color=[_NODE_COLORS[role] for role in roles],
                size=2.0,
                name="fitted graph nodes",
                visible=False,
            )

    if recording.final_graph is not None:
        add_graph(viewer, recording.final_graph, name="final graph", color="cyan")
        for layer in viewer.layers[-2:]:
            layer.visible = False
    if recording.ground_truth is not None:
        add_graph(viewer, recording.ground_truth, name="ground truth", color="red")
        for layer in viewer.layers[-2:]:
            layer.visible = False

    viewer.dims.axis_labels = ("snapshot", *("z", "y", "x")[-ndim:])
    viewer.text_overlay.visible = True
    viewer.text_overlay.position = "top_left"
    viewer.text_overlay.font_size = 10

    def describe(snapshot: int) -> str:
        iteration = int(recording.iterations[snapshot])
        alive = int((recording.opacity[snapshot] >= prune_threshold).sum())
        total = int((~np.isnan(recording.opacity[snapshot])).sum())
        text = f"iteration {iteration}   alive segments {alive}/{total}"
        if recording.loss:
            # the loss at a snapshot is the loss of the step taken from it
            step = min(iteration, len(next(iter(recording.loss.values()))) - 1)
            terms = "   ".join(
                f"{name} {values[step]:.4f}" for name, values in recording.loss.items()
            )
            text += f"\n{terms}"
        return text

    def update(event: object = None) -> None:
        snapshot = int(np.clip(round(viewer.dims.current_step[0]), 0, n_snapshots - 1))
        viewer.text_overlay.text = describe(snapshot)

    viewer.dims.events.current_step.connect(update)
    viewer.dims.set_current_step(0, 0)
    update()
    return viewer
