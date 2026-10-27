"""Classical baseline: morphological skeletonisation, converted to a graph."""

import itertools

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize

from spleleton.graph import SkeletonGraph


def skeletonize_to_graph(
    mask: np.ndarray, spacing: tuple[float, ...] | None = None
) -> SkeletonGraph:
    """Skeletonise a mask with scikit-image and convert it to a graph.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask, 2D or 3D.
    spacing : tuple of float, optional
        Physical voxel size per axis, used for coordinates and radii. The
        thinning itself ignores it.

    Returns
    -------
    SkeletonGraph
        The skeleton graph, with radii from the distance transform.
    """
    mask = np.asarray(mask, dtype=bool)
    return skeleton_to_graph(skeletonize(mask), spacing=spacing, mask=mask)


def skeleton_to_graph(
    skeleton: np.ndarray,
    spacing: tuple[float, ...] | None = None,
    mask: np.ndarray | None = None,
) -> SkeletonGraph:
    """Convert a thin binary skeleton into a graph.

    Voxels are connected to all neighbours (8-connectivity in 2D,
    26-connectivity in 3D). A diagonal link is dropped when the same two
    voxels are also joined through a common neighbour by shorter steps, so a
    staircase is a simple path rather than a chain of tiny triangles.

    After that, voxels with exactly two neighbours are path voxels. All other
    voxels are nodes: ends (one neighbour), isolated voxels (none) and
    junction voxels (three or more). Adjacent junction voxels are merged into
    one node at their centroid. Rings without any node become a self-loop on
    one of their voxels.

    Parameters
    ----------
    skeleton : np.ndarray
        Binary skeleton, 2D or 3D.
    spacing : tuple of float, optional
        Physical voxel size per axis. Defaults to 1 on every axis.
    mask : np.ndarray, optional
        The mask the skeleton came from. If given, radii are read from its
        Euclidean distance transform.

    Returns
    -------
    SkeletonGraph
        The graph, in physical coordinates.
    """
    skeleton = np.asarray(skeleton, dtype=bool)
    ndim = skeleton.ndim
    spacing_array = np.ones(ndim) if spacing is None else np.asarray(spacing, float)
    coords = np.argwhere(skeleton)
    positions = coords * spacing_array
    if mask is not None:
        distance = ndimage.distance_transform_edt(mask, sampling=spacing_array)
        voxel_radii = distance[tuple(coords.T)]
    neighbours = _neighbours(skeleton, coords)
    degrees = np.array([len(n) for n in neighbours])

    # merge adjacent junction voxels into one node; every other node voxel
    # (an end or an isolated voxel) is a node of its own
    cluster = np.full(len(coords), -1)
    node_voxels: list[list[int]] = []
    for voxel in np.flatnonzero(degrees != 2):
        if cluster[voxel] >= 0:
            continue
        members = [voxel]
        cluster[voxel] = len(node_voxels)
        if degrees[voxel] >= 3:
            stack = [voxel]
            while stack:
                for other in neighbours[stack.pop()]:
                    if degrees[other] >= 3 and cluster[other] < 0:
                        cluster[other] = len(node_voxels)
                        members.append(other)
                        stack.append(other)
        node_voxels.append(members)

    node_positions = [positions[members].mean(axis=0) for members in node_voxels]
    node_radii = (
        [voxel_radii[members].max() for members in node_voxels]
        if mask is not None
        else None
    )

    edges, paths, radii = [], [], []
    visited_steps: set[tuple[int, int]] = set()
    visited_path_voxels = np.zeros(len(coords), dtype=bool)

    def trace(start_voxel: int, first_step: int) -> tuple[int, list[int]]:
        """Walk through path voxels until the next node voxel."""
        previous, current = start_voxel, first_step
        interior = []
        while cluster[current] < 0:
            visited_path_voxels[current] = True
            interior.append(current)
            following = [n for n in neighbours[current] if n != previous]
            if not following:  # pragma: no cover - path voxels have two neighbours
                break
            previous, current = current, following[0]
            if current == first_step:  # a ring without nodes
                break
        visited_steps.add((current, previous))
        return current, interior

    def add_edge(start_node: int, end_node: int, interior: list[int]) -> None:
        edges.append((start_node, end_node))
        paths.append(
            np.vstack(
                [
                    node_positions[start_node],
                    positions[interior],
                    node_positions[end_node],
                ]
            )
        )
        if node_radii is not None:
            radii.append(
                np.concatenate(
                    [
                        [node_radii[start_node]],
                        voxel_radii[interior],
                        [node_radii[end_node]],
                    ]
                )
            )

    for node, members in enumerate(node_voxels):
        for voxel in members:
            for step in neighbours[voxel]:
                if cluster[step] == node or (voxel, step) in visited_steps:
                    continue
                visited_steps.add((voxel, step))
                end_voxel, interior = trace(voxel, step)
                end_node = int(cluster[end_voxel])
                # a tiny loop back into the same junction is part of the junction
                if end_node == node and len(interior) < 3:
                    continue
                add_edge(node, end_node, interior)

    # rings of path voxels that no node touches
    for voxel in np.flatnonzero((degrees == 2) & ~visited_path_voxels):
        if visited_path_voxels[voxel]:
            continue
        node = len(node_voxels)
        node_voxels.append([voxel])
        cluster[voxel] = node
        node_positions.append(positions[voxel])
        if node_radii is not None:
            node_radii.append(voxel_radii[voxel])
        visited_path_voxels[voxel] = True
        _, interior = trace(voxel, neighbours[voxel][0])
        add_edge(node, node, interior)

    return SkeletonGraph(
        node_positions=np.array(node_positions).reshape(-1, ndim),
        edges=np.array(edges, dtype=np.int64).reshape(-1, 2),
        edge_paths=paths,
        edge_radii=radii if node_radii is not None else None,
    )


def _neighbours(skeleton: np.ndarray, coords: np.ndarray) -> list[list[int]]:
    """Neighbour lists of skeleton voxels, without redundant diagonal links."""
    ndim = skeleton.ndim
    index = np.full(skeleton.shape, -1, dtype=np.int64)
    index[tuple(coords.T)] = np.arange(len(coords))
    offsets = [
        np.array(offset)
        for offset in itertools.product((-1, 0, 1), repeat=ndim)
        if any(offset)
    ]
    neighbours: list[set[int]] = [set() for _ in range(len(coords))]
    shape = np.array(skeleton.shape)
    for offset in offsets:
        target = coords + offset
        inside = np.all((target >= 0) & (target < shape), axis=1)
        other = np.full(len(coords), -1)
        other[inside] = index[tuple(target[inside].T)]
        for voxel in np.flatnonzero(other >= 0):
            neighbours[voxel].add(int(other[voxel]))

    def steps(a: int, b: int) -> int:
        return int(np.count_nonzero(coords[a] != coords[b]))

    redundant = [
        (a, b)
        for a in range(len(coords))
        for b in neighbours[a]
        if a < b
        and steps(a, b) > 1
        and any(
            steps(a, c) < steps(a, b) and steps(c, b) < steps(a, b)
            for c in neighbours[a] & neighbours[b]
        )
    ]
    for a, b in redundant:
        neighbours[a].discard(b)
        neighbours[b].discard(a)
    return [sorted(n) for n in neighbours]
