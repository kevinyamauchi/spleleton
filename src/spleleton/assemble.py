"""Assemble independently fitted segments into a skeleton graph (Phase 5).

Segments fitted from seeds have no graph structure: each is free. Assembly:

1. drop pruned segments (opacity below a threshold);
2. drop segments thinner than ``min_radius`` (collapsed while fitting noise)
   and *contained* segments: duplicates lying inside the tubes of longer
   segments that run along them (both ends and the midpoint each beside a
   roughly parallel one, within ``contained`` radii of its axis);
3. cluster endpoints that are within ``join`` radii (the larger of the two)
   of each other into nodes;
4. attach free ends that touch the middle of another segment (T-junctions):
   the other segment is split there and the end joins the new node;
5. contract edges shorter than ``min_edge`` radii, and drop self-loops and
   duplicate edges this creates;
6. merge degree-2 nodes, so chains of segments become single edges.

The result is usually cleaned up further with
:func:`spleleton.postprocess.clean_graph`.
"""

from typing import Any

import numpy as np

from spleleton.graph import SkeletonGraph
from spleleton.postprocess import merge_degree_two_nodes
from spleleton.primitives._arrays import as_segment_arrays


def assemble_graph(
    segments: Any,
    prune_threshold: float = 0.1,
    contained: float = 0.5,
    join: float = 1.0,
    min_edge: float = 0.5,
    min_radius: float = 0.75,
    parallel: float = 0.7,
) -> SkeletonGraph:
    """Turn a set of independent segments into a graph.

    Parameters
    ----------
    segments : SegmentArrays, TorchLineSegments or JaxLineSegments
        Fitted segments.
    prune_threshold : float
        Segments with a lower opacity are dropped.
    contained : float
        A segment is redundant if both ends and its midpoint lie within this
        many radii of the centreline of longer, roughly parallel segments
        (see ``parallel``).
    join : float
        Endpoints closer than this many radii (the larger of the two) are
        joined, and so is a free end this close to another segment's
        interior.
    min_edge : float
        Edges shorter than this many radii are contracted.
    min_radius : float
        Segments with a smaller mean radius are dropped: they are not tubes
        but segments whose radius collapsed while fitting noise.
    parallel : float
        Only segments whose directions have an absolute cosine of at least
        this cover each other's probes for ``contained``, so a crossing tube
        or a parent meeting a child at its junction covers nothing.

    Returns
    -------
    SkeletonGraph
        The assembled graph, with radii.
    """
    arrays = as_segment_arrays(segments)
    start, end, radius = arrays.start, arrays.end, arrays.radius
    alive = arrays.opacity >= prune_threshold
    alive &= radius.mean(axis=1) >= min_radius
    keep = _drop_contained(start, end, radius, alive, contained, parallel)
    start, end, radius = start[keep], end[keep], radius[keep]
    n = len(start)
    ndim = arrays.ndim
    if n == 0:
        return SkeletonGraph(np.empty((0, ndim)), np.empty((0, 2)), [], [])

    # 3. cluster endpoints: endpoint i is start of segment i, n + i its end
    points = np.vstack([start, end])
    radii = np.concatenate([radius[:, 0], radius[:, 1]])
    parent = list(range(2 * n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    distance = np.linalg.norm(points[:, None] - points[None], axis=-1)
    limit = join * np.maximum(radii[:, None], radii[None, :])
    for i, j in zip(*np.nonzero(np.triu(distance < limit, k=1)), strict=True):
        if i % n != j % n:
            parent[find(int(i))] = find(int(j))
    roots = sorted({find(i) for i in range(2 * n)})
    node_of = {root: index for index, root in enumerate(roots)}
    endpoint_node = np.array([node_of[find(i)] for i in range(2 * n)])
    nodes = [points[endpoint_node == k].mean(axis=0) for k in range(len(roots))]
    node_radius = [radii[endpoint_node == k].max() for k in range(len(roots))]
    edges = [(int(endpoint_node[i]), int(endpoint_node[n + i])) for i in range(n)]
    edge_radii = [radius[i].copy() for i in range(n)]

    # 4. T-junctions: free ends touching another edge's interior
    changed = True
    while changed:
        changed = False
        degrees = np.bincount(np.array(edges).ravel(), minlength=len(nodes))
        for node in np.flatnonzero(degrees == 1):
            point = nodes[node]
            for index, (u, v) in enumerate(edges):
                if node in (u, v) or u == v:
                    continue
                a, b = nodes[u], nodes[v]
                vector = b - a
                length = float(np.linalg.norm(vector))
                if length == 0:
                    continue
                t = float(np.clip((point - a) @ vector / length**2, 0, 1))
                along = t * length
                r = edge_radii[index][0] + t * (
                    edge_radii[index][1] - edge_radii[index][0]
                )
                if not (r < along < length - r):
                    continue  # near an end: endpoint clustering handles it
                if np.linalg.norm(a + t * vector - point) > join * max(
                    r, node_radius[node]
                ):
                    continue
                # split the edge at the projection and move the free end there
                projection = a + t * vector
                nodes[node] = projection
                edges[index] = (u, int(node))
                edges.append((int(node), v))
                edge_radii.append(np.array([r, edge_radii[index][1]]))
                edge_radii[index] = np.array([edge_radii[index][0], r])
                node_radius[node] = max(node_radius[node], r)
                changed = True
                break
            if changed:
                break

    # 5. contract short edges, then drop self-loops and duplicates
    nodes, edges, edge_radii = _contract_short(
        nodes, node_radius, edges, edge_radii, min_edge
    )
    seen = set()
    final_edges, final_paths, final_radii = [], [], []
    for (u, v), r in zip(edges, edge_radii, strict=True):
        key = (min(u, v), max(u, v))
        if u == v or key in seen:
            continue
        seen.add(key)
        final_edges.append((u, v))
        final_paths.append(np.stack([nodes[u], nodes[v]]))
        final_radii.append(r)
    used = sorted({node for edge in final_edges for node in edge})
    new_index = {node: index for index, node in enumerate(used)}
    graph = SkeletonGraph(
        node_positions=np.array([nodes[i] for i in used]).reshape(-1, ndim),
        edges=np.array(
            [(new_index[u], new_index[v]) for u, v in final_edges], dtype=np.int64
        ).reshape(-1, 2),
        edge_paths=final_paths,
        edge_radii=final_radii,
    )
    return merge_degree_two_nodes(graph)


def _drop_contained(
    start: np.ndarray,
    end: np.ndarray,
    radius: np.ndarray,
    alive: np.ndarray,
    contained: float,
    parallel: float = 0.7,
) -> np.ndarray:
    """Indices of alive segments not contained in the longer alive segments.

    Segments are visited from longest to shortest. A segment is contained if
    each of its probes (both ends and the midpoint) is covered by *some* kept
    segment, so a chain of short duplicates running inside other tubes is
    dropped too, not only a segment inside a single one. A kept segment
    covers a probe that lies beside it (the probe's projection falls on the
    segment, not beyond an end) within ``contained`` radii of its axis, and
    only if the two segments run roughly parallel (``|cos| >= parallel``).
    Otherwise a tube crossing a segment's midpoint, or the ends of children
    meeting a parent at a junction, would drop it, and break the graph there.
    """
    lengths = np.linalg.norm(end - start, axis=1)
    # a point-like segment has no direction: it is parallel to everything
    directions = (end - start) / np.maximum(lengths, 1e-12)[:, None]
    directions[lengths < 1e-9] = np.nan
    keep: list[int] = []
    order = [i for i in np.argsort(-lengths, kind="stable") if alive[i]]
    for i in order:
        probes = np.stack([start[i], 0.5 * (start[i] + end[i]), end[i]])
        covered = np.zeros(len(probes), dtype=bool)
        for j in keep:
            if abs(float(directions[i] @ directions[j])) < parallel:  # NaN: never
                continue
            vector = end[j] - start[j]
            squared = max(float(vector @ vector), 1e-12)
            t = (probes - start[j]) @ vector / squared
            beside = (t >= 0) & (t <= 1)
            t = np.clip(t, 0, 1)
            distance = np.linalg.norm(start[j] + t[:, None] * vector - probes, axis=1)
            r = radius[j, 0] + t * (radius[j, 1] - radius[j, 0])
            covered |= beside & (distance <= contained * r)
            if covered.all():
                break
        if not covered.all():
            keep.append(i)
    return np.array(sorted(keep), dtype=int)


def _contract_short(
    nodes: list[np.ndarray],
    node_radius: list[float],
    edges: list[tuple[int, int]],
    edge_radii: list[np.ndarray],
    min_edge: float,
) -> tuple[list[np.ndarray], list[tuple[int, int]], list[np.ndarray]]:
    """Merge the two nodes of every edge shorter than ``min_edge`` radii."""
    parent = list(range(len(nodes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for (u, v), r in zip(edges, edge_radii, strict=True):
        if np.linalg.norm(nodes[u] - nodes[v]) < min_edge * float(np.mean(r)):
            parent[find(u)] = find(v)
    groups: dict[int, list[int]] = {}
    for i in range(len(nodes)):
        groups.setdefault(find(i), []).append(i)
    merged = list(nodes)
    for members in groups.values():
        weights = np.array([node_radius[i] for i in members])
        centre = (np.array([nodes[i] for i in members]) * weights[:, None]).sum(
            0
        ) / weights.sum()
        for i in members:
            merged[i] = centre
    root = [find(i) for i in range(len(nodes))]
    return merged, [(root[u], root[v]) for u, v in edges], edge_radii
