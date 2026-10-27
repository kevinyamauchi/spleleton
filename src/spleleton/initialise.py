"""Turn skeleton graphs into line segments for fitting, and fitted segments back.

The segments are optimised independently (topology option (b) of the plan), so
the graph structure lives outside them, in a :class:`SegmentLayout`: which
chain of segments makes up each edge, and which nodes each edge joins. After
fitting, :meth:`SegmentLayout.to_graph` snaps the chains back together into a
graph with the same topology, minus the edges whose segments were pruned.
"""

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np

from spleleton.graph import SkeletonGraph, simplify_polyline
from spleleton.primitives._arrays import SegmentArrays, as_segment_arrays


@dataclass(frozen=True)
class SegmentLayout:
    """How segments make up the edges of a graph.

    Attributes
    ----------
    n_nodes : int
        Number of nodes.
    edges : np.ndarray
        Node indices of each edge, shape ``(E, 2)``.
    edge_segments : list of np.ndarray
        For each edge, the indices of its segments in order: the first starts
        at the edge's first node, each one starts where the previous ends, and
        the last ends at the edge's second node.
    """

    n_nodes: int
    edges: np.ndarray
    edge_segments: list[np.ndarray]

    def to_dict(self) -> dict[str, np.ndarray]:
        """Flatten into a dictionary of arrays, e.g. for ``np.savez``."""
        lengths = [len(chain) for chain in self.edge_segments]
        return {
            "n_nodes": np.array(self.n_nodes),
            "edges": np.asarray(self.edges, dtype=np.int64).reshape(-1, 2),
            "segments": (
                np.concatenate(self.edge_segments).astype(np.int64)
                if self.edge_segments
                else np.empty(0, dtype=np.int64)
            ),
            "offsets": np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        }

    @classmethod
    def from_dict(cls, data: dict[str, np.ndarray]) -> "SegmentLayout":
        """Rebuild a layout from the output of :meth:`to_dict`."""
        offsets = data["offsets"]
        return cls(
            n_nodes=int(data["n_nodes"]),
            edges=np.asarray(data["edges"]).reshape(-1, 2),
            edge_segments=[
                np.asarray(data["segments"][start:stop])
                for start, stop in itertools.pairwise(offsets)
            ],
        )

    def to_graph(self, segments: Any, prune_threshold: float = 0.1) -> SkeletonGraph:
        """Snap fitted segments back into a graph with this layout's topology.

        An edge is *dead* if more than half of its segments have an opacity
        below ``prune_threshold``. Pruning never disconnects the skeleton:
        a dead edge is removed only if removing it keeps its two nodes
        connected (an edge on a cycle) or leaves one of them with no other
        edge (a spur). This is repeated, so a chain of dead edges hanging off
        the tree is removed from its tip inwards. A dead edge that holds the
        tree together is kept, because an internal edge can lose all its
        voxels to its neighbours (e.g. a short edge between two nearby
        junctions) without being wrong.

        Each node is placed at the mean of the segment ends that meet there,
        and each joint inside an edge at the mean of the two segment ends that
        meet there; radii are averaged the same way. Nodes left without edges
        are dropped.

        Parameters
        ----------
        segments : SegmentArrays, TorchLineSegments or JaxLineSegments
            The fitted segments.
        prune_threshold : float
            Opacity below which a segment counts as pruned.

        Returns
        -------
        SkeletonGraph
            The fitted graph, with radii.
        """
        arrays = as_segment_arrays(segments)
        start, end, radius = arrays.start, arrays.end, arrays.radius
        alive = arrays.opacity >= prune_threshold

        removed = self._removable_dead_edges(
            [alive[chain].mean() < 0.5 for chain in self.edge_segments]
        )
        kept = [index for index in range(len(self.edges)) if index not in removed]
        ndim = start.shape[1]
        position_sum = np.zeros((self.n_nodes, ndim))
        radius_sum = np.zeros(self.n_nodes)
        count = np.zeros(self.n_nodes)
        for index in kept:
            (u, v), chain = self.edges[index], self.edge_segments[index]
            position_sum[u] += start[chain[0]]
            position_sum[v] += end[chain[-1]]
            radius_sum[u] += radius[chain[0], 0]
            radius_sum[v] += radius[chain[-1], 1]
            count[u] += 1
            count[v] += 1

        used = np.flatnonzero(count > 0)
        new_index = np.full(self.n_nodes, -1)
        new_index[used] = np.arange(len(used))
        node_positions = position_sum[used] / count[used, None]
        node_radii = radius_sum[used] / count[used]

        edges, paths, radii = [], [], []
        for index in kept:
            (u, v), chain = self.edges[index], self.edge_segments[index]
            joints = 0.5 * (end[chain[:-1]] + start[chain[1:]])
            joint_radii = 0.5 * (radius[chain[:-1], 1] + radius[chain[1:], 0])
            u, v = new_index[u], new_index[v]
            edges.append((u, v))
            paths.append(np.vstack([node_positions[u], joints, node_positions[v]]))
            radii.append(
                np.concatenate([[node_radii[u]], joint_radii, [node_radii[v]]])
            )
        return SkeletonGraph(
            node_positions=node_positions.reshape(-1, ndim),
            edges=np.array(edges, dtype=np.int64).reshape(-1, 2),
            edge_paths=paths,
            edge_radii=radii,
        )

    def _removable_dead_edges(self, dead: list[bool]) -> set[int]:
        """Dead edges whose removal does not disconnect the graph."""
        import networkx as nx

        graph = nx.MultiGraph()
        graph.add_nodes_from(range(self.n_nodes))
        for index, (u, v) in enumerate(self.edges):
            graph.add_edge(int(u), int(v), key=index)
        removed: set[int] = set()
        changed = True
        while changed:
            changed = False
            for index in np.flatnonzero(dead):
                if index in removed:
                    continue
                u, v = (int(node) for node in self.edges[index])
                graph.remove_edge(u, v, key=index)
                spur = graph.degree(u) == 0 or graph.degree(v) == 0
                if spur or u == v or nx.has_path(graph, u, v):
                    removed.add(int(index))
                    changed = True
                else:
                    graph.add_edge(u, v, key=index)
        return removed


def segments_from_graph(
    graph: SkeletonGraph,
    tolerance: float = 1.0,
    radius_offset: float = 0.0,
    min_radius: float = 0.5,
    opacity: float = 0.5,
) -> tuple[SegmentArrays, SegmentLayout]:
    """Initial segments from a (voxel) skeleton graph.

    Each edge's polyline is simplified with Douglas-Peucker, and every piece
    of the simplified polyline becomes one segment.

    Parameters
    ----------
    graph : SkeletonGraph
        Skeleton graph, e.g. from
        :func:`spleleton.baseline.skeletonize_to_graph`.
    tolerance : float
        Douglas-Peucker tolerance, in physical units.
    radius_offset : float
        Subtracted from the graph's radii. On the synthetic data, radii from
        the distance transform at skeleton voxels overestimate the true radius
        by only about 0.1 voxels, so the default is 0.
    min_radius : float
        Smallest initial radius.
    opacity : float
        Initial opacity of every segment.

    Returns
    -------
    segments : SegmentArrays
        The initial segments; convert them with ``TorchLineSegments.from_arrays``
        or ``JaxLineSegments.from_arrays`` to fit them.
    layout : SegmentLayout
        Which segments make up which edge.
    """
    starts, ends, radii, edge_segments = [], [], [], []
    for index, path in enumerate(graph.edge_paths):
        keep = simplify_polyline(path, tolerance)
        vertices = path[keep]
        if graph.edge_radii is None:
            vertex_radii = np.full(len(keep), 1.0)
        else:
            vertex_radii = graph.edge_radii[index][keep] - radius_offset
        vertex_radii = np.maximum(vertex_radii, min_radius)
        first = len(starts)
        starts.extend(vertices[:-1])
        ends.extend(vertices[1:])
        radii.extend(np.stack([vertex_radii[:-1], vertex_radii[1:]], axis=1))
        edge_segments.append(np.arange(first, len(starts)))

    def array(values: list, width: int) -> np.ndarray:
        return np.array(values, dtype=float).reshape(-1, width)

    segments = SegmentArrays(
        array(starts, graph.ndim),
        array(ends, graph.ndim),
        radius=array(radii, 2),
        opacity=opacity,
    )
    layout = SegmentLayout(
        n_nodes=graph.n_nodes, edges=graph.edges.copy(), edge_segments=edge_segments
    )
    return segments, layout
