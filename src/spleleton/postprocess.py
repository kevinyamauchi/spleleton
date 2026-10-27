"""Clean-up of skeleton graphs: spur pruning, break bridging and simplification.

All functions take a :class:`~spleleton.graph.SkeletonGraph` with radii and
return a new one; none of them modifies its input. Lengths are in units of the
local tube radius, so the same settings work for thin and thick branches.
"""

from dataclasses import dataclass

import numpy as np

from spleleton.graph import SkeletonGraph


@dataclass
class _Editable:
    """A mutable copy of a graph, for local edits."""

    nodes: list[np.ndarray]
    edges: list[tuple[int, int]]
    paths: list[np.ndarray]
    radii: list[np.ndarray]

    @classmethod
    def from_graph(cls, graph: SkeletonGraph) -> "_Editable":
        if graph.edge_radii is None:
            raise ValueError("clean-up needs a graph with radii")
        return cls(
            nodes=list(graph.node_positions),
            edges=[(int(u), int(v)) for u, v in graph.edges],
            paths=[path.copy() for path in graph.edge_paths],
            radii=[radii.copy() for radii in graph.edge_radii],
        )

    def degrees(self) -> np.ndarray:
        degrees = np.zeros(len(self.nodes), dtype=int)
        for u, v in self.edges:
            degrees[u] += 1
            degrees[v] += 1
        return degrees

    def incident(self, node: int) -> list[int]:
        return [i for i, (u, v) in enumerate(self.edges) if node in (u, v)]

    def remove_edges(self, indices: set[int]) -> None:
        keep = [i for i in range(len(self.edges)) if i not in indices]
        self.edges = [self.edges[i] for i in keep]
        self.paths = [self.paths[i] for i in keep]
        self.radii = [self.radii[i] for i in keep]

    def oriented(self, edge: int, start: int) -> tuple[np.ndarray, np.ndarray]:
        """An edge's path and radii, oriented to start at the given node."""
        u, _ = self.edges[edge]
        if u == start:
            return self.paths[edge], self.radii[edge]
        return self.paths[edge][::-1], self.radii[edge][::-1]

    def to_graph(self, ndim: int) -> SkeletonGraph:
        """Build the graph, dropping nodes without edges."""
        used = sorted({node for edge in self.edges for node in edge})
        new_index = {node: index for index, node in enumerate(used)}
        return SkeletonGraph(
            node_positions=np.array([self.nodes[n] for n in used]).reshape(-1, ndim),
            edges=np.array(
                [(new_index[u], new_index[v]) for u, v in self.edges], dtype=np.int64
            ).reshape(-1, 2),
            edge_paths=self.paths,
            edge_radii=self.radii,
        )


def merge_degree_two_nodes(graph: SkeletonGraph) -> SkeletonGraph:
    """Join the two edges at every node of degree 2 into one edge.

    A ring made only of degree-2 nodes is reduced to a single self-loop.
    """
    work = _Editable.from_graph(graph)
    merged = True
    while merged:
        merged = False
        for node in np.flatnonzero(work.degrees() == 2):
            node = int(node)
            incident = work.incident(node)
            if len(incident) != 2:
                continue  # a self-loop
            first, second = incident
            path_in, radii_in = work.oriented(first, node)
            path_out, radii_out = work.oriented(second, node)
            start = _other_end(work.edges[first], node)
            end = _other_end(work.edges[second], node)
            work.remove_edges({first, second})
            work.edges.append((start, end))
            work.paths.append(np.vstack([path_in[::-1], path_out[1:]]))
            work.radii.append(np.concatenate([radii_in[::-1], radii_out[1:]]))
            merged = True
            break
    return work.to_graph(graph.ndim)


def _other_end(edge: tuple[int, int], node: int) -> int:
    u, v = edge
    return v if u == node else u


def prune_short_leaves(
    graph: SkeletonGraph, min_protrusion: float = 1.5
) -> SkeletonGraph:
    """Remove leaf branches that barely stick out of their parent tube.

    A leaf edge (one end of degree 1, the other a junction of degree 3 or
    more) protrudes by the length of its centreline that lies outside the
    tubes of all other edges. If that is less than ``min_protrusion`` times
    the leaf's median radius, the leaf is a spur and is removed. All such
    leaves are removed together, then degree-2 nodes are merged, and this
    repeats until nothing changes; removing leaves never disconnects the
    graph.

    Measuring outside the other tubes, rather than as the leaf's length minus
    its junction's radius, also works when the leaf does not start on its
    parent's axis, e.g. when it attaches at a crowded junction away from the
    bump that caused it.

    Removing them together matters for the small fork that skeletonisation
    often leaves at a flat branch end: both twigs go at once, instead of one
    twig surviving as the branch's continuation.

    Parameters
    ----------
    graph : SkeletonGraph
        Graph with radii.
    min_protrusion : float
        Minimum protrusion of a leaf, in units of its radius.

    Returns
    -------
    SkeletonGraph
        The pruned graph, with degree-2 nodes merged.
    """
    while True:
        work = _Editable.from_graph(graph)
        degrees = work.degrees()
        spurs = set()
        for index, (u, v) in enumerate(work.edges):
            if u == v or (degrees[u] == 1) == (degrees[v] == 1):
                continue
            junction = v if degrees[u] == 1 else u
            if degrees[junction] < 3:
                continue
            radii = work.radii[index]
            protrusion = _length_outside(work, index)
            if protrusion < min_protrusion * float(np.median(radii)):
                spurs.add(index)
        if not spurs:
            return merge_degree_two_nodes(graph)
        work.remove_edges(spurs)
        graph = merge_degree_two_nodes(work.to_graph(graph.ndim))


def _length_outside(work: _Editable, index: int, spacing: float = 0.25) -> float:
    """Length of an edge's centreline outside the tubes of all other edges."""
    from spleleton.graph import resample_polyline

    points, _ = resample_polyline(work.paths[index], spacing)
    outside = np.ones(len(points), dtype=bool)
    for other, (path, radii) in enumerate(zip(work.paths, work.radii, strict=True)):
        if other == index:
            continue
        start, end = path[:-1], path[1:]
        vector = end - start
        squared = np.maximum((vector * vector).sum(axis=1), 1e-12)
        offset = points[:, None, :] - start[None]
        t = np.clip((offset * vector[None]).sum(axis=-1) / squared[None], 0.0, 1.0)
        closest = start[None] + t[..., None] * vector[None]
        distance = np.linalg.norm(points[:, None, :] - closest, axis=-1)
        radius = radii[:-1][None] + t * (radii[1:] - radii[:-1])[None]
        outside &= ~np.any(distance <= radius, axis=1)
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    # a step counts as outside if both of its ends are
    return float(steps[outside[:-1] & outside[1:]].sum())


@dataclass(frozen=True)
class Bridge:
    """A bridge added across a break.

    Attributes
    ----------
    start : np.ndarray
        The free end the bridge starts from.
    end : np.ndarray
        The point on the other component it joins (a node, or a new node
        inside an edge).
    gap : float
        Surface-to-surface gap between the two tubes.
    """

    start: np.ndarray
    end: np.ndarray
    gap: float


def bridge_breaks(
    graph: SkeletonGraph,
    reach: float = 2.0,
    max_angle: float = 30.0,
    tangent_length: float = 2.0,
) -> tuple[SkeletonGraph, list[Bridge]]:
    """Join free ends to other components that they point at across a gap.

    For every free end ``a`` (a node of degree 1), the candidate target on
    each edge of another connected component is the closest point ``q`` of
    that edge's centreline. The pair is a candidate bridge if

    - the gap between the tubes, ``|q - p_a| - r_a - r_q``, is at most
      ``reach`` times their mean radius, and
    - the free end's outward tangent points at ``q`` within ``max_angle``
      degrees, or within 90 degrees if the two tubes already touch (gap of
      at most zero), since the mask itself joins them there.

    The target can be another free end, a junction, or a point inside an
    edge, which is then split into a new junction. This matters when a branch
    is cut next to its parent's junction: the parent's stump points straight
    on, while the child leaves at an angle, so only the child's end points at
    the parent.

    Bridges are added one at a time, smallest gap first, and only between
    different components, so bridging never creates a cycle. A free end's
    radius is the median radius over the last ``tangent_length`` radii of its
    edge, because the distance transform right at a cut face measures the
    distance to the cut, not to the tube wall. Its tangent is the direction
    from the point ``tangent_length`` radii back along the edge to the end.

    Parameters
    ----------
    graph : SkeletonGraph
        Graph with radii.
    reach : float
        Largest bridged gap, in units of the mean radius of the two tubes.
    max_angle : float
        Largest angle between the free end's tangent and the bridge, in
        degrees.
    tangent_length : float
        How far back along the edge the tangent and radius are measured, in
        radii.

    Returns
    -------
    graph : SkeletonGraph
        The graph with bridges added and degree-2 nodes merged.
    bridges : list of Bridge
        The bridges that were added.
    """
    cos_limit = np.cos(np.deg2rad(max_angle))
    work = _Editable.from_graph(graph)
    bridges = []
    while True:
        component = _components(len(work.nodes), work.edges)
        best = None
        for end in _free_ends(work, tangent_length):
            node, point, tangent, radius = end
            for index, (u, _) in enumerate(work.edges):
                if component[u] == component[node]:
                    continue
                target, target_radius, piece, t = _closest_on_path(
                    work.paths[index], work.radii[index], point
                )
                vector = target - point
                distance = float(np.linalg.norm(vector))
                gap = distance - radius - target_radius
                if gap > reach * 0.5 * (radius + target_radius):
                    continue
                # tubes that already touch are joined in the mask itself, so
                # the end only has to point roughly towards the target
                limit = cos_limit if gap > 0 else 0.0
                if distance > 0 and tangent @ vector / distance < limit:
                    continue
                if best is None or gap < best[0]:
                    best = (gap, node, radius, index, piece, t, target, target_radius)
        if best is None:
            break
        gap, node, radius, index, piece, t, target, target_radius = best
        target_node = _split_edge(work, index, piece, t, target, target_radius)
        work.edges.append((node, target_node))
        work.paths.append(np.stack([work.nodes[node], work.nodes[target_node]]))
        work.radii.append(np.array([radius, target_radius]))
        bridges.append(
            Bridge(work.nodes[node].copy(), work.nodes[target_node].copy(), gap)
        )
    return merge_degree_two_nodes(work.to_graph(graph.ndim)), bridges


def _free_ends(
    work: _Editable, tangent_length: float
) -> list[tuple[int, np.ndarray, np.ndarray, float]]:
    """Node, position, outward unit tangent and radius of every free end."""
    ends = []
    for node in np.flatnonzero(work.degrees() == 1):
        (edge,) = work.incident(int(node))
        path, radii = work.oriented(edge, int(node))
        steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(steps)])
        near = arc <= tangent_length * float(np.median(radii))
        radius = float(np.median(radii[near]))
        tangent = path[0] - _point_along(path, tangent_length * radius)
        norm = float(np.linalg.norm(tangent))
        if norm > 0:
            ends.append((int(node), path[0], tangent / norm, radius))
    return ends


def _closest_on_path(
    path: np.ndarray, radii: np.ndarray, point: np.ndarray
) -> tuple[np.ndarray, float, int, float]:
    """Closest point on a polyline: position, radius, piece index and parameter."""
    start, end = path[:-1], path[1:]
    vector = end - start
    squared = np.maximum((vector * vector).sum(axis=1), 1e-12)
    t = np.clip(((point - start) * vector).sum(axis=1) / squared, 0.0, 1.0)
    closest = start + t[:, None] * vector
    piece = int(np.argmin(np.linalg.norm(closest - point, axis=1)))
    radius = float(radii[piece] + t[piece] * (radii[piece + 1] - radii[piece]))
    return closest[piece], radius, piece, float(t[piece])


def _split_edge(
    work: _Editable,
    index: int,
    piece: int,
    t: float,
    point: np.ndarray,
    radius: float,
) -> int:
    """Insert a node into an edge at a point of one of its pieces.

    Returns the node at the point: an existing end node if the point is at
    one of the edge's ends, otherwise a new node.
    """
    u, v = work.edges[index]
    path, radii = work.paths[index], work.radii[index]
    if piece == 0 and t <= 1e-6:
        return u
    if piece == len(path) - 2 and t >= 1 - 1e-6:
        return v
    node = len(work.nodes)
    work.nodes.append(np.asarray(point, dtype=float))
    first_path = np.vstack([path[: piece + 1], point])
    second_path = np.vstack([point, path[piece + 1 :]])
    first_radii = np.concatenate([radii[: piece + 1], [radius]])
    second_radii = np.concatenate([[radius], radii[piece + 1 :]])
    # drop a duplicated vertex when the point coincides with one
    if np.allclose(first_path[-1], first_path[-2]):
        first_path, first_radii = first_path[:-1], first_radii[:-1]
        first_path[-1] = point
    if np.allclose(second_path[0], second_path[1]):
        second_path, second_radii = second_path[1:], second_radii[1:]
        second_path[0] = point
    work.remove_edges({index})
    work.edges += [(u, node), (node, v)]
    work.paths += [first_path, second_path]
    work.radii += [first_radii, second_radii]
    return node


def drop_small_components(
    graph: SkeletonGraph, min_length: float = 3.0
) -> SkeletonGraph:
    """Remove connected components that are too small to be a branch.

    A component is dropped if its total centreline length is less than
    ``min_length`` times its median radius. Such islands come from bumps or
    noise that the fit covered with a segment of its own but that nothing
    joined to the tree (not even a bridge).

    Parameters
    ----------
    graph : SkeletonGraph
        Graph with radii.
    min_length : float
        Minimum total length of a component, in units of its median radius.

    Returns
    -------
    SkeletonGraph
        The graph without the small components.
    """
    work = _Editable.from_graph(graph)
    component = _components(len(work.nodes), work.edges)
    lengths = [
        float(np.linalg.norm(np.diff(path, axis=0), axis=1).sum())
        for path in work.paths
    ]
    drop = set()
    for label in np.unique(component):
        members = [i for i, (u, _) in enumerate(work.edges) if component[u] == label]
        if not members:
            continue
        radius = float(np.median(np.concatenate([work.radii[i] for i in members])))
        if sum(lengths[i] for i in members) < min_length * radius:
            drop.update(members)
    work.remove_edges(drop)
    return work.to_graph(graph.ndim)


def clean_graph(
    graph: SkeletonGraph,
    min_protrusion: float = 1.5,
    bridge_reach: float = 2.0,
    bridge_angle: float = 30.0,
    min_island: float = 3.0,
) -> SkeletonGraph:
    """Bridge breaks, prune spurs, bridge again, and drop small islands.

    The order matters (measured on synthetic 2D trees with gaps):

    - bridging first reconnects the stub that remains when a branch is cut
      next to its junction; pruned first, that stub looks like a spur;
    - the second bridging pass joins ends that pruning exposed, e.g. a branch
      end whose small fork was a spur;
    - islands are dropped last, so that every fragment had its chance to be
      bridged.
    """
    graph, _ = bridge_breaks(graph, bridge_reach, bridge_angle)
    graph = prune_short_leaves(graph, min_protrusion)
    graph, _ = bridge_breaks(graph, bridge_reach, bridge_angle)
    return drop_small_components(graph, min_island)


def _point_along(path: np.ndarray, distance: float) -> np.ndarray:
    """The point at a given arc length from the start of a path (or its end)."""
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    if distance >= arc[-1]:
        return path[-1]
    index = int(np.searchsorted(arc, distance, side="right")) - 1
    fraction = (distance - arc[index]) / max(steps[index], 1e-12)
    return path[index] + fraction * (path[index + 1] - path[index])


def _components(n_nodes: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Connected component label of each node."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    if not edges:
        return np.arange(n_nodes)
    rows, columns = np.array(edges).T
    adjacency = coo_matrix(
        (np.ones(len(edges)), (rows, columns)), shape=(n_nodes, n_nodes)
    ).tocsr()
    return connected_components(adjacency, directed=False)[1]
