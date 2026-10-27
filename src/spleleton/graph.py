"""A minimal skeleton graph: nodes, edges and a polyline path per edge.

The same container holds ground-truth trees, baseline skeletons and fitted
skeletons, so that metrics can compare any two of them.

Coordinates are physical: the voxel with index ``i`` has its centre at
``i * spacing``.
"""

import itertools
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import networkx as nx


@dataclass
class SkeletonGraph:
    """Nodes, edges, and a polyline (with optional radii) per edge.

    Parameters
    ----------
    node_positions : np.ndarray
        Node coordinates, shape ``(K, D)``.
    edges : np.ndarray
        Node indices of each edge, shape ``(E, 2)``. For generated trees the
        first node is the parent.
    edge_paths : list of np.ndarray
        One polyline per edge, each of shape ``(P, D)`` with ``P >= 2``. The
        first point is at the edge's first node and the last at its second
        node.
    edge_radii : list of np.ndarray, optional
        Tube radius at each polyline point, each of shape ``(P,)``.
    """

    node_positions: np.ndarray
    edges: np.ndarray
    edge_paths: list[np.ndarray]
    edge_radii: list[np.ndarray] | None = field(default=None)

    def __post_init__(self) -> None:
        """Convert inputs to arrays and check that they are consistent."""
        self.node_positions = np.asarray(self.node_positions, dtype=float)
        self.edges = np.asarray(self.edges, dtype=np.int64).reshape(-1, 2)
        self.edge_paths = [np.asarray(path, dtype=float) for path in self.edge_paths]
        if self.edge_radii is not None:
            self.edge_radii = [np.asarray(r, dtype=float) for r in self.edge_radii]
        self._validate()

    def _validate(self) -> None:
        if self.node_positions.ndim != 2 or self.node_positions.shape[1] not in (2, 3):
            raise ValueError(
                "node_positions must have shape (K, D) with D = 2 or 3, "
                f"got {self.node_positions.shape}"
            )
        if len(self.edge_paths) != self.n_edges:
            raise ValueError(
                f"expected {self.n_edges} edge paths, got {len(self.edge_paths)}"
            )
        if self.n_edges and (self.edges.min() < 0 or self.edges.max() >= self.n_nodes):
            raise ValueError("edges refer to nodes that do not exist")
        for index, (path, (u, v)) in enumerate(
            zip(self.edge_paths, self.edges, strict=True)
        ):
            if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != self.ndim:
                raise ValueError(
                    f"edge path {index} must have shape (P >= 2, {self.ndim}), "
                    f"got {path.shape}"
                )
            if not (
                np.allclose(path[0], self.node_positions[u])
                and np.allclose(path[-1], self.node_positions[v])
            ):
                raise ValueError(
                    f"edge path {index} does not start and end at its nodes"
                )
        if self.edge_radii is not None:
            if len(self.edge_radii) != self.n_edges:
                raise ValueError(
                    f"expected {self.n_edges} edge radii, got {len(self.edge_radii)}"
                )
            for index, (radii, path) in enumerate(
                zip(self.edge_radii, self.edge_paths, strict=True)
            ):
                if radii.shape != (path.shape[0],):
                    raise ValueError(
                        f"edge radii {index} must have shape ({path.shape[0]},), "
                        f"got {radii.shape}"
                    )

    @property
    def ndim(self) -> int:
        """Number of spatial dimensions."""
        return self.node_positions.shape[1]

    @property
    def n_nodes(self) -> int:
        """Number of nodes."""
        return self.node_positions.shape[0]

    @property
    def n_edges(self) -> int:
        """Number of edges."""
        return self.edges.shape[0]

    @property
    def degrees(self) -> np.ndarray:
        """Degree of each node, shape ``(K,)``. A self-loop counts twice."""
        return np.bincount(self.edges.ravel(), minlength=self.n_nodes)

    @property
    def end_nodes(self) -> np.ndarray:
        """Indices of nodes with degree 1."""
        return np.flatnonzero(self.degrees == 1)

    @property
    def junction_nodes(self) -> np.ndarray:
        """Indices of nodes with degree 3 or more."""
        return np.flatnonzero(self.degrees >= 3)

    @property
    def edge_lengths(self) -> np.ndarray:
        """Arc length of each edge's polyline, shape ``(E,)``."""
        return np.array([_polyline_length(path) for path in self.edge_paths])

    def node_radii(self) -> np.ndarray | None:
        """Radius at each node: the largest radius of the incident edge ends.

        Returns
        -------
        np.ndarray or None
            Shape ``(K,)``, NaN for isolated nodes, or ``None`` without radii.
        """
        if self.edge_radii is None:
            return None
        radii = np.full(self.n_nodes, np.nan)
        for (u, v), edge_radii in zip(self.edges, self.edge_radii, strict=True):
            radii[u] = np.fmax(radii[u], edge_radii[0])
            radii[v] = np.fmax(radii[v], edge_radii[-1])
        return radii

    def n_components(self) -> int:
        """Number of connected components, counting isolated nodes."""
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        adjacency = coo_matrix(
            (np.ones(self.n_edges), (self.edges[:, 0], self.edges[:, 1])),
            shape=(self.n_nodes, self.n_nodes),
        ).tocsr()
        n_components, _ = connected_components(adjacency, directed=False)
        return int(n_components)

    def sample_points(
        self, spacing: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
        """Resample every edge path at (at most) a fixed arc-length spacing.

        Parameters
        ----------
        spacing : float
            Maximum distance between consecutive samples along an edge.

        Returns
        -------
        points : np.ndarray
            Sample coordinates, shape ``(S, D)``.
        radii : np.ndarray or None
            Interpolated radius at each sample, shape ``(S,)``.
        edge_index : np.ndarray
            Index of the edge each sample belongs to, shape ``(S,)``.
        """
        points, radii, edge_index = [], [], []
        for index, path in enumerate(self.edge_paths):
            values = None if self.edge_radii is None else self.edge_radii[index]
            edge_points, edge_values = resample_polyline(path, spacing, values)
            points.append(edge_points)
            edge_index.append(np.full(len(edge_points), index))
            if edge_values is not None:
                radii.append(edge_values)
        if not points:
            return (np.empty((0, self.ndim)), None, np.empty(0, dtype=np.int64))
        return (
            np.concatenate(points),
            np.concatenate(radii) if self.edge_radii is not None else None,
            np.concatenate(edge_index),
        )

    def to_dict(self) -> dict[str, np.ndarray]:
        """Flatten into a dictionary of arrays, e.g. for ``np.savez``.

        The ragged paths are concatenated, with ``path_offsets`` marking where
        each edge's points start (and a final entry for the end).
        """
        lengths = [len(path) for path in self.edge_paths]
        data = {
            "node_positions": self.node_positions,
            "edges": self.edges,
            "path_points": (
                np.concatenate(self.edge_paths)
                if self.edge_paths
                else np.empty((0, self.ndim))
            ),
            "path_offsets": np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        }
        if self.edge_radii is not None:
            data["path_radii"] = (
                np.concatenate(self.edge_radii) if self.edge_radii else np.empty(0)
            )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, np.ndarray]) -> "SkeletonGraph":
        """Rebuild a graph from the output of :meth:`to_dict`."""
        offsets = data["path_offsets"]
        slices = [slice(start, stop) for start, stop in itertools.pairwise(offsets)]
        radii = data.get("path_radii")
        return cls(
            node_positions=data["node_positions"],
            edges=data["edges"],
            edge_paths=[data["path_points"][s] for s in slices],
            edge_radii=None if radii is None else [radii[s] for s in slices],
        )

    def to_networkx(self) -> "nx.MultiGraph":
        """Convert to a networkx ``MultiGraph``.

        Nodes carry ``position`` (and ``radius``); edges carry ``index``,
        ``path`` (and ``radii``). A multigraph is used so that parallel edges
        and self-loops in imperfect skeletons survive the conversion.
        """
        import networkx as nx

        graph = nx.MultiGraph()
        node_radii = self.node_radii()
        for node, position in enumerate(self.node_positions):
            attributes = {"position": position}
            if node_radii is not None:
                attributes["radius"] = node_radii[node]
            graph.add_node(node, **attributes)
        for index, (u, v) in enumerate(self.edges):
            attributes = {"index": index, "path": self.edge_paths[index]}
            if self.edge_radii is not None:
                attributes["radii"] = self.edge_radii[index]
            graph.add_edge(int(u), int(v), **attributes)
        return graph


def resample_polyline(
    points: np.ndarray, spacing: float, values: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray | None]:
    """Resample a polyline uniformly by arc length.

    Parameters
    ----------
    points : np.ndarray
        Polyline vertices, shape ``(P, D)``.
    spacing : float
        Maximum distance between consecutive output samples.
    values : np.ndarray, optional
        Values at the vertices, shape ``(P,)``, interpolated linearly along
        the arc length.

    Returns
    -------
    samples : np.ndarray
        Samples including both ends, shape ``(S, D)``.
    sample_values : np.ndarray or None
        Interpolated values, shape ``(S,)``.
    """
    arc_length = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    )
    n_samples = max(2, int(np.ceil(arc_length[-1] / spacing)) + 1)
    targets = np.linspace(0.0, arc_length[-1], n_samples)
    samples = np.stack(
        [
            np.interp(targets, arc_length, points[:, axis])
            for axis in range(points.shape[1])
        ],
        axis=1,
    )
    sample_values = None if values is None else np.interp(targets, arc_length, values)
    return samples, sample_values


def simplify_polyline(points: np.ndarray, tolerance: float) -> np.ndarray:
    """Douglas-Peucker simplification of a polyline.

    Parameters
    ----------
    points : np.ndarray
        Polyline vertices, shape ``(P, D)``.
    tolerance : float
        Largest allowed distance of a dropped vertex from the simplified
        polyline.

    Returns
    -------
    np.ndarray
        Sorted indices of the kept vertices, always including both ends.
    """
    keep = np.zeros(len(points), dtype=bool)
    keep[[0, -1]] = True
    stack = [(0, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        if last - first < 2:
            continue
        start, end = points[first], points[last]
        inner = points[first + 1 : last]
        vector = end - start
        squared_length = float(vector @ vector)
        if squared_length > 0:
            t = np.clip((inner - start) @ vector / squared_length, 0.0, 1.0)
            distance = np.linalg.norm(inner - (start + t[:, None] * vector), axis=1)
        else:
            distance = np.linalg.norm(inner - start, axis=1)
        farthest = int(np.argmax(distance))
        if distance[farthest] > tolerance:
            split = first + 1 + farthest
            keep[split] = True
            stack += [(first, split), (split, last)]
    return np.flatnonzero(keep)


def _polyline_length(points: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
