import numpy as np
import pytest

from spleleton.graph import SkeletonGraph, resample_polyline


def _y_graph(radii=True):
    """A Y: root (0) -> junction (1) -> two leaves (2, 3)."""
    nodes = np.array([[0.0, 0.0], [10.0, 0.0], [15.0, 5.0], [15.0, -5.0]])
    edges = np.array([[0, 1], [1, 2], [1, 3]])
    paths = [nodes[[u, v]] for u, v in edges]
    edge_radii = [np.array([3.0, 2.0]), np.array([2.0, 1.5]), np.array([1.5, 1.0])]
    return SkeletonGraph(nodes, edges, paths, edge_radii if radii else None)


def test_basic_properties():
    graph = _y_graph()
    assert graph.ndim == 2
    assert (graph.n_nodes, graph.n_edges) == (4, 3)
    np.testing.assert_array_equal(graph.degrees, [1, 3, 1, 1])
    np.testing.assert_array_equal(graph.end_nodes, [0, 2, 3])
    np.testing.assert_array_equal(graph.junction_nodes, [1])
    np.testing.assert_allclose(graph.edge_lengths, [10, 50**0.5, 50**0.5])
    np.testing.assert_allclose(graph.node_radii(), [3.0, 2.0, 1.5, 1.0])
    assert graph.n_components() == 1


def test_components_count_isolated_nodes():
    graph = _y_graph()
    graph = SkeletonGraph(
        np.vstack([graph.node_positions, [[50.0, 50.0]]]),
        graph.edges,
        graph.edge_paths,
        graph.edge_radii,
    )
    assert graph.n_components() == 2


@pytest.mark.parametrize(
    "change",
    [
        {"edge_paths": "too few"},
        {"edges": np.array([[0, 1], [1, 2], [1, 7]])},
        {"path_start": np.array([1.0, 1.0])},
        {"radii_shape": True},
    ],
)
def test_validation(change):
    graph = _y_graph()
    paths = list(graph.edge_paths)
    radii = list(graph.edge_radii)
    edges = change.get("edges", graph.edges)
    if "edge_paths" in change:
        paths = paths[:2]
    if "path_start" in change:
        paths[0] = np.vstack([change["path_start"], paths[0][1:]])
    if "radii_shape" in change:
        radii[0] = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        SkeletonGraph(graph.node_positions, edges, paths, radii)


def test_dict_round_trip():
    for graph in (_y_graph(), _y_graph(radii=False)):
        restored = SkeletonGraph.from_dict(graph.to_dict())
        np.testing.assert_array_equal(restored.node_positions, graph.node_positions)
        np.testing.assert_array_equal(restored.edges, graph.edges)
        for a, b in zip(restored.edge_paths, graph.edge_paths, strict=True):
            np.testing.assert_array_equal(a, b)
        assert (restored.edge_radii is None) == (graph.edge_radii is None)


def test_to_networkx():
    nx_graph = _y_graph().to_networkx()
    assert nx_graph.number_of_nodes() == 4
    assert nx_graph.number_of_edges() == 3
    assert nx_graph.degree[1] == 3
    np.testing.assert_allclose(nx_graph.nodes[1]["position"], [10.0, 0.0])


def test_resample_polyline():
    points = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 3.0]])
    samples, values = resample_polyline(points, 1.0, np.array([0.0, 4.0, 7.0]))
    assert len(samples) == 8
    np.testing.assert_allclose(samples[0], points[0])
    np.testing.assert_allclose(samples[-1], points[-1])
    np.testing.assert_allclose(np.linalg.norm(np.diff(samples, axis=0), axis=1), 1.0)
    # values interpolate linearly in arc length, which here equals the value
    np.testing.assert_allclose(values, np.arange(8.0))


def test_sample_points():
    points, radii, edge_index = _y_graph().sample_points(0.5)
    assert points.shape[1] == 2
    assert radii.shape == edge_index.shape == (len(points),)
    assert set(edge_index) == {0, 1, 2}
    assert radii.min() >= 1.0
    assert radii.max() <= 3.0
