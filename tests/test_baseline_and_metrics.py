import numpy as np
import pytest

from spleleton.baseline import skeleton_to_graph, skeletonize_to_graph
from spleleton.graph import SkeletonGraph
from spleleton.metrics import compare_graphs
from spleleton.synthetic.rasterize import rasterize_graph


def _y_graph():
    nodes = np.array([[2.0, 30.0], [25.0, 30.0], [50.0, 15.0], [50.0, 45.0]])
    edges = np.array([[0, 1], [1, 2], [1, 3]])
    return SkeletonGraph(
        nodes,
        edges,
        [nodes[[u, v]] for u, v in edges],
        [np.array([4.0, 4.0]), np.array([3.0, 3.0]), np.array([3.0, 3.0])],
    )


# ------------------------------------------------------------------ baseline


def test_skeleton_to_graph_line():
    skeleton = np.zeros((10, 10), dtype=bool)
    skeleton[5, 1:9] = True
    graph = skeleton_to_graph(skeleton)
    assert (graph.n_nodes, graph.n_edges) == (2, 1)
    np.testing.assert_allclose(graph.edge_lengths, [7.0])


def test_skeleton_to_graph_staircase_is_a_single_path():
    skeleton = np.zeros((12, 12), dtype=bool)
    for i in range(8):
        skeleton[i + 1, i + 1] = True
        skeleton[i + 1, i + 2] = True
    graph = skeleton_to_graph(skeleton)
    assert (graph.n_nodes, graph.n_edges) == (2, 1)


def test_skeleton_to_graph_cross_and_ring():
    skeleton = np.zeros((30, 30), dtype=bool)
    skeleton[5, 2:13] = True
    skeleton[1:10, 7] = True
    # a separate ring
    skeleton[20, 15:25] = skeleton[27, 15:25] = True
    skeleton[20:28, 15] = skeleton[20:28, 24] = True
    graph = skeleton_to_graph(skeleton, spacing=(2.0, 1.0))
    degrees = np.sort(graph.degrees)
    # the cross: 4 ends and one degree-4 junction; the ring: a self-loop
    assert list(degrees) == [1, 1, 1, 1, 2, 4]
    assert graph.n_components() == 2
    junction = graph.node_positions[graph.junction_nodes[0]]
    np.testing.assert_allclose(junction, [10.0, 7.0])


def test_skeletonize_to_graph_radii():
    graph = _y_graph()
    mask = rasterize_graph(graph, (60, 60))
    result = skeletonize_to_graph(mask)
    assert result.edge_radii is not None
    _, radii, _ = result.sample_points()
    assert 2.0 < np.median(radii) < 5.0


def test_skeletonize_to_graph_3d():
    nodes = np.array(
        [[2.0, 20.0, 20.0], [20.0, 20.0, 20.0], [35.0, 10.0, 20.0], [35.0, 30.0, 20.0]]
    )
    edges = np.array([[0, 1], [1, 2], [1, 3]])
    graph = SkeletonGraph(
        nodes,
        edges,
        [nodes[[u, v]] for u, v in edges],
        [np.array([3.0, 3.0]), np.array([2.5, 2.5]), np.array([2.5, 2.5])],
    )
    result = skeletonize_to_graph(rasterize_graph(graph, (40, 40, 40)))
    metrics = compare_graphs(result, graph)
    assert metrics.centreline_f1 > 0.9
    assert metrics.junction_f1 == 1.0
    assert metrics.n_components == 1


# ------------------------------------------------------------------- metrics


def test_identical_graphs_score_perfectly():
    graph = _y_graph()
    metrics = compare_graphs(graph, graph)
    assert metrics.centreline_f1 == 1.0
    assert metrics.junction_f1 == 1.0
    assert (metrics.n_spur_ends, metrics.n_break_ends) == (0, 0)
    assert (metrics.n_components, metrics.n_cycles) == (1, 0)
    # at a junction, the nearest true sample may belong to a neighbouring edge
    assert metrics.radius_error < 0.05


def test_spur_is_detected():
    graph = _y_graph()
    # a spur sticking out sideways from the root branch
    nodes = np.vstack([graph.node_positions, [[12.0, 30.0], [12.0, 37.0]]])
    edges = np.array([[0, 4], [4, 1], [1, 2], [1, 3], [4, 5]])
    predicted = SkeletonGraph(nodes, edges, [nodes[[u, v]] for u, v in edges])
    metrics = compare_graphs(predicted, graph)
    assert metrics.n_spur_ends == 1
    assert metrics.n_break_ends == 0
    assert metrics.centreline_precision < 1.0
    assert metrics.centreline_recall == 1.0
    # the spur's base is a false junction
    assert metrics.junction_precision == 0.5
    assert np.isnan(metrics.radius_error)


def test_break_is_detected():
    graph = _y_graph()
    # the root branch, with a cut wider than the centreline tolerance
    nodes = np.vstack([graph.node_positions, [[8.0, 30.0], [16.0, 30.0]]])
    edges = np.array([[0, 4], [5, 1], [1, 2], [1, 3]])
    predicted = SkeletonGraph(nodes, edges, [nodes[[u, v]] for u, v in edges])
    metrics = compare_graphs(predicted, graph)
    assert metrics.n_break_ends == 2
    assert metrics.n_spur_ends == 0
    assert metrics.n_components == 2
    assert metrics.centreline_precision == 1.0
    assert metrics.centreline_recall < 1.0


def test_empty_prediction():
    graph = _y_graph()
    empty = SkeletonGraph(np.empty((0, 2)), np.empty((0, 2)), [])
    metrics = compare_graphs(empty, graph)
    assert metrics.centreline_recall == 0.0
    assert metrics.centreline_f1 == 0.0
    assert np.isnan(metrics.centreline_precision)


def test_target_needs_radii():
    graph = _y_graph()
    without_radii = SkeletonGraph(graph.node_positions, graph.edges, graph.edge_paths)
    with pytest.raises(ValueError):
        compare_graphs(graph, without_radii)
