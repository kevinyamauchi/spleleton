import dataclasses

import numpy as np
import pytest

from spleleton.baseline import skeletonize_to_graph
from spleleton.fit import FitConfig, default_losses
from spleleton.graph import SkeletonGraph, simplify_polyline
from spleleton.initialise import segments_from_graph
from spleleton.metrics import compare_graphs
from spleleton.synthetic.dataset import airway_config, generate_sample


def test_simplify_polyline():
    # an L with jitter: only the corner survives a tolerance of 0.5
    points = np.array(
        [[0, 0], [1, 0.1], [2, -0.1], [3, 0], [3, 1], [3.1, 2], [3, 3]], dtype=float
    )
    np.testing.assert_array_equal(simplify_polyline(points, 0.5), [0, 3, 6])
    # the jittered points 1 and 2 are kept, but (3, 1) lies within 0.05 of the
    # chord from (3, 0) to (3.1, 2) and is dropped
    np.testing.assert_array_equal(simplify_polyline(points, 0.05), [0, 1, 2, 3, 5, 6])
    np.testing.assert_array_equal(simplify_polyline(points[:2], 0.5), [0, 1])


def _y_graph():
    """A Y with a detour on the root edge, so it needs two segments."""
    nodes = np.array([[0.0, 0.0], [10.0, 0.0], [15.0, 5.0], [15.0, -5.0]])
    paths = [
        np.array([[0.0, 0.0], [5.0, 3.0], [10.0, 0.0]]),
        nodes[[1, 2]],
        nodes[[1, 3]],
    ]
    radii = [np.array([3.0, 2.5, 2.0]), np.array([2.0, 1.5]), np.array([2.0, 1.0])]
    return SkeletonGraph(nodes, [[0, 1], [1, 2], [1, 3]], paths, radii)


def test_segments_from_graph_and_back():
    graph = _y_graph()
    segments, layout = segments_from_graph(graph, tolerance=0.5, opacity=0.8)
    assert len(segments) == 4
    assert [list(chain) for chain in layout.edge_segments] == [[0, 1], [2], [3]]
    np.testing.assert_allclose(segments.opacity, np.full(4, 0.8))
    np.testing.assert_allclose(segments.radius[1], [2.5, 2.0])

    restored = layout.to_graph(segments)
    np.testing.assert_allclose(restored.node_positions, graph.node_positions, atol=1e-6)
    np.testing.assert_array_equal(restored.edges, graph.edges)
    np.testing.assert_allclose(restored.edge_paths[0], graph.edge_paths[0], atol=1e-6)
    # the junction's radius is the mean of the three edge ends meeting there
    np.testing.assert_allclose(restored.edge_radii[1][0], 2.0, atol=1e-6)


def _with_opacity(segments, opacity):
    return dataclasses.replace(segments, opacity=np.asarray(opacity))


def test_dead_spur_is_pruned():
    segments, layout = segments_from_graph(_y_graph(), tolerance=0.5)
    segments = _with_opacity(segments, [0.9, 0.9, 0.9, 0.01])
    graph = layout.to_graph(segments)
    assert graph.n_edges == 2
    assert graph.n_nodes == 3
    assert graph.n_components() == 1


def test_dead_internal_edge_is_kept():
    segments, layout = segments_from_graph(_y_graph(), tolerance=0.5)
    # the root edge holds the tree together only with node 0; with both of
    # its segments dead it is a dead leaf edge and goes...
    segments = _with_opacity(segments, [0.01, 0.01, 0.9, 0.9])
    assert layout.to_graph(segments).n_edges == 2
    # ...but an internal edge between two junctions never goes
    nodes = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [25, 5], [25, -5], [-5, 5]])
    edges = [[0, 1], [1, 2], [2, 3], [2, 4], [0, 5]]
    chain = SkeletonGraph(
        np.vstack([nodes, [[-5.0, -5.0]]]),
        [*edges, [0, 6]],
        [
            np.vstack([nodes[u], np.vstack([nodes, [[-5.0, -5.0]]])[v]])
            for u, v in [*edges, [0, 6]]
        ],
    )
    segments, layout = segments_from_graph(chain)
    segments = _with_opacity(segments, [0.9, 0.01, 0.9, 0.9, 0.9, 0.9])
    assert layout.to_graph(segments).n_edges == 6


def test_dead_cycle_edge_is_pruned():
    nodes = np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 5.0], [20.0, 0.0]])
    edges = [[0, 1], [1, 2], [2, 0], [1, 3]]
    graph = SkeletonGraph(nodes, edges, [nodes[[u, v]] for u, v in edges])
    segments, layout = segments_from_graph(graph)
    segments = _with_opacity(segments, [0.9, 0.01, 0.9, 0.9])
    result = layout.to_graph(segments)
    assert result.n_edges == 3
    assert result.n_edges - result.n_nodes + result.n_components() == 0


@pytest.mark.parametrize("corrupted", [False, True])
def test_fitting_from_the_skeleton_keeps_the_tree_connected(backend, corrupted):
    sample = generate_sample(airway_config(2, corrupted=corrupted), 1)
    baseline = skeletonize_to_graph(sample.mask)
    arrays, layout = segments_from_graph(baseline)
    segments = backend.LineSegments.from_arrays(arrays, dtype=backend.dtype("float32"))
    segments, _ = backend.fit(
        segments,
        sample.mask.astype(np.float32),
        config=FitConfig(n_iterations=150, losses=default_losses(sparsity=0.1)),
    )
    fitted = layout.to_graph(segments)
    assert fitted.n_components() <= baseline.n_components()
    before = compare_graphs(baseline, sample.graph)
    after = compare_graphs(fitted, sample.graph)
    assert after.centreline_f1 >= before.centreline_f1 - 0.02
    assert after.n_cycles <= before.n_cycles
