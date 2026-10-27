import numpy as np
import pytest

from spleleton.graph import SkeletonGraph
from spleleton.postprocess import (
    bridge_breaks,
    clean_graph,
    merge_degree_two_nodes,
    prune_short_leaves,
)


def _graph(nodes, edges, radius=2.0):
    nodes = np.asarray(nodes, dtype=float)
    return SkeletonGraph(
        nodes,
        edges,
        [nodes[[u, v]] for u, v in edges],
        [np.full(2, radius) for _ in edges],
    )


def test_merge_degree_two_nodes():
    graph = _graph([[0, 0], [5, 0], [10, 0], [15, 5]], [[0, 1], [1, 2], [2, 3]])
    merged = merge_degree_two_nodes(graph)
    assert (merged.n_nodes, merged.n_edges) == (2, 1)
    path = merged.edge_paths[0]
    if path[0, 0] != 0:
        path = path[::-1]
    np.testing.assert_allclose(path, graph.node_positions)
    # a ring of degree-2 nodes becomes one self-loop
    ring = merge_degree_two_nodes(
        _graph([[0, 0], [5, 0], [5, 5]], [[0, 1], [1, 2], [2, 0]])
    )
    assert ring.n_edges == 1
    assert ring.edges[0, 0] == ring.edges[0, 1]


def test_prune_short_leaves_removes_spurs_only():
    # a long trunk with a short spur (protrudes 1 radius) and a long side branch
    nodes = [[0, 0], [20, 0], [40, 0], [20, 4], [30, 0], [30, 15]]
    edges = [[0, 1], [1, 4], [4, 2], [1, 3], [4, 5]]
    pruned = prune_short_leaves(_graph(nodes, edges), min_protrusion=1.5)
    # the spur at node 3 goes; the side branch stays; node 1 is merged away
    assert pruned.n_edges == 3
    assert sorted(pruned.degrees) == [1, 1, 1, 3]


def test_prune_removes_a_terminal_fork_together():
    # a trunk ending in two short twigs (a typical skeletonisation artefact)
    nodes = [[0, 0], [20, 0], [22, 2], [22, -2]]
    pruned = prune_short_leaves(_graph(nodes, [[0, 1], [1, 2], [1, 3]]))
    assert pruned.n_edges == 1
    np.testing.assert_allclose(pruned.edge_paths[0][-1], [20, 0])


def test_bridge_breaks_joins_facing_ends():
    # a straight branch cut by a 3-voxel gap (radius 2: reach 4)
    graph = _graph([[0, 0], [10, 0], [17, 0], [30, 0]], [[0, 1], [2, 3]])
    bridged, bridges = bridge_breaks(graph)
    assert len(bridges) == 1
    assert bridged.n_components() == 1
    assert bridged.n_edges == 1


@pytest.mark.parametrize(
    ("far_end", "reason"),
    [([25.0, 0.0], "gap too wide"), ([10.0, 12.0], "not facing")],
)
def test_bridge_breaks_rejects(far_end, reason):
    near = [far_end[0] + (5.0 if reason == "gap too wide" else 0.0), far_end[1] + 10.0]
    graph = _graph([[0, 0], [10, 0], far_end, near], [[0, 1], [2, 3]])
    _, bridges = bridge_breaks(graph)
    assert bridges == [], reason


def test_bridge_breaks_reaches_into_an_edge():
    # a child cut off next to its parent's junction points at the parent's edge
    graph = _graph([[0, 0], [30, 0], [15, 5], [15, 25]], [[0, 1], [2, 3]])
    bridged, bridges = bridge_breaks(graph)
    assert len(bridges) == 1
    assert bridged.n_components() == 1
    assert sorted(bridged.degrees) == [1, 1, 1, 3]
    np.testing.assert_allclose(bridges[0].end, [15, 0])


def test_bridging_never_creates_cycles():
    # two ends of the same component facing each other are left alone
    graph = _graph([[0, 0], [10, 0], [10, 10], [0, 3]], [[0, 1], [1, 2], [2, 3]])
    _, bridges = bridge_breaks(graph, reach=10.0, max_angle=90.0)
    assert bridges == []


def test_clean_graph_needs_radii():
    nodes = np.array([[0.0, 0.0], [5.0, 0.0]])
    with pytest.raises(ValueError):
        clean_graph(SkeletonGraph(nodes, [[0, 1]], [nodes]))


def test_drop_small_components():
    from spleleton.postprocess import drop_small_components

    # a long trunk and a tiny island (length 2, radius 2: under 3 radii)
    graph = _graph([[0, 0], [30, 0], [50, 50], [52, 50]], [[0, 1], [2, 3]])
    kept = drop_small_components(graph, min_length=3.0)
    assert kept.n_edges == 1
    assert kept.n_components() == 1
    np.testing.assert_allclose(kept.edge_paths[0], [[0, 0], [30, 0]])


def test_bridge_breaks_joins_touching_tubes_at_steep_angles():
    # neither end points at the other piece within 30 degrees (68 and 40
    # degrees), but the tubes overlap (gap -1.3), so the mask joins them
    graph = _graph([[0, 0], [10, 0], [11, 2.5], [25, 10]], [[0, 1], [2, 3]])
    _, bridges = bridge_breaks(graph, max_angle=30.0)
    assert len(bridges) == 1
    assert bridges[0].gap < 0
    # with a small positive gap at similar angles, nothing is bridged
    apart = _graph([[0, 0], [10, 0], [12, 4.5], [26, 12]], [[0, 1], [2, 3]])
    _, none = bridge_breaks(apart, max_angle=30.0)
    assert none == []


def test_prune_measures_protrusion_outside_other_tubes():
    # a thick trunk (radius 4); a leaf attached at a junction 6 voxels away
    # from where it leaves the tube runs mostly inside the trunk: a spur
    nodes = [[0, 0], [30, 0], [60, 0], [36, 5]]
    edges = [[0, 1], [1, 2], [1, 3]]
    graph = SkeletonGraph(
        np.array(nodes, dtype=float),
        edges,
        [np.array(nodes, dtype=float)[[u, v]] for u, v in edges],
        [np.full(2, 4.0), np.full(2, 4.0), np.full(2, 1.5)],
    )
    pruned = prune_short_leaves(graph, min_protrusion=1.5)
    assert pruned.n_edges == 1
    # the same leaf sticking well out of the trunk stays
    long_nodes = [[0, 0], [30, 0], [60, 0], [36, 15]]
    kept = prune_short_leaves(
        SkeletonGraph(
            np.array(long_nodes, dtype=float),
            edges,
            [np.array(long_nodes, dtype=float)[[u, v]] for u, v in edges],
            [np.full(2, 4.0), np.full(2, 4.0), np.full(2, 1.5)],
        )
    )
    assert kept.n_edges == 3
