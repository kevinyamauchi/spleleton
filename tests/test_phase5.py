import numpy as np
import pytest
from conftest import TORCH_NOT_IN_SYNC

from spleleton.assemble import assemble_graph
from spleleton.fit import FitConfig, default_losses
from spleleton.initialise import segments_from_graph
from spleleton.metrics import compare_graphs
from spleleton.pipeline import ScratchFitConfig, SkeletonFitConfig
from spleleton.primitives import SegmentArrays
from spleleton.recording import FitRecorder
from spleleton.seeding import seed_segments
from spleleton.synthetic.dataset import airway_config, generate_sample


def _segments(starts, ends, radius=2.0, opacity=0.9):
    return SegmentArrays(starts, ends, radius=radius, opacity=opacity)


def test_seed_segments_follow_a_tube():
    mask = np.zeros((40, 80), dtype=bool)
    mask[16:25, 5:75] = True  # a horizontal tube of radius ~4.5
    seeds = seed_segments(mask)
    assert len(seeds) >= 5
    directions = seeds.end - seeds.start
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    assert np.all(np.abs(directions[:, 1]) > 0.9)  # along axis 1
    assert np.allclose(0.5 * (seeds.start + seeds.end)[:, 0], 20.0, atol=1.0)
    # restricting the candidates restricts the seeds
    left = np.zeros_like(mask)
    left[:, :30] = True
    restricted = seed_segments(mask, candidates=left)
    assert 0 < len(restricted) < len(seeds)
    assert seed_segments(np.zeros((10, 10), dtype=bool)).start.shape == (0, 2)


def _endpoint_pairs(backend, segments, reach):
    """The backend's endpoint pairs, as a list of [a, b]."""
    pairs = backend.fn("priors", "endpoint_pairs")(segments, reach)
    if backend.name == "jax":
        pairs, valid = pairs
        pairs = pairs[valid]
    return backend.asnumpy(pairs).tolist()


def test_endpoint_pairs_and_connection_loss(backend):
    arrays = _segments(
        [[0.0, 0.0], [10.5, 0.0], [30.0, 0.0]],
        [[10.0, 0.0], [20.0, 0.0], [40.0, 0.0]],
    )
    segments = backend.LineSegments.from_arrays(arrays, dtype=backend.dtype("float64"))
    from spleleton import priors

    # only the end of segment 0 (index 3) and the start of segment 1 (index 1)
    assert _endpoint_pairs(backend, segments, reach=1.5) == [[1, 3]]
    loss = getattr(priors, f"{backend.name.capitalize()}ConnectionLoss")(
        n_voxels=100, weight=2.0
    )
    value = backend.prior_terms(loss, segments)["connection"]
    assert float(backend.asnumpy(value)) == pytest.approx(2.0 * 0.25 / 100)


def test_free_end_cost_counts_leaves(backend):
    arrays, layout = segments_from_graph(_y_graph(), opacity=0.5)
    segments = backend.LineSegments.from_arrays(arrays, dtype=backend.dtype("float64"))
    from spleleton import priors

    cost = getattr(priors, f"{backend.name.capitalize()}FreeEndCost")(
        layout, n_voxels=1000, weight=3.0
    )
    assert len(cost.leaf_chains) == 3
    value = backend.prior_terms(cost, segments)["free_end"]
    radii = backend.asnumpy(segments.radius).mean(axis=-1)
    expected = 3.0 * sum(r**2 * 0.5 for r in radii) / 1000
    assert float(backend.asnumpy(value)) == pytest.approx(expected)


def test_leaf_chains():
    from spleleton.priors._leaves import leaf_chains

    _, layout = segments_from_graph(_y_graph())
    assert [list(chain) for chain in leaf_chains(layout)] == [[0], [1], [2]]


def _y_graph():
    from spleleton.graph import SkeletonGraph

    nodes = np.array([[0.0, 20.0], [20.0, 20.0], [35.0, 8.0], [35.0, 32.0]])
    edges = [[0, 1], [1, 2], [1, 3]]
    return SkeletonGraph(
        nodes, edges, [nodes[[u, v]] for u, v in edges], [np.full(2, 3.0)] * 3
    )


def test_assemble_chain_junction_and_duplicates():
    # a trunk of two segments whose ends nearly meet, a branch starting in
    # the middle of the second (a T-junction), a duplicate inside the trunk,
    # and a pruned segment
    segments = _segments(
        [[0, 20], [10.4, 20], [17, 20.5], [4, 20], [30, 30]],
        [[10, 20], [25, 20], [17, 35], [8, 20], [35, 30]],
        radius=2.0,
        opacity=[0.9, 0.9, 0.9, 0.9, 0.01],
    )
    graph = assemble_graph(segments)
    assert graph.n_components() == 1
    assert sorted(graph.degrees) == [1, 1, 1, 3]
    junction = graph.node_positions[graph.junction_nodes[0]]
    np.testing.assert_allclose(junction, [17, 20], atol=0.6)


def test_assemble_empty():
    graph = assemble_graph(_segments([[0.0, 0.0]], [[5.0, 0.0]], opacity=0.01))
    assert graph.n_edges == 0


def test_concatenate_segments():
    a = _segments([[0.0, 0.0]], [[1.0, 0.0]], radius=1.0, opacity=0.3)
    b = _segments([[5.0, 0.0]], [[6.0, 0.0]], radius=2.0, opacity=[1.0])
    both = a.concatenate(b)
    assert len(both) == 2
    np.testing.assert_allclose(both.opacity, [0.3, 1 - 1e-6])
    np.testing.assert_allclose(both.radius, [[1.0, 1.0], [2.0, 2.0]])


def test_concatenated_segments_convert_to_a_backend(backend):
    both = _segments([[0.0, 0.0]], [[1.0, 0.0]], opacity=0.3).concatenate(
        _segments([[5.0, 0.0]], [[6.0, 0.0]], opacity=[1.0])
    )
    restored = backend.LineSegments.from_arrays(both, dtype=backend.dtype("float64"))
    assert np.isfinite(backend.asnumpy(restored.opacity_logit)).all()
    if backend.name == "torch":
        assert restored.start.requires_grad


@pytest.mark.parametrize("pipeline", ["phase4", "phase5"])
def test_pipelines_beat_skeletonize_on_a_corrupted_tree(backend, pipeline):
    from spleleton.baseline import skeletonize_to_graph

    if backend.name == "torch":
        pytest.skip(TORCH_NOT_IN_SYNC)

    sample = generate_sample(airway_config(2), 0)
    if pipeline == "phase4":
        config = SkeletonFitConfig(
            fit=FitConfig(n_iterations=200, losses=default_losses(sparsity=0.1))
        )
        result = backend.fn("pipeline", "skeleton_fit")(sample.mask, config=config)
    else:
        config = ScratchFitConfig(
            fit=FitConfig(n_iterations=100, losses=default_losses(sparsity=0.1))
        )
        result = backend.fn("pipeline", "scratch_fit")(sample.mask, config=config)
    assert isinstance(result.segments, backend.LineSegments)
    before = compare_graphs(skeletonize_to_graph(sample.mask), sample.graph)
    after = compare_graphs(result.graph, sample.graph)
    assert after.n_spur_ends < before.n_spur_ends
    assert after.n_components <= before.n_components
    assert after.junction_f1 >= before.junction_f1


def test_recorder_follows_rounds_with_new_segments(backend):
    if backend.name == "torch":
        pytest.skip(TORCH_NOT_IN_SYNC)
    sample = generate_sample(airway_config(2, shape=(64, 64)), 3)
    recorder = FitRecorder(sample.mask.shape, every=10)
    config = ScratchFitConfig(fit=FitConfig(n_iterations=20), n_rounds=2)
    scratch_fit = backend.fn("pipeline", "scratch_fit")
    result = scratch_fit(sample.mask, config=config, callback=recorder)
    recording = recorder.finish(result.history, final_graph=result.graph)
    # 0, 10, 20 (end of round 1), 20 (start of round 2), 30, 40
    assert list(recording.iterations) == [0, 10, 20, 20, 30, 40]
    assert len(recording.loss["loss"]) == 40
    assert recording.start.shape[1] == len(result.segments)
    first = np.isnan(recording.opacity[0])
    assert first.sum() == len(result.segments) - (~first).sum()
    np.testing.assert_allclose(
        recording.segments(len(recording.iterations) - 1).start,
        backend.asnumpy(result.segments.start),
        rtol=1e-6,
    )


def test_assemble_drops_collapsed_and_chained_duplicates():
    # a trunk of one long segment; a collapsed (thin) segment elsewhere; and a
    # chain of two short segments running inside the trunk's tube
    segments = _segments(
        [[0, 20], [10, 40], [3, 20.5], [9, 20.5]],
        [[30, 20], [20, 40], [9, 20.5], [15, 20.5]],
        radius=[3.0, 0.3, 1.5, 1.5],
    )
    graph = assemble_graph(segments)
    assert graph.n_edges == 1
    np.testing.assert_allclose(sorted(graph.node_positions[:, 0]), [0, 30])


def _chain(start, end, n, radius):
    t = np.linspace(0, 1, n + 1)[:, None]
    points = np.asarray(start, float) + t * (np.subtract(end, start))
    return points[:-1], points[1:], np.full(n, radius)


def test_assemble_keeps_tubes_under_a_crossing_segment():
    # two parallel tubes, chained, and a false join crossing between them at
    # x = 32 (proto-cases' "contact"): the join's axis runs through the
    # midpoints of the tube pieces under it, which must not drop them
    parts = [_chain([6, 26], [58, 26], 7, 3.5), _chain([6, 38], [58, 38], 7, 3.5)]
    starts = np.concatenate([p[0] for p in parts] + [[[32.0, 26.0]]])
    ends = np.concatenate([p[1] for p in parts] + [[[32.0, 38.0]]])
    radii = np.concatenate([p[2] for p in parts] + [[1.5]])
    graph = assemble_graph(_segments(starts, ends, radius=radii))
    assert graph.n_components() == 1
    assert sorted(graph.degrees) == [1, 1, 1, 1, 3, 3]


def test_assemble_keeps_a_parent_link_at_a_junction():
    # fitted segments at a junction of proto-v1 gaps-w6 (seed 0): the link
    # (1) from the parent's last segment (0) to the junction runs along the
    # parent; its start is at the children's starts (2, 3), at about 30
    # degrees, and its middle near the parent's end. Ends that do not run
    # along it cover nothing, so it stays and the junction holds.
    segments = _segments(
        [
            [84.0, 55.9, 87.6],
            [86.3, 57.2, 88.5],
            [90.0, 61.3, 93.3],
            [95.7, 58.2, 87.1],
        ],
        [
            [76.4, 52.0, 84.4],
            [83.0, 55.6, 87.1],
            [86.8, 57.5, 88.8],
            [87.1, 57.5, 88.6],
        ],
        radius=[[2.25, 2.86], [2.35, 2.22], [1.44, 2.26], [1.17, 1.7]],
    )
    graph = assemble_graph(segments)
    assert graph.n_components() == 1
    assert sorted(graph.degrees) == [1, 1, 1, 3]
    # a duplicate running along the parent is still dropped
    duplicate = _segments(
        np.vstack([segments.start, [[82.0, 55.0, 86.8]]]),
        np.vstack([segments.end, [[78.0, 53.0, 85.1]]]),
        radius=np.vstack([segments.radius, [[2.5, 2.5]]]),
    )
    assert sorted(assemble_graph(duplicate).degrees) == [1, 1, 1, 3]
