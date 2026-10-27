import numpy as np
import pytest

from spleleton.graph import SkeletonGraph
from spleleton.primitives import SegmentArrays
from spleleton.render import RenderConfig
from spleleton.synthetic.dataset import airway_config, generate_sample
from spleleton.synthetic.rasterize import rasterize_graph


def _dice(a: np.ndarray, b: np.ndarray) -> float:
    return 2 * np.sum(a & b) / (a.sum() + b.sum())


def _random_arrays(ndim, n=12, extent=40.0, seed=0):
    rng = np.random.default_rng(seed)
    start = extent * rng.random((n, ndim))
    return SegmentArrays(
        start,
        start + 8 * (rng.random((n, ndim)) - 0.5),
        radius=0.5 + 3 * rng.random((n, 2)),
        opacity=0.2 + 0.7 * rng.random(n),
    )


def _random_segments(backend, ndim, **kwargs):
    arrays = _random_arrays(ndim, **kwargs)
    return backend.segments(arrays.start, arrays.end, arrays.radius, arrays.opacity)


def _line(backend, n_copies=1, **kwargs):
    """A vertical segment in a (21, 31) grid, possibly repeated."""
    return backend.segments(
        [[10.0, 5.0]] * n_copies, [[10.0, 25.0]] * n_copies, **kwargs
    )


@pytest.mark.parametrize("union", ["max", "soft", "softmax"])
@pytest.mark.parametrize("mode", ["tube", "centreline"])
@pytest.mark.parametrize(("ndim", "shape"), [(2, (45, 37)), (3, (21, 18, 25))])
@pytest.mark.parametrize("block_size", [4, 8, 16])
def test_binned_render_matches_dense(backend, union, mode, ndim, shape, block_size):
    segments = _random_segments(backend, ndim, extent=min(shape))
    config = RenderConfig(mode=mode, union=union, block_size=block_size)
    binned = backend.asnumpy(backend.render(segments, shape, config=config))
    dense = backend.asnumpy(backend.render_dense(segments, shape, config=config))
    assert binned.shape == shape
    # the cutoff drops tails below sigmoid(-5) (tube) or exp(-12.5) (centreline)
    tolerance = 1e-2 if mode == "tube" else 1e-5
    assert np.abs(binned - dense).max() < tolerance
    # with a cutoff beyond the grid, the binned result is exact (the softmax
    # union applies the same cutoff in the dense reference, so it is always
    # exact)
    if union == "softmax":
        np.testing.assert_allclose(binned, dense, rtol=1e-7, atol=1e-7)
    wide = RenderConfig(mode=mode, union=union, block_size=block_size, cutoff=40.0)
    np.testing.assert_allclose(
        backend.asnumpy(backend.render(segments, shape, config=wide)),
        backend.asnumpy(backend.render_dense(segments, shape, config=wide)),
        rtol=1e-7,
        atol=1e-7,
    )


def test_small_batches_give_the_same_result(backend):
    segments = _random_segments(backend, 2)
    shape = (40, 40)
    np.testing.assert_allclose(
        backend.asnumpy(
            backend.render(
                segments, shape, config=RenderConfig(block_size=8, max_pairs=1)
            )
        ),
        backend.asnumpy(
            backend.render(segments, shape, config=RenderConfig(block_size=8))
        ),
        rtol=1e-12,
        atol=1e-12,
    )


def test_tube_profile(backend):
    segments = _line(backend, radius=3.0, opacity=0.8)
    image = backend.asnumpy(
        backend.render(segments, (21, 31), config=RenderConfig(wall_width=0.5))
    )
    # at the wall (distance == radius) the sigmoid is one half
    assert image[13, 15] == pytest.approx(0.4, rel=1e-12)
    assert image[10, 15] == pytest.approx(0.8 / (1 + np.exp(-6.0)), rel=1e-12)


def test_centreline_profile(backend):
    segments = _line(backend, radius=3.0, opacity=0.8)
    image = backend.asnumpy(
        backend.render(
            segments, (21, 31), config=RenderConfig(mode="centreline", sigma=1.5)
        )
    )
    for distance in range(4):
        expected = 0.8 * np.exp(-0.5 * (distance / 1.5) ** 2)
        assert image[10 + distance, 15] == pytest.approx(expected, rel=1e-6)


def test_soft_union_saturates(backend):
    config = RenderConfig(union="soft")
    single = backend.asnumpy(
        backend.render(_line(backend, opacity=0.6), (21, 31), config=config)
    )
    double = backend.asnumpy(
        backend.render(_line(backend, 2, opacity=0.6), (21, 31), config=config)
    )
    np.testing.assert_allclose(double, 1 - (1 - single) ** 2, rtol=1e-12, atol=1e-15)
    assert double.max() < 1


@pytest.mark.parametrize("mode", ["tube", "centreline"])
def test_max_union_does_not_depend_on_how_a_tube_is_cut(backend, mode):
    config = RenderConfig(mode=mode)
    knots = np.linspace(5.0, 25.0, 21)
    pieces = backend.segments(
        np.stack([np.full(20, 10.0), knots[:-1]], axis=1),
        np.stack([np.full(20, 10.0), knots[1:]], axis=1),
        radius=3.0,
    )
    np.testing.assert_allclose(
        backend.asnumpy(backend.render(pieces, (21, 31), config=config)),
        backend.asnumpy(
            backend.render(_line(backend, radius=3.0), (21, 31), config=config)
        ),
        rtol=1e-12,
        atol=1e-15,
    )


def test_empty_and_outside(backend):
    shape = (16, 16)
    bin_primitives = backend.fn("render", "bin_primitives")
    empty = backend.segments(np.empty((0, 2)), np.empty((0, 2)))
    np.testing.assert_array_equal(
        backend.asnumpy(backend.render(empty, shape)), np.zeros(shape)
    )
    outside = backend.segments([[100.0, 100.0]], [[110.0, 100.0]])
    np.testing.assert_array_equal(
        backend.asnumpy(backend.render(outside, shape)), np.zeros(shape)
    )
    spacing = backend.array(np.ones(2))
    assert bin_primitives(outside, shape, spacing, RenderConfig()).n_pairs == 0
    assert bin_primitives(empty, shape, spacing, RenderConfig()).n_pairs == 0


def test_dimension_mismatch(backend):
    segments = _random_segments(backend, 3)
    with pytest.raises(ValueError):
        backend.render(segments, (10, 10))


@pytest.mark.parametrize("ndim", [2, 3])
def test_rendered_ground_truth_matches_mask(backend, ndim):
    shape = (128, 128) if ndim == 2 else (64, 64, 64)
    config = airway_config(ndim, corrupted=False, shape=shape)
    for seed in range(3):
        sample = generate_sample(config, seed)
        segments = backend.LineSegments.from_graph(
            sample.graph, dtype=backend.dtype("float64")
        )
        image = backend.asnumpy(backend.render(segments, shape))
        assert _dice(image > 0.5, sample.clean_mask) > 0.95


def test_anisotropic_spacing(backend):
    nodes = np.array([[5.0, 10.0], [60.0, 50.0]])
    graph = SkeletonGraph(nodes, [[0, 1]], [nodes], [np.array([4.0, 3.0])])
    spacing = (2.0, 1.0)
    shape = (40, 64)
    mask = rasterize_graph(graph, shape, spacing=spacing)
    segments = backend.LineSegments.from_graph(graph, dtype=backend.dtype("float64"))
    image = backend.asnumpy(backend.render(segments, shape, spacing=spacing))
    assert _dice(image > 0.5, mask) > 0.95


def test_gradient_pulls_a_shifted_tube_back(backend):
    shape = (40, 40)
    target = backend.array(
        backend.asnumpy(
            backend.render(
                backend.segments([[20.0, 10.0]], [[20.0, 30.0]], radius=3.0), shape
            )
        )
    )
    shifted = backend.segments([[22.0, 10.0]], [[22.0, 30.0]], radius=3.0)

    def loss(segments):
        return ((backend.render(segments, shape) - target) ** 2).sum()

    grads = backend.segment_grads(loss, shifted)
    # gradient descent moves both endpoints back towards axis-0 position 20
    assert grads["start"][0, 0] > 0
    assert grads["end"][0, 0] > 0
    for gradient in grads.values():
        assert np.isfinite(gradient).all()


def test_softmax_union_limits_and_duplicates(backend):
    arrays = _random_arrays(2)
    segments = backend.segments(arrays.start, arrays.end, arrays.radius, arrays.opacity)
    shape = (40, 40)

    def render(segments, config=None):
        return backend.asnumpy(backend.render(segments, shape, config=config))

    maximum = render(segments)
    zero = render(segments, RenderConfig(union="softmax", temperature=0.0))
    cold = render(segments, RenderConfig(union="softmax", temperature=1e-3))
    np.testing.assert_allclose(zero, maximum, rtol=1e-12, atol=1e-15)
    assert np.abs(cold - maximum).max() < 1e-2
    # never above the max, and duplicates render the same as one copy
    warm_config = RenderConfig(union="softmax", temperature=0.5)
    warm = render(segments, warm_config)
    assert (warm <= maximum + 1e-12).all()
    doubled = arrays.concatenate(arrays)
    doubled = backend.segments(
        doubled.start, doubled.end, doubled.radius, doubled.opacity
    )
    np.testing.assert_allclose(render(doubled, warm_config), warm, rtol=1e-12)


def test_softmax_union_shares_gradients_between_near_ties(backend):
    # a thin tube inside a thick one: it never wins a voxel under the max,
    # but in the interior both are close to their opacity and share gradient
    segments = backend.segments(
        [[10.0, 5.0], [10.0, 8.0]],
        [[10.0, 25.0], [10.0, 22.0]],
        radius=np.array([3.0, 1.5]),
    )
    for union, both in (("max", False), ("softmax", True)):
        config = RenderConfig(union=union, temperature=0.2)
        grads = backend.segment_grads(
            lambda s, config=config: backend.render(s, (21, 31), config=config).sum(),
            segments,
        )
        has_gradient = np.abs(grads["opacity_logit"]) > 1e-3
        assert bool(has_gradient.all()) == both


@pytest.mark.parametrize("union", ["max", "soft", "softmax"])
def test_gradients_reach_radius_and_opacity(backend, union):
    segments = _random_segments(backend, 2)
    config = RenderConfig(union=union)
    grads = backend.segment_grads(
        lambda s: backend.render(s, (40, 40), config=config).sum(), segments
    )
    # a wider or more opaque tube covers more, so no gradient is negative; with
    # the max union, a segment hidden under others gets no gradient at all
    assert (grads["opacity_logit"] > 0).sum() >= len(segments) // 2
    if union == "softmax":
        # a weighted mean: raising a contribution well below the others' lowers
        # the mean (its weight grows), so single gradients may be negative
        assert grads["log_radius"].sum() > 0
        assert grads["opacity_logit"].sum() > 0
        return
    assert (grads["log_radius"] >= 0).all()
    assert (grads["opacity_logit"] >= 0).all()


def test_from_graph(backend):
    nodes = np.array([[0.0, 0.0], [4.0, 0.0]])
    graph = SkeletonGraph(
        nodes,
        [[0, 1]],
        [np.array([[0.0, 0.0], [2.0, 1.0], [4.0, 0.0]])],
        [np.array([3.0, 2.0, 1.0])],
    )
    segments = backend.LineSegments.from_graph(graph, opacity=0.9)
    assert len(segments) == 2
    np.testing.assert_allclose(
        backend.asnumpy(segments.start), [[0.0, 0.0], [2.0, 1.0]]
    )
    np.testing.assert_allclose(backend.asnumpy(segments.end), [[2.0, 1.0], [4.0, 0.0]])
    np.testing.assert_allclose(
        backend.asnumpy(segments.radius), [[3.0, 2.0], [2.0, 1.0]], rtol=1e-6
    )
    np.testing.assert_allclose(backend.asnumpy(segments.opacity), 0.9, rtol=1e-6)
