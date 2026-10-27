"""Tests specific to the JAX backend: pytrees, jit, autodiff and precision."""

import dataclasses
import re

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from jax.test_util import check_grads  # noqa: E402

from spleleton.geometry import pairwise_closest_points_jax  # noqa: E402
from spleleton.primitives import JaxLineSegments, SegmentArrays  # noqa: E402


@pytest.fixture
def x64():
    with jax.enable_x64(True):
        yield


def _segments(dtype=None):
    return JaxLineSegments.create(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 0.5]],
        [[4.0, 1.0, 0.0], [3.0, -1.0, 2.0]],
        radius=[[1.0, 2.0], [1.5, 0.5]],
        opacity=[0.3, 0.8],
        dtype=dtype,
    )


_POINTS = [[2.0, 2.0, 1.0], [1.0, -1.0, 0.5], [2.5, 0.5, 3.0]]


@pytest.mark.usefixtures("x64")
def test_check_grads_distance():
    # generic configuration: projections inside the segments, nonzero distances
    start = jnp.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.5]])
    end = jnp.array([[4.0, 1.0, 0.0], [3.0, -1.0, 2.0]])
    points = jnp.array(_POINTS)

    def distance(points, start, end):
        return pairwise_closest_points_jax(points, start, end).distance

    check_grads(distance, (points, start, end), order=2, modes=("fwd", "rev"))


@pytest.mark.usefixtures("x64")
def test_check_grads_query():
    segments = _segments()
    points = jnp.array(_POINTS)

    def summary(segments):
        query = segments.query(points)
        return (
            query.distance.sum()
            + query.radius.sum()
            + (segments.opacity * segments.tube_volumes()).sum()
        )

    check_grads(summary, (segments,), order=1, modes=("fwd", "rev"))


def test_segments_are_a_pytree():
    segments = _segments()
    leaves, tree = jax.tree_util.tree_flatten(segments)
    assert len(leaves) == 4
    rebuilt = jax.tree_util.tree_unflatten(tree, leaves)
    assert isinstance(rebuilt, JaxLineSegments)
    assert rebuilt.start is segments.start
    doubled = jax.tree_util.tree_map(lambda x: 2 * x, segments)
    np.testing.assert_allclose(doubled.start, 2 * np.asarray(segments.start))
    gradient = jax.grad(lambda s: s.lengths.sum())(segments)
    assert isinstance(gradient, JaxLineSegments)
    np.testing.assert_array_equal(gradient.log_radius, 0.0)


def test_jit_matches_eager():
    segments = _segments()
    points = jnp.array(_POINTS)

    def evaluate(segments, points):
        query = segments.query_indexed(points[None], jnp.array([[1, 0]]))
        return query.distance, query.radius, segments.tube_volumes()

    for eager, jitted in zip(
        evaluate(segments, points), jax.jit(evaluate)(segments, points), strict=True
    ):
        np.testing.assert_allclose(jitted, eager, rtol=1e-6)


def test_create_inside_jit_checks_shapes_but_not_values():
    @jax.jit
    def build(start, radius):
        return JaxLineSegments.create(start, start + 1, radius=radius)

    # a non-positive radius cannot be checked on a tracer
    segments = build(jnp.zeros((2, 3)), jnp.array([1.0, -1.0]))
    assert len(segments) == 2
    with pytest.raises(ValueError):
        build(jnp.zeros((2, 4)), jnp.ones(2))
    with pytest.raises(ValueError):
        JaxLineSegments.create(jnp.zeros((2, 3)), jnp.ones((2, 3)), radius=-1.0)


def test_default_precision_is_float32_without_warnings():
    # without 64-bit mode, float64 inputs become float32 silently (no warning,
    # which pytest would turn into an error)
    arrays = SegmentArrays(np.zeros((2, 3)), np.ones((2, 3)), radius=2.0)
    segments = JaxLineSegments.from_arrays(arrays)
    assert segments.start.dtype == jnp.float32
    assert segments.log_radius.dtype == jnp.float32
    assert _segments().start.dtype == jnp.float32
    assert segments.to_arrays().start.dtype == np.float64


def test_from_arrays_places_on_a_device():
    device = jax.devices("cpu")[0]
    segments = JaxLineSegments.from_arrays(
        SegmentArrays(np.zeros((2, 3)), np.ones((2, 3))), device=device
    )
    assert segments.start.devices() == {device}


def test_repr_is_short():
    assert repr(_segments()) == "JaxLineSegments(n_segments=2, dtype=float32)"


# --- renderer ---------------------------------------------------------------


def _random_arrays(ndim, n=10, extent=20.0, seed=0):
    rng = np.random.default_rng(seed)
    start = extent * rng.random((n, ndim))
    return SegmentArrays(
        start,
        start + 8 * (rng.random((n, ndim)) - 0.5),
        radius=0.5 + 2 * rng.random((n, 2)),
        opacity=0.2 + 0.7 * rng.random(n),
    )


def _reference_bins(arrays, shape, spacing, config):
    """Primitives per block, from each padded box, with NumPy loops."""
    block = config.block_size
    grid_blocks = [-(-n // block) for n in shape]
    bins = {}
    for i in range(len(arrays)):
        lower = np.minimum(arrays.start[i], arrays.end[i])
        upper = np.maximum(arrays.start[i], arrays.end[i])
        if config.mode == "tube":
            reach = arrays.radius[i].max() + config.cutoff * config.wall_width
        else:
            reach = config.cutoff * config.sigma
        first = np.maximum(np.ceil((lower - reach) / spacing), 0).astype(int)
        last = np.minimum(np.floor((upper + reach) / spacing), np.array(shape) - 1)
        last = last.astype(int)
        if np.any(first > last):
            continue
        ranges = [
            range(f // block, la // block + 1)
            for f, la in zip(first, last, strict=True)
        ]
        for multi in np.stack(np.meshgrid(*ranges, indexing="ij"), -1).reshape(
            -1, len(shape)
        ):
            bins.setdefault(int(np.ravel_multi_index(multi, grid_blocks)), set()).add(i)
    return bins


def _table(bins):
    """A block table as {block: set of primitives}, without padding."""
    table = {}
    for block, row in zip(
        np.asarray(bins.active_blocks), np.asarray(bins.primitives), strict=True
    ):
        if block < np.prod(bins.grid_blocks):
            table[int(block)] = {int(i) for i in row if i >= 0}
    return table


@pytest.mark.parametrize(
    ("shape", "spacing", "mode", "block_size"),
    [
        ((45, 37), (1.0, 1.0), "tube", 8),
        ((45, 37), (0.7, 1.6), "centreline", 4),
        ((21, 18, 25), (1.0, 1.5, 0.7), "tube", 8),
        ((21, 18, 25), (1.0, 1.0, 1.0), "centreline", 16),
    ],
)
def test_binning_matches_a_numpy_reference(shape, spacing, mode, block_size):
    from spleleton.render import RenderConfig, bin_primitives_jax

    # extent beyond the grid, so some primitives are clipped or outside
    arrays = _random_arrays(len(shape), n=30, extent=1.3 * max(shape), seed=3)
    config = RenderConfig(mode=mode, block_size=block_size)
    with jax.enable_x64(True):
        segments = JaxLineSegments.from_arrays(arrays, dtype=jnp.float64)
        bins = bin_primitives_jax(segments, shape, spacing, config)
    expected = _reference_bins(arrays, shape, np.array(spacing), config)
    assert not bool(bins.overflow)
    assert _table(bins) == expected
    assert bins.needed_capacity == (
        len(expected),
        max(len(v) for v in expected.values()),
    )


def test_bin_capacity_headroom_and_caps():
    from spleleton.render import BinCapacity

    assert BinCapacity.for_need(10, 3, n_blocks=512, n_primitives=100) == (16, 8)
    assert BinCapacity.for_need(10, 3, 512, 100, headroom=0) == (16, 8)
    assert BinCapacity.for_need(16, 9, 512, 100, headroom=0) == (16, 16)
    # never more than the grid's blocks or the number of primitives
    assert BinCapacity.for_need(30, 7, n_blocks=20, n_primitives=5) == (20, 5)
    assert BinCapacity.for_need(0, 0, n_blocks=4, n_primitives=0) == (4, 1)


@pytest.mark.usefixtures("x64")
def test_overflow_is_reported_and_exact_capacity_suffices():
    from spleleton.render import (
        BinCapacity,
        RenderConfig,
        bin_primitives_jax,
        render_jax,
    )

    shape, config = (40, 40), RenderConfig(block_size=8, union="soft")
    segments = JaxLineSegments.from_arrays(
        _random_arrays(2, n=20, extent=40.0), dtype=jnp.float64
    )
    needed = bin_primitives_jax(segments, shape, config=config).needed_capacity
    tiny = bin_primitives_jax(segments, shape, config=config, capacity=(1, 1))
    assert bool(tiny.overflow)
    assert tiny.needed_capacity == needed
    exact = bin_primitives_jax(segments, shape, config=config, capacity=needed)
    assert not bool(exact.overflow)
    narrow = BinCapacity(needed.blocks, needed.per_block - 1)
    assert bool(bin_primitives_jax(segments, shape, None, config, narrow).overflow)

    ample = render_jax(segments, shape, config=config, capacity=(25, 20))
    np.testing.assert_allclose(
        render_jax(segments, shape, config=config, capacity=needed),
        ample,
        rtol=1e-12,
        atol=1e-15,
    )
    # a truncated table drops primitives
    truncated = render_jax(segments, shape, config=config, capacity=(1, 1))
    assert float(jnp.abs(truncated - ample).max()) > 0.1


@pytest.mark.usefixtures("x64")
@pytest.mark.parametrize(
    ("mode", "union", "dense"),
    [
        ("tube", "soft", True),
        ("tube", "softmax", False),
        ("centreline", "soft", False),
        ("centreline", "softmax", True),
    ],
)
def test_check_grads_render(mode, union, dense):
    from spleleton.render import (
        RenderConfig,
        bin_capacity_jax,
        render_dense_jax,
        render_jax,
    )

    # the max union and the clamp have kinks, so check the smooth unions at
    # generic positions (voxel centres are not on the segments' ends)
    shape, spacing = (14, 11, 9), (1.0, 1.3, 0.8)
    config = RenderConfig(mode=mode, union=union, block_size=4, temperature=0.3)
    segments = JaxLineSegments.from_arrays(
        _random_arrays(3, n=4, extent=9.0, seed=5), dtype=jnp.float64
    )
    weights = jnp.asarray(np.random.default_rng(6).random(shape))
    if dense:

        def summary(segments):
            return (render_dense_jax(segments, shape, spacing, config) * weights).sum()

    else:
        capacity = bin_capacity_jax(segments, shape, spacing, config)

        def summary(segments):
            image = render_jax(segments, shape, spacing, config, capacity=capacity)
            return (image * weights).sum()

    check_grads(summary, (segments,), order=1, modes=("fwd", "rev"))


def test_render_jits_and_matches_eager():
    from spleleton.render import (
        RenderConfig,
        bin_capacity_jax,
        bin_primitives_jax,
        render_jax,
    )

    shape, config = (30, 26), RenderConfig(union="softmax", block_size=8)
    segments = JaxLineSegments.from_arrays(_random_arrays(2, extent=26.0))
    capacity = bin_capacity_jax(segments, shape, config=config)

    @jax.jit
    def render(segments, temperature):
        bins = bin_primitives_jax(segments, shape, config=config, capacity=capacity)
        image = render_jax(
            segments, shape, config=config, capacity=capacity, temperature=temperature
        )
        return image, bins

    image, bins = render(segments, 0.1)
    assert image.dtype == jnp.float32
    assert not bool(bins.overflow)
    np.testing.assert_allclose(
        image, render_jax(segments, shape, config=config), rtol=1e-6, atol=1e-7
    )
    # without a capacity, a traced call cannot size the table
    with pytest.raises(TypeError, match="capacity"):
        jax.jit(lambda s: render_jax(s, shape))(segments)
    with pytest.raises(TypeError, match="capacity"):
        jax.grad(lambda s: render_jax(s, shape).sum())(segments)


def test_temperature_changes_do_not_recompile():
    from spleleton.render import RenderConfig, bin_capacity_jax, render_jax
    from spleleton.render._render_jax import _render_jit

    shape, config = (30, 26), RenderConfig(union="softmax", block_size=8)
    segments = JaxLineSegments.from_arrays(_random_arrays(2, extent=26.0, seed=7))
    capacity = bin_capacity_jax(segments, shape, config=config)
    render_jax(segments, shape, config=config, capacity=capacity)
    compiled = _render_jit._cache_size()
    images = [
        render_jax(segments, shape, config=config, capacity=capacity, temperature=t)
        for t in np.geomspace(0.5, 0.01, 10)
    ]
    # the config's own temperature is not part of the compiled function either
    for temperature in (0.2, 0.0):
        tempered = dataclasses.replace(config, temperature=temperature)
        images.append(render_jax(segments, shape, config=tempered, capacity=capacity))
    assert _render_jit._cache_size() == compiled
    assert not np.allclose(images[0], images[-2])
    # a temperature of 0 is the max
    maximum = render_jax(
        segments, shape, config=RenderConfig(block_size=8), capacity=capacity
    )
    np.testing.assert_allclose(images[-1], maximum, rtol=1e-6)


# --- performance structure ----------------------------------------------------
# These check the compiled HLO for the two properties that make the JAX
# renderer fast on a CPU (see plans/jax_backend_plan.md, Section 13.4).


def _hlo_shapes(text):
    """Every array shape in an HLO text, as tuples of sizes."""
    return {
        tuple(int(n) for n in sizes.split(","))
        for sizes in re.findall(r"(?:f32|f64|pred)\[([\d,]+)\]", text)
    }


def test_query_has_no_coordinate_axis_per_pair():
    # 50 points x 40 segments in 3D: no (50, 40, 3) array, forward or backward
    arrays = _random_arrays(3, n=40, seed=8)
    segments = JaxLineSegments.from_arrays(arrays)
    points = jnp.asarray(np.random.default_rng(9).random((1, 50, 3)) * 20)
    index = jnp.arange(40)[None]

    def summary(segments):
        query = segments.query_indexed(points, index)
        return (query.distance + query.radius + query.t).sum()

    for function in (summary, jax.grad(summary)):
        text = jax.jit(function).lower(segments).compile().as_text()
        shapes = _hlo_shapes(text)
        assert any(shape[-2:] == (50, 40) for shape in shapes)  # the pairs exist
        assert not any(shape[-3:] == (50, 40, 3) for shape in shapes)


def test_backward_keeps_no_per_pair_values_across_batches():
    from spleleton.render import RenderConfig, bin_capacity_jax, render_jax

    shape = (24, 20, 16)
    segments = JaxLineSegments.from_arrays(_random_arrays(3, n=30, extent=20.0))
    capacity = bin_capacity_jax(segments, shape, config=RenderConfig(block_size=4))
    voxels = 4**3
    # three blocks per batch, so the blocks are evaluated in a loop
    config = RenderConfig(block_size=4, max_pairs=3 * voxels * capacity.per_block)

    def loss(segments):
        return render_jax(segments, shape, config=config, capacity=capacity).sum()

    text = jax.jit(jax.grad(loss)).lower(segments).compile().as_text()
    loops = [line for line in text.splitlines() if re.search(r"\) while\(", line)]
    assert loops
    for line in loops:
        carried = _hlo_shapes(line.split(" while(")[0])
        assert not any(
            (voxels, capacity.per_block) == shape[-2:] for shape in carried
        ), line[:200]


# --- fit and priors -----------------------------------------------------------


def _fit_problem(n=6, seed=13):
    """A 2D target (a tube) and perturbed segments along it, as NumPy."""
    target = np.zeros((40, 44))
    target[16:25, 4:40] = 1.0
    rng = np.random.default_rng(seed)
    x = np.linspace(5.0, 39.0, n + 1)
    start = np.stack([np.full(n, 20.0), x[:-1]], axis=1) + rng.normal(0, 1, (n, 2))
    end = np.stack([np.full(n, 20.0), x[1:]], axis=1) + rng.normal(0, 1, (n, 2))
    return target, SegmentArrays(start, end, radius=3.0, opacity=0.6)


@pytest.mark.parametrize("switch_to_max", [None, 0.8])
def test_temperature_schedule_compiles_the_step_once(switch_to_max):
    from spleleton.fit import FitConfig, fit_jax
    from spleleton.fit._fit_jax import _step
    from spleleton.render import RenderConfig

    target, arrays = _fit_problem()
    config = FitConfig(
        n_iterations=50,
        render=RenderConfig(union="softmax"),
        temperature=(1.0, 0.01),
        switch_to_max=switch_to_max,
    )
    _step.clear_cache()
    _, history = fit_jax(JaxLineSegments.from_arrays(arrays), target, config=config)
    assert len(history) == 50
    # once, and once more for the switch to the max union
    assert _step._cache_size() == (1 if switch_to_max is None else 2)


@pytest.mark.usefixtures("x64")
def test_overflowing_capacity_grows_and_gives_the_same_fit():
    from spleleton.fit import FitConfig, fit_jax
    from spleleton.fit._fit_jax import _step
    from spleleton.render import BinCapacity

    target, arrays = _fit_problem()
    segments = JaxLineSegments.from_arrays(arrays, dtype=jnp.float64)
    config = FitConfig(n_iterations=20)
    ample, ample_history = fit_jax(segments, target, config=config)
    _step.clear_cache()
    tiny, tiny_history = fit_jax(
        segments, target, config=config, capacity=BinCapacity(1, 1)
    )
    # compiled for the tiny capacity, then again for each larger one
    assert _step._cache_size() >= 2
    for name in ("start", "end", "log_radius", "opacity_logit"):
        np.testing.assert_allclose(
            getattr(tiny, name), getattr(ample, name), rtol=1e-12, atol=1e-12
        )
    for tiny_row, ample_row in zip(tiny_history, ample_history, strict=True):
        assert tiny_row == pytest.approx(ample_row, rel=1e-12, abs=1e-15)


@pytest.mark.usefixtures("x64")
def test_priors_jit_and_match_eager():
    from spleleton.graph import SkeletonGraph
    from spleleton.initialise import segments_from_graph
    from spleleton.priors import JaxConnectionLoss, JaxFreeEndCost

    nodes = np.array([[0.0, 20.0], [20.0, 20.0], [35.0, 8.0], [35.0, 32.0]])
    edges = [[0, 1], [1, 2], [1, 3]]
    graph = SkeletonGraph(
        nodes, edges, [nodes[[u, v]] for u, v in edges], [np.full(2, 3.0)] * 3
    )
    arrays, layout = segments_from_graph(graph, opacity=0.5)
    # gaps at the junction, so that the connection loss is not zero
    arrays = SegmentArrays(
        arrays.start + 0.3, arrays.end, radius=arrays.radius, opacity=arrays.opacity
    )
    segments = JaxLineSegments.from_arrays(arrays, dtype=jnp.float64)
    priors = [JaxFreeEndCost(layout, n_voxels=1000), JaxConnectionLoss(n_voxels=1000)]
    for prior in priors:
        hash(prior)  # a static argument of the fit step
        state = prior.update(segments, prior.init(segments), 0)
        eager = prior(segments, state)
        jitted = jax.jit(lambda segments, state, prior=prior: prior(segments, state))
        for name, value in jitted(segments, state).items():
            assert float(value) > 0
            np.testing.assert_allclose(value, eager[name], rtol=1e-12)
    # the layout only sets the state, so it is not part of the prior's identity
    _, other_layout = segments_from_graph(graph, tolerance=0.1)
    assert JaxFreeEndCost(other_layout, 1000) == priors[0]


@pytest.mark.usefixtures("x64")
def test_connection_pairs_grow_when_segments_gather():
    from spleleton.priors import JaxConnectionLoss

    # 12 segments of length 10 in a row: far apart, then with gaps of 0.5
    n = 12

    def row(spacing):
        start = np.stack([spacing * np.arange(n), np.zeros(n)], axis=1)
        arrays = SegmentArrays(start, start + np.array([10.0, 0.0]))
        return JaxLineSegments.from_arrays(arrays, dtype=jnp.float64)

    loss = JaxConnectionLoss(n_voxels=100, update_every=10)
    state = loss.init(row(20.0))
    assert not bool(state[1].any())
    initial_size = state[0].shape[0]
    gathered = row(10.5)
    # between updates the state is kept
    assert loss.update(gathered, state, 5) is state
    # at an update, each end meets the next start: 11 pairs, more than fit
    pairs, valid = loss.update(gathered, state, 10)
    expected = [[i + 1, n + i] for i in range(n - 1)]
    assert np.asarray(pairs)[np.asarray(valid)].tolist() == expected
    assert pairs.shape[0] > initial_size >= 8
    value = loss(gathered, (pairs, valid))["connection"]
    assert float(value) == pytest.approx((n - 1) * 0.25 / 100)


def test_fit_places_on_a_device():
    from spleleton.fit import FitConfig, fit_jax

    target, arrays = _fit_problem()
    device = jax.devices("cpu")[0]
    fitted, _ = fit_jax(
        JaxLineSegments.from_arrays(arrays),
        target,
        config=FitConfig(n_iterations=2),
        device=device,
    )
    assert fitted.start.devices() == {device}
    assert fitted.start.dtype == jnp.float32
