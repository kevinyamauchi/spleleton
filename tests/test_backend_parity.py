"""Torch and JAX give the same values and gradients for the same inputs.

The backends do not agree bit for bit: their log, exp, sqrt and division
differ in the last bit or two, so float64 results agree to about 1e-15
relative and float32 ones to about 1e-6. The tolerances below leave margin
for that and nothing more.
"""

import numpy as np
import pytest
from conftest import TORCH_NOT_IN_SYNC

from spleleton.primitives import SegmentArrays
from spleleton.synthetic.dataset import airway_config, generate_sample

#: float64 tolerances (relative, absolute)
RTOL, ATOL = 1e-12, 1e-12


def _assert_close(torch_value, jax_value, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(
        np.asarray(jax_value), np.asarray(torch_value), rtol=rtol, atol=atol
    )


# A segment along x, and points whose projections land exactly on its start
# (t = 0) and end (t = 1), beyond both ends, inside, and on the segment
# (distance 0); plus a degenerate segment (a point). Exact boundaries test that
# both backends split the gradient at the clamp the same way.
_START = np.array([[0.0, 0.0, 0.0], [3.0, 3.0, 3.0], [1.0, -2.0, 0.5]])
_END = np.array([[10.0, 0.0, 0.0], [3.0, 3.0, 3.0], [4.0, 2.0, -1.0]])
_POINTS = np.array(
    [
        [0.0, 3.0, 0.0],
        [10.0, 2.0, 1.0],
        [-2.0, 1.0, 0.0],
        [12.0, -1.0, 2.0],
        [4.0, 1.5, -0.5],
        [5.0, 0.0, 0.0],
    ]
)


@pytest.mark.parametrize("output", ["t", "point", "distance"])
def test_closest_points(both_backends, output):
    weights = np.random.default_rng(0).random(
        (len(_POINTS), len(_START), 3) if output == "point" else (len(_POINTS), 3)
    )
    results = []
    for backend in both_backends:
        pairwise = backend.fn("geometry", "pairwise_closest_points")

        def summary(points, start, end, pairwise=pairwise, backend=backend):
            value = getattr(pairwise(points, start, end), output)
            return (value * backend.array(weights)).sum()

        arrays = [backend.array(x) for x in (_POINTS, _START, _END)]
        value = backend.asnumpy(getattr(pairwise(*arrays), output))
        results.append((value, backend.grad(summary, _POINTS, _START, _END)))
    (torch_value, torch_grads), (jax_value, jax_grads) = results
    _assert_close(torch_value, jax_value)
    for torch_grad, jax_grad in zip(torch_grads, jax_grads, strict=True):
        _assert_close(torch_grad, jax_grad)


@pytest.mark.parametrize(
    "name", ["segment_lengths", "segment_tangents", "points_on_segments"]
)
def test_segment_functions(both_backends, name):
    rng = np.random.default_rng(1)
    t = np.array([0.0, 0.3, 0.5, 1.0])
    inputs = [_START, _END] + ([t[:3]] if name == "points_on_segments" else [])
    weights = rng.random(_START.shape if name != "segment_lengths" else 3)
    results = []
    for backend in both_backends:
        function = backend.fn("geometry", name)

        def summary(*arrays, function=function, backend=backend):
            return (function(*arrays) * backend.array(weights)).sum()

        value = backend.asnumpy(function(*(backend.array(x) for x in inputs)))
        results.append((value, backend.grad(summary, *inputs)))
    (torch_value, torch_grads), (jax_value, jax_grads) = results
    _assert_close(torch_value, jax_value)
    for torch_grad, jax_grad in zip(torch_grads, jax_grads, strict=True):
        _assert_close(torch_grad, jax_grad)


def _random_arrays(ndim: int, n: int = 12, seed: int = 0) -> SegmentArrays:
    rng = np.random.default_rng(seed)
    return SegmentArrays(
        20 * rng.random((n, ndim)),
        20 * rng.random((n, ndim)),
        radius=0.5 + 3 * rng.random((n, 2)),
        opacity=0.05 + 0.9 * rng.random(n),
    )


@pytest.mark.parametrize("ndim", [2, 3])
def test_segment_properties_and_queries(both_backends, ndim):
    arrays = _random_arrays(ndim)
    rng = np.random.default_rng(2)
    points = 20 * rng.random((4, 6, ndim))
    index = rng.integers(0, len(arrays), size=(4, 5))
    t = rng.random((3, len(arrays)))

    values, grads = [], []
    for backend in both_backends:
        segments = backend.LineSegments.from_arrays(
            arrays, dtype=backend.dtype("float64")
        )
        point_array = backend.array(points)
        index_array = backend.array(index, dtype="int64")
        t_array = backend.array(t)

        def evaluate(
            segments, point_array=point_array, index_array=index_array, t_array=t_array
        ):
            query = segments.query_indexed(point_array, index_array)
            full = segments.query(point_array.reshape(-1, ndim))
            lower, upper = segments.bounds()
            return {
                "radius": segments.radius,
                "opacity": segments.opacity,
                "lengths": segments.lengths,
                "tangents": segments.tangents,
                "midpoints": segments.midpoints,
                "tube_volumes": segments.tube_volumes(),
                "radius_at": segments.radius_at(t_array),
                "bounds": upper - lower,
                "query_t": query.t,
                "query_point": query.point,
                "query_distance": query.distance,
                "query_radius": query.radius,
                "full_distance": full.distance,
                "full_radius": full.radius,
            }

        def summary(segments, evaluate=evaluate):
            return sum(value.sum() for value in evaluate(segments).values())

        values.append(
            {key: backend.asnumpy(v) for key, v in evaluate(segments).items()}
        )
        grads.append(backend.segment_grads(summary, segments))

    for key in values[0]:
        _assert_close(values[0][key], values[1][key], atol=1e-11)
    for key in grads[0]:
        _assert_close(grads[0][key], grads[1][key], rtol=1e-11, atol=1e-10)


def test_float32_construction(both_backends):
    arrays = _random_arrays(3, n=100)
    stored = []
    for backend in both_backends:
        segments = backend.LineSegments.from_arrays(
            arrays, dtype=backend.dtype("float32")
        )
        stored.append(backend.parameters(segments))
    torch_values, jax_values = stored
    for key, value in torch_values.items():
        assert jax_values[key].dtype == value.dtype == np.float32
        # endpoints are copied, so identical; logs differ in the last bit
        if key in ("start", "end"):
            np.testing.assert_array_equal(jax_values[key], value)
        else:
            np.testing.assert_allclose(jax_values[key], value, rtol=1e-6, atol=1e-6)


# --- renderer ----------------------------------------------------------------

#: renderer tolerance (relative and absolute)
RENDER_TOL = 1e-10

_RENDER_CASES = [
    ((45, 37), (1.0, 1.0), 8),
    ((21, 18, 25), (1.0, 1.5, 0.7), 4),
]


def _render_arrays(shape, spacing, n=14, seed=4) -> SegmentArrays:
    """Random segments over (and a little beyond) a grid, in physical units."""
    rng = np.random.default_rng(seed)
    extent = np.array(shape) * np.array(spacing)
    start = 1.1 * extent * rng.random((n, len(shape)))
    return SegmentArrays(
        start,
        start + 8 * (rng.random((n, len(shape))) - 0.5),
        radius=0.5 + 3 * rng.random((n, 2)),
        opacity=0.05 + 0.9 * rng.random(n),
    )


def _renderer(backend, segments, shape, spacing, config, dense):
    """The backend's render as a function of the segments (JAX: jittable)."""
    if dense:
        function = backend.fn("render", "render_dense")
        return lambda s: function(s, shape, spacing, config)
    function = backend.fn("render", "render")
    if backend.name == "jax":
        capacity = backend.fn("render", "bin_capacity")(
            segments, shape, spacing, config
        )
        return lambda s: function(s, shape, spacing, config, capacity=capacity)
    return lambda s: function(s, shape, spacing, config)


@pytest.mark.parametrize(("shape", "spacing", "block_size"), _RENDER_CASES)
@pytest.mark.parametrize("mode", ["tube", "centreline"])
def test_binning(both_backends, shape, spacing, block_size, mode):
    from spleleton.render import RenderConfig

    arrays = _render_arrays(shape, spacing, n=30)
    config = RenderConfig(mode=mode, block_size=block_size)
    tables = []
    for backend in both_backends:
        segments = backend.LineSegments.from_arrays(
            arrays, dtype=backend.dtype("float64")
        )
        bins = backend.fn("render", "bin_primitives")(
            segments, shape, backend.array(spacing), config
        )
        n_blocks = np.prod(bins.grid_blocks)
        tables.append(
            {
                int(block): {int(i) for i in row if i >= 0}
                for block, row in zip(
                    backend.asnumpy(bins.active_blocks),
                    backend.asnumpy(bins.primitives),
                    strict=True,
                )
                if block < n_blocks
            }
        )
    assert tables[0] == tables[1]


@pytest.mark.parametrize("dense", [False, True], ids=["binned", "dense"])
@pytest.mark.parametrize(("shape", "spacing", "block_size"), _RENDER_CASES)
@pytest.mark.parametrize(
    ("union", "temperature"), [("max", 0.1), ("soft", 0.1), ("softmax", 0.3)]
)
@pytest.mark.parametrize("mode", ["tube", "centreline"])
def test_render(
    both_backends, mode, union, temperature, shape, spacing, block_size, dense
):
    from spleleton.render import RenderConfig

    arrays = _render_arrays(shape, spacing)
    config = RenderConfig(
        mode=mode, union=union, temperature=temperature, block_size=block_size
    )
    weights = np.random.default_rng(5).random(shape)
    values, grads = [], []
    for backend in both_backends:
        segments = backend.LineSegments.from_arrays(
            arrays, dtype=backend.dtype("float64")
        )
        render = _renderer(backend, segments, shape, spacing, config, dense)
        weight_array = backend.array(weights)

        def summary(segments, render=render, weight_array=weight_array):
            return (render(segments) * weight_array).sum()

        values.append(backend.asnumpy(render(segments)))
        grads.append(backend.segment_grads(summary, segments))
    assert values[0].max() > 0.5  # the case is not empty
    _assert_close(values[0], values[1], rtol=RENDER_TOL, atol=RENDER_TOL)
    for key in grads[0]:
        # the centreline kernel ignores the radius
        if not (mode == "centreline" and key == "log_radius"):
            assert np.abs(grads[0][key]).max() > 0
        _assert_close(grads[0][key], grads[1][key], rtol=RENDER_TOL, atol=RENDER_TOL)


@pytest.mark.parametrize("union", ["max", "soft", "softmax"])
@pytest.mark.parametrize("mode", ["tube", "centreline"])
def test_render_in_batches(both_backends, mode, union):
    """The batched path (a loop over batches, rematerialised in JAX)."""
    from spleleton.render import RenderConfig

    shape, spacing = (21, 18, 25), (1.0, 1.5, 0.7)
    arrays = _render_arrays(shape, spacing)
    # a few blocks per batch, so both backends loop over batches
    config = RenderConfig(
        mode=mode, union=union, temperature=0.3, block_size=4, max_pairs=2 * 64 * 16
    )
    weights = np.random.default_rng(6).random(shape)
    values, grads = [], []
    for backend in both_backends:
        segments = backend.LineSegments.from_arrays(
            arrays, dtype=backend.dtype("float64")
        )
        render = _renderer(backend, segments, shape, spacing, config, dense=False)
        weight_array = backend.array(weights)

        def summary(segments, render=render, weight_array=weight_array):
            return (render(segments) * weight_array).sum()

        values.append(backend.asnumpy(render(segments)))
        grads.append(backend.segment_grads(summary, segments))
    _assert_close(values[0], values[1], rtol=RENDER_TOL, atol=RENDER_TOL)
    for key in grads[0]:
        _assert_close(grads[0][key], grads[1][key], rtol=RENDER_TOL, atol=RENDER_TOL)


# --- losses, priors and the fit ------------------------------------------------

#: fit tolerances (relative, absolute) per dtype, after n iterations
FIT_TOL = {"float64": 1e-8, "float32": 1e-4}


def _y_tree(shape=(48, 44), spacing=(1.0, 1.2)):
    """A Y-shaped tree: its mask, and perturbed segments with their layout."""
    from spleleton.graph import SkeletonGraph
    from spleleton.initialise import segments_from_graph
    from spleleton.synthetic.rasterize import rasterize_graph

    nodes = np.array([[6.0, 26.0], [24.0, 26.0], [40.0, 10.0], [40.0, 42.0]])
    edges = [[0, 1], [1, 2], [1, 3]]
    # slightly curved edges, so that each becomes a chain of segments
    bend = np.sin(np.linspace(0, np.pi, 5))[:, None] * np.array([1.5, -1.5])
    paths = [np.linspace(nodes[u], nodes[v], 5) + bend for u, v in edges]
    graph = SkeletonGraph(nodes, edges, paths, [np.full(5, 3.0)] * 3)
    mask = rasterize_graph(graph, shape, spacing=spacing)
    arrays, layout = segments_from_graph(graph, tolerance=0.1, opacity=0.6)
    rng = np.random.default_rng(10)
    arrays = SegmentArrays(
        arrays.start + rng.normal(0, 1.0, arrays.start.shape),
        arrays.end + rng.normal(0, 1.0, arrays.end.shape),
        radius=arrays.radius * (1 + 0.2 * rng.random(arrays.radius.shape)),
        opacity=arrays.opacity,
    )
    return mask.astype(float), arrays, layout


def _priors(backend, layout, n_voxels):
    from spleleton import priors

    prefix = backend.name.capitalize()
    return [
        getattr(priors, f"{prefix}FreeEndCost")(layout, n_voxels, weight=3.0),
        getattr(priors, f"{prefix}ConnectionLoss")(
            n_voxels, weight=1.0, update_every=5
        ),
    ]


@pytest.mark.parametrize("name", ["bce_loss", "dice_loss"])
def test_losses(both_backends, name):
    rng = np.random.default_rng(11)
    occupancy = rng.random((9, 7))
    occupancy[0, :2] = [0.0, 1.0]  # clamped
    target = (rng.random((9, 7)) > 0.5).astype(float)
    results = []
    for backend in both_backends:
        loss = backend.fn("fit", name)

        def value(occupancy, target, loss=loss):
            return loss(occupancy, target)

        results.append(
            (
                backend.asnumpy(value(backend.array(occupancy), backend.array(target))),
                backend.grad(value, occupancy, target),
            )
        )
    (torch_value, torch_grads), (jax_value, jax_grads) = results
    _assert_close(torch_value, jax_value, rtol=RENDER_TOL, atol=RENDER_TOL)
    for torch_grad, jax_grad in zip(torch_grads, jax_grads, strict=True):
        _assert_close(torch_grad, jax_grad, rtol=RENDER_TOL, atol=RENDER_TOL)


def test_endpoint_pairs(both_backends):
    rng = np.random.default_rng(12)
    arrays = SegmentArrays(
        30 * rng.random((40, 2)),
        30 * rng.random((40, 2)),
        radius=0.5 + rng.random((40, 2)),
    )
    found = []
    for backend in both_backends:
        segments = backend.LineSegments.from_arrays(
            arrays, dtype=backend.dtype("float64")
        )
        pairs = backend.fn("priors", "endpoint_pairs")(segments, 1.5)
        if backend.name == "jax":
            pairs, valid = pairs
            pairs = pairs[valid]
        found.append(backend.asnumpy(pairs).tolist())
    assert len(found[0]) > 10
    assert found[0] == found[1]


@pytest.mark.parametrize("prior", [0, 1], ids=["free_end", "connection"])
def test_priors(both_backends, prior):
    mask, arrays, layout = _y_tree()
    values, grads = [], []
    for backend in both_backends:
        segments = backend.LineSegments.from_arrays(
            arrays, dtype=backend.dtype("float64")
        )
        term = _priors(backend, layout, mask.size)[prior]

        def summary(segments, backend=backend, term=term):
            return sum(backend.prior_terms(term, segments).values())

        values.append(float(backend.asnumpy(summary(segments))))
        grads.append(backend.segment_grads(summary, segments))
    assert values[0] > 0
    _assert_close(values[0], values[1], rtol=RENDER_TOL, atol=RENDER_TOL)
    for key in grads[0]:
        _assert_close(grads[0][key], grads[1][key], rtol=RENDER_TOL, atol=RENDER_TOL)


def test_sparsity_term(both_backends):
    # the fit's sparsity term after one iteration, through the history
    from spleleton.fit import FitConfig

    mask, arrays, _ = _y_tree()
    sparsity = []
    for backend in both_backends:
        segments = backend.LineSegments.from_arrays(
            arrays, dtype=backend.dtype("float64")
        )
        _, history = backend.fit(
            segments, mask, (1.0, 1.2), config=FitConfig(n_iterations=1)
        )
        sparsity.append(history[0]["sparsity"])
    assert sparsity[0] > 0
    _assert_close(sparsity[0], sparsity[1], rtol=RENDER_TOL, atol=RENDER_TOL)


@pytest.mark.parametrize(
    ("dtype", "n_iterations"), [("float64", 1), ("float64", 20), ("float32", 20)]
)
@pytest.mark.parametrize("union", ["max", "soft", "softmax"])
def test_fit(both_backends, union, dtype, n_iterations):
    from spleleton.fit import FitConfig, default_losses
    from spleleton.render import RenderConfig

    mask, arrays, layout = _y_tree()
    # the softmax union with a temperature schedule and the switch to the max
    schedule = (
        {"temperature": (0.5, 0.05), "switch_to_max": 0.75}
        if union == "softmax"
        else {}
    )
    config = FitConfig(
        n_iterations=n_iterations,
        losses=default_losses(sparsity=0.1),
        render=RenderConfig(union=union, block_size=8),
        **schedule,
    )
    fitted, histories = [], []
    for backend in both_backends:
        segments = backend.LineSegments.from_arrays(arrays, dtype=backend.dtype(dtype))
        segments, history = backend.fit(
            segments,
            mask,
            (1.0, 1.2),
            config=config,
            regularisers=_priors(backend, layout, mask.size),
        )
        fitted.append(backend.parameters(segments))
        histories.append(history)
    tolerance = FIT_TOL[dtype]
    assert len(arrays) > 6
    assert np.abs(fitted[0]["start"] - arrays.start).max() > 0.05  # the fit moved
    for key, value in fitted[0].items():
        assert fitted[1][key].dtype == value.dtype
        _assert_close(value, fitted[1][key], rtol=tolerance, atol=tolerance)
    assert set(histories[0][0]) == set(histories[1][0])
    for torch_row, jax_row in zip(*histories, strict=True):
        for key, value in torch_row.items():
            _assert_close(value, jax_row[key], rtol=tolerance, atol=tolerance)


# --- pipelines ---------------------------------------------------------------
# Float32 fits from a real initialisation do not have a reproducible
# trajectory: a 1-ULP change of the input moves segments by about a voxel
# within 10 iterations, in torch alone (Section 13.7 of the JAX plan). So the
# phase-5 test uses short rounds, where both backends still agree, and the
# phase-4 test compares the JAX pipeline's distance from torch with torch's
# own distance from a run with the spacing scaled by 1 + 2^-22.

_NUDGE = 1 + 2.0**-22


def _pipeline_runs(name, config, seeds, nudge=False):
    from spleleton.metrics import compare_graphs
    from spleleton.pipeline import _pipeline_jax, _pipeline_torch

    runs = []
    for seed in seeds:
        sample = generate_sample(airway_config(2, shape=(96, 96)), seed)
        spacing = tuple(sample.spacing)
        pipelines = {
            "torch": (getattr(_pipeline_torch, f"{name}_torch"), spacing),
            "jax": (getattr(_pipeline_jax, f"{name}_jax"), spacing),
        }
        if nudge:
            nudged = tuple(step * _NUDGE for step in spacing)
            pipelines["nudged"] = (pipelines["torch"][0], nudged)
        run = {}
        for label, (pipeline, steps) in pipelines.items():
            graph = pipeline(sample.mask, steps, config).graph
            run[label] = (graph, compare_graphs(graph, sample.graph).to_dict())
        runs.append(run)
    return runs


def _size(graph):
    return graph.n_nodes, graph.n_edges, graph.n_components()


@pytest.mark.skip(reason=TORCH_NOT_IN_SYNC)
@pytest.mark.usefixtures("both_backends")
def test_scratch_fit_pipelines_agree():
    from spleleton.fit import FitConfig, default_losses
    from spleleton.pipeline import ScratchFitConfig

    config = ScratchFitConfig(
        fit=FitConfig(n_iterations=60, losses=default_losses(sparsity=0.5)), n_rounds=2
    )
    runs = _pipeline_runs("scratch_fit", config, seeds=range(3))
    assert sum(run["torch"][0].n_edges for run in runs) >= 6  # not trivial
    for run in runs:
        (torch_graph, torch_scores), (jax_graph, jax_scores) = run["torch"], run["jax"]
        assert _size(jax_graph) == _size(torch_graph)
        for key in ("centreline_f1", "junction_f1", "radius_error"):
            assert jax_scores[key] == pytest.approx(torch_scores[key], abs=0.02)


@pytest.mark.skip(reason=TORCH_NOT_IN_SYNC)
@pytest.mark.usefixtures("both_backends")
def test_skeleton_fit_pipelines_agree_within_torch_noise():
    from spleleton.fit import FitConfig, default_losses
    from spleleton.pipeline import SkeletonFitConfig

    config = SkeletonFitConfig(
        fit=FitConfig(n_iterations=100, losses=default_losses(sparsity=0.25))
    )
    runs = _pipeline_runs("skeleton_fit", config, seeds=range(6), nudge=True)

    def same_size(label):
        return sum(_size(run[label][0]) == _size(run["torch"][0]) for run in runs)

    assert same_size("jax") >= same_size("nudged")
    for key in ("centreline_f1", "junction_f1"):

        def gap(label, key=key):
            return np.mean(
                [abs(run[label][1][key] - run["torch"][1][key]) for run in runs]
            )

        assert gap("jax") <= 2 * gap("nudged") + 0.01, key
