import numpy as np
import pytest

from spleleton.primitives import SegmentArrays


def _segments(backend, **kwargs):
    start = [[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]]
    end = [[10.0, 0.0, 0.0], [0.0, 5.0, 4.0]]
    return backend.segments(start, end, **kwargs)


def test_construction_and_constrained_values(backend):
    segments = _segments(backend, radius=2.0, opacity=0.25)
    assert len(segments) == 2
    np.testing.assert_allclose(backend.asnumpy(segments.radius), np.full((2, 2), 2.0))
    np.testing.assert_allclose(backend.asnumpy(segments.opacity), np.full(2, 0.25))
    np.testing.assert_allclose(backend.asnumpy(segments.lengths), [10.0, 4.0])
    np.testing.assert_allclose(
        backend.asnumpy(segments.tangents), [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    np.testing.assert_allclose(
        backend.asnumpy(segments.midpoints), [[5.0, 0.0, 0.0], [0.0, 5.0, 2.0]]
    )
    lower, upper = segments.bounds()
    np.testing.assert_allclose(backend.asnumpy(lower), [[0, 0, 0], [0, 5, 0]])
    np.testing.assert_allclose(backend.asnumpy(upper), [[10, 0, 0], [0, 5, 4]])
    assert set(backend.parameters(segments)) == {
        "start",
        "end",
        "log_radius",
        "opacity_logit",
    }
    assert segments.backend == backend.name


@pytest.mark.parametrize(
    ("radius", "expected"),
    [
        (1.5, [[1.5, 1.5], [1.5, 1.5]]),
        (np.array([1.0, 3.0]), [[1.0, 1.0], [3.0, 3.0]]),
        (np.array([[1.0, 2.0], [3.0, 4.0]]), [[1.0, 2.0], [3.0, 4.0]]),
    ],
)
def test_radius_shapes(backend, radius, expected):
    segments = _segments(backend, radius=radius)
    np.testing.assert_allclose(backend.asnumpy(segments.radius), expected)


def test_radius_is_interpolated_along_the_segment(backend):
    segments = _segments(backend, radius=np.array([[1.0, 3.0], [2.0, 2.0]]))
    t = backend.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
    np.testing.assert_allclose(
        backend.asnumpy(segments.radius_at(t)), [[1.0, 2.0], [2.0, 2.0], [3.0, 2.0]]
    )


def test_tube_volumes(backend):
    segments = backend.segments(
        [[0.0, 0.0], [0.0, 0.0]], [[10.0, 0.0], [0.0, 0.0]], radius=np.array([2.0, 1.0])
    )
    np.testing.assert_allclose(
        backend.asnumpy(segments.tube_volumes()), [40 + 4 * np.pi, np.pi]
    )
    tube = backend.segments([[0.0, 0.0, 0.0]], [[10.0, 0.0, 0.0]], radius=2.0)
    expected = np.pi * 4 * 10 + 4 / 3 * np.pi * 8
    np.testing.assert_allclose(backend.asnumpy(tube.tube_volumes()), [expected])


def test_query(backend):
    segments = _segments(backend, radius=np.array([[1.0, 3.0], [2.0, 2.0]]))
    points = backend.array([[5.0, 2.0, 0.0], [0.0, 5.0, -1.0], [0.0, 0.0, 0.0]])
    query = segments.query(points)
    assert query.t.shape == (3, 2)
    assert query.point.shape == (3, 2, 3)
    distance, t = backend.asnumpy(query.distance), backend.asnumpy(query.t)
    np.testing.assert_allclose(distance[0, 0], 2.0)
    np.testing.assert_allclose(backend.asnumpy(query.radius)[0, 0], 2.0)
    np.testing.assert_allclose(distance[1, 1], 1.0)
    np.testing.assert_allclose(t[1, 1], 0.0)
    np.testing.assert_allclose(distance[2, 0], 0.0)


def test_query_indexed_matches_query(backend):
    rng = np.random.default_rng(0)
    segments = backend.segments(
        10 * rng.random((5, 3)), 10 * rng.random((5, 3)), radius=1 + rng.random((5, 2))
    )
    points = 10 * rng.random((2, 4, 3))  # 2 groups of 4 points
    index = np.array([[0, 2, 4], [1, 1, 3]])  # 3 segments per group
    query = segments.query_indexed(
        backend.array(points), backend.array(index, dtype="int64")
    )
    assert query.distance.shape == (2, 4, 3)
    assert query.point.shape == (2, 4, 3, 3)
    full = segments.query(backend.array(points.reshape(-1, 3)))
    for field in ("t", "distance", "radius"):
        expected = backend.asnumpy(getattr(full, field)).reshape(2, 4, 5)
        for group in range(2):
            np.testing.assert_allclose(
                backend.asnumpy(getattr(query, field))[group],
                expected[group][:, index[group]],
            )


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_dtype_is_preserved(backend, dtype):
    segments = backend.segments(np.zeros((3, 3)), np.ones((3, 3)), dtype=dtype)
    for values in backend.parameters(segments).values():
        assert values.dtype == np.dtype(dtype)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start": np.zeros((4, 4)), "end": np.zeros((4, 4))},
        {"start": np.zeros((4, 3)), "end": np.zeros((3, 3))},
        {"start": np.zeros((4, 3)), "end": np.ones((4, 3)), "radius": 0.0},
        {"start": np.zeros((4, 3)), "end": np.ones((4, 3)), "radius": np.ones(3)},
        {"start": np.zeros((4, 3)), "end": np.ones((4, 3)), "opacity": 1.0},
        {"start": np.zeros((4, 3)), "end": np.ones((4, 3)), "opacity": np.ones(2)},
    ],
)
def test_invalid_inputs(backend, kwargs):
    with pytest.raises(ValueError):
        backend.segments(**kwargs)


def test_endpoints_can_be_fitted_by_gradient_descent(backend):
    """A perturbed segment recovers a target segment from samples along it."""
    on_segments = backend.fn("geometry", "points_on_segments")
    closest = backend.fn("geometry", "closest_points_on_segments")
    target_start = np.array([2.0, 1.0, 0.0])
    target_end = np.array([8.0, 4.0, 3.0])
    t = backend.array(np.linspace(0, 1, 50))
    start_array, end_array = backend.array(target_start), backend.array(target_end)
    target_samples = on_segments(start_array, end_array, t)

    def loss(segments):
        # symmetric: the target must lie on the segment and the segment on the target
        coverage = (segments.query(target_samples).distance ** 2).mean()
        samples = on_segments(segments.start[0], segments.end[0], t)
        accuracy = (closest(samples, start_array, end_array).distance ** 2).mean()
        return coverage + accuracy

    segments = backend.segments(
        target_start[None] + np.array([[1.0, -1.0, 0.5]]),
        target_end[None] + np.array([[-0.5, 1.0, 1.0]]),
    )
    segments = backend.optimise(loss, segments, ["start", "end"], 0.05, 500)
    np.testing.assert_allclose(
        backend.asnumpy(segments.start)[0], target_start, atol=1e-2
    )
    np.testing.assert_allclose(backend.asnumpy(segments.end)[0], target_end, atol=1e-2)


def test_2d_segments(backend):
    segments = backend.segments([[0.0, 0.0], [5.0, 5.0]], [[4.0, 0.0], [5.0, 9.0]])
    query = segments.query(backend.array([[2.0, 3.0], [6.0, 7.0]]))
    # (2, 3) is 3 above segment 0 and sqrt(13) from the start (5, 5) of segment 1;
    # (6, 7) is sqrt(53) from the end (4, 0) of segment 0 and 1 beside segment 1
    np.testing.assert_allclose(
        backend.asnumpy(query.distance), [[3.0, 13.0**0.5], [53.0**0.5, 1.0]]
    )
    np.testing.assert_allclose(backend.asnumpy(segments.tangents), [[1, 0], [0, 1]])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_array_round_trip(backend, dtype):
    arrays = SegmentArrays(
        [[0.0, 0.0], [1.0, 2.0]],
        [[3.0, 4.0], [5.0, 6.0]],
        radius=[1.0, 2.0],
        opacity=0.3,
    )
    segments = backend.LineSegments.from_arrays(arrays, dtype=backend.dtype(dtype))
    assert backend.parameters(segments)["start"].dtype == np.dtype(dtype)
    np.testing.assert_allclose(
        backend.asnumpy(segments.radius), [[1.0, 1.0], [2.0, 2.0]]
    )
    back = segments.to_arrays()
    for field in ("start", "end", "radius", "opacity"):
        np.testing.assert_allclose(
            getattr(back, field), getattr(arrays, field), rtol=1e-6
        )
    assert back.start.dtype == np.float64


def test_from_graph(backend):
    from spleleton.graph import SkeletonGraph

    nodes = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0]])
    paths = [np.array([[0.0, 0.0], [5.0, 1.0], [10.0, 0.0]]), nodes[[1, 2]]]
    graph = SkeletonGraph(nodes, [[0, 1], [1, 2]], paths)
    segments = backend.LineSegments.from_graph(graph, radius=2.0, opacity=0.9)
    assert len(segments) == 3
    np.testing.assert_allclose(backend.asnumpy(segments.end)[0], [5.0, 1.0])
    np.testing.assert_allclose(backend.asnumpy(segments.radius), 2.0, rtol=1e-6)
    np.testing.assert_allclose(backend.asnumpy(segments.opacity), 0.9, rtol=1e-6)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_to_arrays_is_a_copy(backend, dtype):
    # a snapshot must not change when the segments are updated later (torch
    # updates its parameters in place)
    segments = _segments(backend, dtype=dtype)
    arrays = segments.to_arrays()
    before = {
        name: getattr(arrays, name).copy() for name in arrays.__dataclass_fields__
    }
    if backend.name == "torch":
        import torch

        with torch.no_grad():
            for parameter in backend.parameters(segments):
                getattr(segments, parameter).add_(1.0)
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(arrays, name), value)
    assert arrays.start.dtype == np.float64
