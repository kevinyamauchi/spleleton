import numpy as np
import pytest


def _x_axis_segment():
    return np.array([0.0, 0.0, 0.0]), np.array([10.0, 0.0, 0.0])


@pytest.mark.parametrize(
    ("point", "expected_t", "expected_point", "expected_distance"),
    [
        # projects inside the segment
        ([3.0, 4.0, 0.0], 0.3, [3.0, 0.0, 0.0], 4.0),
        # beyond the start: clamped to the start
        ([-3.0, 4.0, 0.0], 0.0, [0.0, 0.0, 0.0], 5.0),
        # beyond the end: clamped to the end
        ([13.0, 0.0, 4.0], 1.0, [10.0, 0.0, 0.0], 5.0),
        # on the segment
        ([7.0, 0.0, 0.0], 0.7, [7.0, 0.0, 0.0], 0.0),
    ],
)
def test_closest_point_on_segment(
    backend, point, expected_t, expected_point, expected_distance
):
    closest = backend.fn("geometry", "closest_points_on_segments")
    start, end = _x_axis_segment()
    result = closest(backend.array(point), backend.array(start), backend.array(end))
    np.testing.assert_allclose(backend.asnumpy(result.t), expected_t)
    np.testing.assert_allclose(backend.asnumpy(result.point), expected_point)
    np.testing.assert_allclose(backend.asnumpy(result.distance), expected_distance)


def test_degenerate_segment_behaves_like_a_point(backend):
    closest = backend.fn("geometry", "closest_points_on_segments")
    tangents = backend.fn("geometry", "segment_tangents")
    start = backend.array([1.0, 2.0, 3.0])
    result = closest(backend.array([1.0, 2.0, 7.0]), start, start)
    assert backend.asnumpy(result.t) == 0.0
    np.testing.assert_allclose(backend.asnumpy(result.point), [1.0, 2.0, 3.0])
    np.testing.assert_allclose(backend.asnumpy(result.distance), 4.0)
    np.testing.assert_array_equal(backend.asnumpy(tangents(start, start)), np.zeros(3))


def test_pairwise_shapes(backend):
    pairwise = backend.fn("geometry", "pairwise_closest_points")
    rng = np.random.default_rng(0)
    result = pairwise(
        backend.array(rng.random((7, 3))),
        backend.array(rng.random((5, 3))),
        backend.array(rng.random((5, 3))),
    )
    assert result.t.shape == (7, 5)
    assert result.distance.shape == (7, 5)
    assert result.point.shape == (7, 5, 3)


def test_matches_dense_sampling(backend):
    pairwise = backend.fn("geometry", "pairwise_closest_points")
    on_segments = backend.fn("geometry", "points_on_segments")
    lengths = backend.fn("geometry", "segment_lengths")
    rng = np.random.default_rng(0)
    points, start, end = (20 * rng.random((n, 3)) for n in (50, 8, 8))

    result = backend.asnumpy(
        pairwise(
            backend.array(points), backend.array(start), backend.array(end)
        ).distance
    )

    # brute force: densely sample every segment and take the nearest sample
    t_samples = np.linspace(0, 1, 20001)
    samples = backend.asnumpy(
        on_segments(
            backend.array(start[:, None]),
            backend.array(end[:, None]),
            backend.array(t_samples[None, :]),
        )
    )
    brute_force = np.stack(
        [
            np.linalg.norm(points[:, None] - samples[i][None], axis=-1).min(axis=1)
            for i in range(len(start))
        ],
        axis=1,
    )

    # dense sampling overestimates the distance by at most half a sample spacing
    spacing = backend.asnumpy(
        lengths(backend.array(start), backend.array(end))
    ).max() / (len(t_samples) - 1)
    assert np.all(result <= brute_force + 1e-12)
    assert np.all(brute_force - result <= spacing)


def test_tangents_are_unit_and_point_from_start_to_end(backend):
    tangents = backend.fn("geometry", "segment_tangents")
    lengths = backend.fn("geometry", "segment_lengths")
    rng = np.random.default_rng(1)
    start, end = rng.random((10, 3)), rng.random((10, 3))
    unit = backend.asnumpy(tangents(backend.array(start), backend.array(end)))
    np.testing.assert_allclose(np.linalg.norm(unit, axis=-1), np.ones(10))
    length = backend.asnumpy(lengths(backend.array(start), backend.array(end)))
    np.testing.assert_allclose(start + length[:, None] * unit, end)


def test_safe_norm_gradient_is_zero_at_zero(backend):
    safe_norm = backend.fn("geometry", "safe_norm")
    (gradient,) = backend.grad(lambda v: safe_norm(v).sum(), np.zeros((2, 3)))
    np.testing.assert_array_equal(gradient, np.zeros((2, 3)))


def test_gradient_is_finite_for_points_on_the_segment(backend):
    closest = backend.fn("geometry", "closest_points_on_segments")
    gradients = backend.grad(
        lambda points, start, end: closest(points, start, end).distance.sum(),
        np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.array([0.0, 0.0, 0.0]),
        np.array([4.0, 0.0, 0.0]),
    )
    for gradient in gradients:
        assert np.isfinite(gradient).all()


def test_distance_gradient_matches_finite_differences(backend):
    # generic configuration: projections inside the segment and nonzero distances
    pairwise = backend.fn("geometry", "pairwise_closest_points")
    start = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 0.5]])
    end = np.array([[4.0, 1.0, 0.0], [3.0, -1.0, 2.0]])
    points = np.array([[2.0, 2.0, 1.0], [1.0, -1.0, 0.5], [2.5, 0.5, 3.0]])
    weights = np.arange(1.0, 7.0).reshape(3, 2)

    def weighted_distance(points, start, end):
        distance = pairwise(points, start, end).distance
        return (distance * backend.array(weights)).sum()

    backend.check_gradient(weighted_distance, [points, start, end])
