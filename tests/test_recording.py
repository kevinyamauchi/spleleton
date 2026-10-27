import numpy as np
import pytest

from spleleton.fit import FitConfig
from spleleton.graph import SkeletonGraph
from spleleton.initialise import SegmentLayout, segments_from_graph
from spleleton.primitives import SegmentArrays
from spleleton.recording import FitRecorder, FitRecording
from spleleton.synthetic.dataset import airway_config, generate_sample


def _problem(backend):
    nodes = np.array([[5.0, 20.0], [20.0, 20.0], [32.0, 10.0], [32.0, 30.0]])
    edges = [[0, 1], [1, 2], [1, 3]]
    truth = SkeletonGraph(
        nodes,
        edges,
        [nodes[[u, v]] for u, v in edges],
        [np.array([3.0, 3.0]), np.array([2.5, 2.5]), np.array([2.5, 2.5])],
    )
    arrays, layout = segments_from_graph(truth)
    float32 = backend.dtype("float32")
    shifted = SegmentArrays(
        arrays.start + 1.0, arrays.end, radius=arrays.radius, opacity=arrays.opacity
    )
    segments = backend.LineSegments.from_arrays(shifted, dtype=float32)
    truth_segments = backend.LineSegments.from_arrays(arrays, dtype=float32)
    rendering = backend.asnumpy(backend.render(truth_segments, (40, 40)))
    target = (rendering > 0.5).astype(np.float32)
    return truth, segments, layout, target


@pytest.mark.parametrize(("n_iterations", "every"), [(25, 10), (20, 10), (5, 1)])
def test_recorder_takes_snapshots_and_the_final_state(backend, n_iterations, every):
    _, segments, _, target = _problem(backend)
    recorder = FitRecorder(target.shape, every=every)
    segments, history = backend.fit(
        segments, target, config=FitConfig(n_iterations=n_iterations), callback=recorder
    )
    recording = recorder.finish(history, target=target)
    expected = list(range(0, n_iterations + 1, every))
    if expected[-1] != n_iterations:
        expected.append(n_iterations)
    assert list(recording.iterations) == expected
    n = len(expected)
    assert recording.start.shape == (n, len(segments), 2)
    assert recording.radius.shape == (n, len(segments), 2)
    assert recording.opacity.shape == (n, len(segments))
    assert recording.renderings.shape == (n, 40, 40)
    assert recording.renderings.dtype == np.float16
    assert set(recording.loss) == {"loss", "bce", "dice", "sparsity"}
    assert len(recording.loss["loss"]) == n_iterations
    # the last snapshot is the fitted state
    np.testing.assert_allclose(recording.start[-1], backend.asnumpy(segments.start))
    np.testing.assert_allclose(
        recording.opacity[-1], backend.asnumpy(segments.opacity), rtol=1e-6
    )
    # earlier snapshots are earlier states, not views of the live segments
    assert not np.allclose(recording.start[0], recording.start[-1])


def test_recording_round_trip(backend, tmp_path):
    truth, segments, layout, target = _problem(backend)
    recorder = FitRecorder(target.shape, every=5, spacing=(1.0, 1.0))
    segments, history = backend.fit(
        segments, target, config=FitConfig(n_iterations=12), callback=recorder
    )
    recording = recorder.finish(
        history, target=target, layout=layout, ground_truth=truth
    )
    path = recording.save(tmp_path / "fit_sparsity_0.1")
    assert path.name == "fit_sparsity_0.1.npz"
    loaded = FitRecording.load(path)
    for key in (
        "iterations",
        "start",
        "end",
        "radius",
        "opacity",
        "renderings",
        "target",
    ):
        np.testing.assert_array_equal(getattr(loaded, key), getattr(recording, key))
    for key, values in recording.loss.items():
        np.testing.assert_array_equal(loaded.loss[key], values)
    assert loaded.spacing == (1.0, 1.0)
    np.testing.assert_array_equal(
        loaded.ground_truth.node_positions, truth.node_positions
    )
    # the graph rebuilt from the last snapshot matches the live one
    last = loaded.graph(loaded.n_snapshots - 1)
    live = layout.to_graph(segments)
    np.testing.assert_allclose(last.node_positions, live.node_positions, atol=1e-5)


def test_recording_without_renderings_or_context(backend, tmp_path):
    _, segments, _, target = _problem(backend)
    recorder = FitRecorder(target.shape, every=3, render_snapshots=False)
    backend.fit(segments, target, config=FitConfig(n_iterations=4), callback=recorder)
    recording = recorder.finish([])
    assert recording.renderings is None
    assert recording.loss == {}
    loaded = FitRecording.load(recording.save(tmp_path / "bare"))
    assert loaded.renderings is None and loaded.layout is None and loaded.target is None
    with pytest.raises(ValueError):
        loaded.graph(0)


def test_recorder_validation():
    with pytest.raises(ValueError):
        FitRecorder((10, 10), every=0)
    with pytest.raises(ValueError):
        FitRecorder((10, 10)).finish([])


def test_layout_round_trip():
    truth = SkeletonGraph(
        np.array([[5.0, 20.0], [20.0, 20.0], [32.0, 10.0]]),
        [[0, 1], [1, 2]],
        [np.array([[5.0, 20.0], [20.0, 20.0]]), np.array([[20.0, 20.0], [32.0, 10.0]])],
        [np.array([3.0, 3.0]), np.array([2.5, 2.5])],
    )
    _, layout = segments_from_graph(truth)
    restored = SegmentLayout.from_dict(layout.to_dict())
    assert restored.n_nodes == layout.n_nodes
    np.testing.assert_array_equal(restored.edges, layout.edges)
    for a, b in zip(restored.edge_segments, layout.edge_segments, strict=True):
        np.testing.assert_array_equal(a, b)


def test_sample_save_keeps_dots_in_names(tmp_path):
    sample = generate_sample(airway_config(2, corrupted=False, shape=(48, 48)), 0)
    assert sample.save(tmp_path / "noise_0.5").name == "noise_0.5.npz"


@pytest.mark.filterwarnings("ignore")
def test_view_fit_builds_layers(backend):
    napari = pytest.importorskip("napari")
    from spleleton.viz import view_fit

    truth, segments, layout, target = _problem(backend)
    recorder = FitRecorder(target.shape, every=5)
    segments, history = backend.fit(
        segments, target, config=FitConfig(n_iterations=10), callback=recorder
    )
    recording = recorder.finish(
        history, target=target, layout=layout, ground_truth=truth
    )
    viewer = napari.Viewer(show=False)
    try:
        view_fit(recording, viewer)
        names = [layer.name for layer in viewer.layers]
        assert names[:3] == ["target", "rendering", "segments"]
        assert "fitted graph" in names
        assert viewer.dims.nsteps[0] == recording.n_snapshots
        assert viewer.text_overlay.text.startswith("iteration 0")
        viewer.dims.set_current_step(0, recording.n_snapshots - 1)
        assert viewer.text_overlay.text.startswith("iteration 10")
    finally:
        viewer.close()
