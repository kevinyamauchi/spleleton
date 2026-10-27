"""The run mode of the viewer, its error browser and its checks (step 0.9).

napari runs headless here (``show=False``); the manual check (stepping
through one sample's errors, and an A/B view) is the user's.
"""

import numpy as np
import pytest

pytest.importorskip("napari")
pytest.importorskip("jax")
pytest.importorskip("pandas")

from spleleton.bench.napari import (
    ZOOM,
    anisotropic_check,
    browser_rows,
    check_view,
    error_on_target,
    record_sample,
    run_view,
    view_run,
)
from spleleton.bench.runner import run_variant
from spleleton.bench.store import Store
from spleleton.bench.suites import build, proto_cases
from spleleton.bench.variants import get_variant

pytestmark = pytest.mark.filterwarnings("ignore")  # napari's own deprecations

SUBSETS = ["tube-bump", "y-thin"]
SCOPE = {"suite": "proto-cases", "split": "cases", "subsets": SUBSETS}
SHORT = ["fit.n_rounds=2", "fit.fit.n_iterations=40"]


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    store = Store(tmp_path_factory.mktemp("bench") / "bench_quick", quick=True)
    build(store, proto_cases(), SUBSETS, ["cases"])
    return store


@pytest.fixture(scope="module")
def runs(store):
    return {
        "phase5": run_variant(
            store, get_variant("phase5", SHORT), progress=None, **SCOPE
        ),
        "skeletonize": run_variant(
            store, get_variant("skeletonize"), progress=None, **SCOPE
        ),
    }


@pytest.fixture
def opened(store, runs):
    view = run_view(store, runs["phase5"], "tube-bump", 0, compare=runs["skeletonize"])
    viewer, layers, browser = view_run(view, show=False)
    yield view, viewer, layers, browser
    viewer.close()


def test_every_layer_and_the_checks(opened):
    view, viewer, layers, _ = opened
    names = {layer.name for layer in viewer.layers}
    expected = {
        "mask", "clean mask", "rendering", "segments", "errors", "error status",
        "ground truth edges", "corruptions", f"{view.name}: clean graph edges",
        "skeletonize: clean graph edges",
    }  # fmt: skip
    assert expected <= names
    assert not viewer.layers["clean mask"].visible
    assert not viewer.layers[f"{view.name}: raw graph edges"].visible
    checks, info = check_view(view, layers)
    assert all(checks.values()), (checks, info)
    assert "every error sits on what it names" in checks


def test_an_error_off_its_target_is_caught(opened):
    view = opened[0]
    error = view.errors.iloc[0].to_dict()
    assert error_on_target(view, error) <= 1e-6
    error["position"] = list(np.asarray(error["position"]) + 3.0)
    assert error_on_target(view, error) > 1


def test_the_browser_steps_and_filters(opened):
    view, viewer, _, browser = opened
    n = len(browser.shown)
    assert n and set(browser.shown["kind"]) == {"error"}
    first = browser.next()
    assert first["id"] in viewer.canvas.overlays.text.text
    camera = viewer.scene.camera
    assert np.allclose(camera.center, first["position"])
    assert camera.zoom == pytest.approx(ZOOM * browser.base_zoom)
    assert browser.next()["id"] == browser.shown.iloc[1 % n]["id"]
    assert browser.previous()["id"] == first["id"]
    assert browser.previous()["id"] == browser.shown.iloc[-1]["id"]  # wraps around
    browser.set_filter(stage="clean")
    assert set(browser.shown["stage"]) <= {"clean"}
    browser.set_filter(kind="event", stage="")
    assert set(browser.shown["kind"]) == {"event"}
    assert len(browser.shown) == len(view.events)


def test_compare_mode_splits_the_errors(opened):
    view = opened[0]
    rows = browser_rows(view)
    statuses = set(view.matched["status"])
    assert statuses <= {"fixed", "new", "shared"} and statuses
    errors = rows[rows["kind"] == "error"]
    assert (errors["status"] != "").all()
    flips = rows[rows["kind"] == "flip"]
    assert len(flips) == len(view.flipped)
    assert flips["note"].str.startswith("null flip rate").all()


def test_a_run_against_itself_has_only_shared_errors(store, runs):
    view = run_view(store, runs["phase5"], "y-thin", 0, compare=runs["phase5"])
    assert set(view.matched["status"]) <= {"shared"}
    assert view.flipped is None or view.flipped.empty


def test_a_missing_layer_scale_is_caught():
    checks, info = anisotropic_check()
    assert all(checks.values()), info


def test_a_recording_reproduces_the_stored_fit(store, runs):
    from spleleton.recording import FitRecording

    path, checks = record_sample(store, runs["phase5"], "y-thin", 0)
    assert all(checks.values()), checks
    recording = FitRecording.load(path)
    assert recording.n_snapshots >= 2
    assert path.parent.name == "recordings"


def test_skeletonize_runs_have_no_segments(store, runs):
    view = run_view(store, runs["skeletonize"], "y-thin", 0)
    viewer, layers, _ = view_run(view, show=False, dock=False)
    try:
        assert "segments" not in layers and "rendering" not in layers
        checks, _ = check_view(view, layers)
        assert all(checks.values())
    finally:
        viewer.close()
