import json
from collections import Counter

import numpy as np
import pytest

from spleleton.bench.cases import build_case
from spleleton.bench.events import (
    THIN_RATIO,
    TrueCentreline,
    event_rows,
    extract_events,
    located_records,
)
from spleleton.bench.store import Store
from spleleton.bench.suites import build, errors3d_v1, load_split


def _events(name):
    sample = build_case(name).sample
    return sample, extract_events(sample)


def _by_id(events):
    return {event.event_id: event for event in events}


def test_a_clean_tube_has_only_continuous_windows():
    sample, events = _events("tube")
    assert Counter(e.event_type for e in events) == {"window": 3}
    true = TrueCentreline(sample.graph)
    windows = [e for e in events if e.event_type == "window"]
    # the windows tile the edge: about 4 r each, consecutive, every sample once
    regions = np.concatenate(
        [w.region[:-1] for w in windows[:-1]] + [windows[-1].region]
    )
    assert (regions == np.arange(len(true.points))).all()
    for window in windows:
        assert window.covariates["length"] >= 4 * 4.0 - 1e-6
        assert window.covariates["continuous"]
        assert window.covariates["nearest_record"] is None
        assert window.covariates["roughness"] == 0
        assert window.covariates["curvature"] == pytest.approx(0, abs=1e-6)
        assert window.anchors == (window.region[0], window.region[-1])
        assert window.slack == pytest.approx(2 * 4.0)


@pytest.mark.parametrize("name", ["y-thin", "y-thin-gap"])
def test_the_thin_child_is_a_thin_junction_branch(name):
    sample, events = _events(name)
    junctions = _by_id(e for e in events if e.event_type == "junction")
    assert set(junctions) == {"junction#1-0", "junction#1-1", "junction#1-2"}
    thin = junctions["junction#1-2"]
    assert thin.covariates["radius_ratio"] == pytest.approx(0.3)
    assert thin.covariates["thin"]
    assert thin.covariates["other"] == 0  # the parent, the thickest other edge
    assert thin.covariates["angle"] == pytest.approx(120, abs=1)
    assert not junctions["junction#1-0"].covariates["thin"]
    assert not junctions["junction#1-1"].covariates["thin"]
    # the anchors lie r_node + 2 r_edge from the node, on the child and the parent
    true = TrueCentreline(sample.graph)
    node = sample.graph.node_positions[1]
    a, b = thin.anchors
    expected = 6.0 + 2 * 1.8
    assert np.linalg.norm(true.points[a] - node) == pytest.approx(expected, abs=0.2)
    assert np.linalg.norm(true.points[b] - node) == pytest.approx(expected, abs=0.2)
    assert true.edge[a] == 2 and true.edge[b] == 0
    assert thin.true_distance == pytest.approx(2 * expected, abs=0.4)
    assert thin.slack == pytest.approx(2 * 6.0)


def test_the_junction_gap_breaks_continuity_near_the_junction():
    _, events = _events("y-thin-gap")
    near = [
        e.event_id
        for e in events
        if e.event_type == "window" and not e.covariates["continuous"]
    ]
    assert near
    assert all(
        e.covariates["nearest_record"] == "junction_gap#0"
        for e in events
        if e.event_id in near
    )


def test_a_bump_is_an_event_and_its_protrusion_recovers_it():
    _, events = _events("tube-bump")
    events = _by_id(events)
    bump = events["bump#0"]
    assert bump.covariates["height_over_r"] == pytest.approx(1.0)
    assert bump.covariates["size_over_r"] == pytest.approx(1.0)
    assert bump.base is not None
    (protrusion,) = [e for e in events.values() if e.event_type == "protrusion"]
    assert protrusion.event_id == "protrusion#0"
    assert protrusion.covariates["bump_inside"] == ["bump#0"]
    # the bump's tip lies 1 r outside the tube (to within a voxel)
    assert protrusion.covariates["height_over_r"] == pytest.approx(1.0, abs=0.3)
    # the windows near the bump are not continuous, and are rough
    windows = [e for e in events.values() if e.event_type == "window"]
    rough = [w for w in windows if w.covariates["roughness"] > 0]
    assert rough
    assert all(not w.covariates["continuous"] for w in rough)
    assert all(w.covariates["nearest_record"] == "bump#0" for w in rough)


def test_a_contact_bridge_is_a_protrusion_without_a_bump():
    _, events = _events("contact")
    (protrusion,) = [e for e in events if e.event_type == "protrusion"]
    assert protrusion.covariates["bump_inside"] == []
    broken = [
        e for e in events if e.event_type == "window" and not e.covariates["continuous"]
    ]
    assert {e.covariates["edge"] for e in broken} == {0, 1}
    assert all(e.covariates["nearest_record"] == "contact#0" for e in broken)


def test_records_without_a_position_are_not_located():
    sample = errors3d_v1().subsets["all"].generate(0)
    types = Counter(r["type"] for r in sample.corruptions)
    records = located_records(sample)
    assert "surface_noise" in types
    assert all(r.type != "surface_noise" for r in records)
    assert len(records) == sum(n for t, n in types.items() if t != "surface_noise")
    assert len({r.id for r in records}) == len(records)


def test_event_rows_are_plain_json():
    _, events = _events("tube-bump")
    rows = event_rows(events)
    assert [r["event_id"] for r in rows] == [e.event_id for e in events]
    json.dumps(rows)


def test_the_protrusions_recover_every_bump_on_the_quick_bump_volume():
    sample = errors3d_v1().subsets["bumps-h1.5"].generate(0)
    events = extract_events(sample)
    bumps = {e.event_id for e in events if e.event_type == "bump"}
    recovered = {
        b
        for e in events
        if e.event_type == "protrusion"
        for b in e.covariates["bump_inside"]
    }
    assert len(bumps) == 6
    assert recovered == bumps


def test_thin_junctions_on_the_asymmetric_tree():
    sample = errors3d_v1().subsets["asym-clean"].generate(0)
    events = extract_events(sample)
    thin = [e for e in events if e.event_type == "junction" and e.covariates["thin"]]
    assert thin
    assert all(e.covariates["radius_ratio"] <= THIN_RATIO for e in thin)
    # a side branch makes a junction of degree 4
    assert {e.covariates["degree"] for e in thin} == {4}


def test_event_ids_are_stable_across_rebuilds(tmp_path):
    suite = errors3d_v1()
    rows = []
    for name in ("first", "second"):
        store = Store(tmp_path / name, quick=True)
        build(store, suite, ["bumps-h1.5"], ["tune"], [0])
        (loaded,) = load_split(store, suite, "bumps-h1.5", "tune", [0])
        rows.append(event_rows(extract_events(loaded.sample)))
    generated = event_rows(extract_events(suite.subsets["bumps-h1.5"].generate(0)))
    assert rows[0] == rows[1] == generated
    ids = [r["event_id"] for r in rows[0]]
    assert len(set(ids)) == len(ids)
