"""Outcomes, fit classes, decision quantities, errors and causes (step 0.7).

The known answers of ``proto-cases`` are stage 0's (P3), with two changes:
gap events are gone (gaps are out of scope), and the ``contact`` case's raw
graph no longer breaks under the false join (the ``_drop_contained`` fix).
"""

from functools import cache

import numpy as np
import pytest

from spleleton.bench.cases import _TUBE, _from_edges, _tube_graph, build_case
from spleleton.bench.decisions import _bottleneck, segment_links
from spleleton.bench.events import (
    STAGES,
    TrueCentreline,
    extract_events,
    score_sample,
)
from spleleton.bench.suites import errors3d_v1
from spleleton.metrics import compare_graphs
from spleleton.pipeline import GraphConfig, graph_stage
from spleleton.primitives import SegmentArrays

JOIN = GraphConfig().join


@cache
def _scored(case_name, prediction):
    case = build_case(case_name)
    return _score(case.sample, case.predictions[prediction])


def _score(sample, arrays):
    config = GraphConfig()
    raw, clean = graph_stage(arrays, None, sample.mask, sample.spacing, config)
    return score_sample(sample, raw, clean, arrays, config)


def _events(case, prediction, kind=None, stage="raw"):
    return [
        r
        for r in _scored(case, prediction).events
        if r["stage"] == stage and (kind is None or r["event_type"] == kind)
    ]


def _errors(case, prediction, stage, kind=None):
    return [
        e
        for e in _scored(case, prediction).errors
        if e["stage"] == stage and (kind is None or e["type"] == kind)
    ]


def _sample_row(case, prediction, stage):
    (row,) = [r for r in _scored(case, prediction).samples if r["stage"] == stage]
    return row


CASES = ("tube", "tube-gap", "y-thin", "y-thin-gap", "tube-bump", "contact")


# -- the true predictions --------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("stage", STAGES)
def test_the_true_segments_make_no_error(case, stage):
    assert _errors(case, "true", stage) == []
    connection = [r for r in _events(case, "true", stage=stage) if "connected" in r]
    assert connection and all(r["connected"] for r in connection)


@pytest.mark.parametrize("case", CASES)
def test_the_true_segments_link_below_join(case):
    rows = [r for r in _events(case, "true") if "q_join" in r]
    assert rows and all(r["q_join"] < JOIN for r in rows)
    assert all(r["tau_join"] == JOIN for r in rows)


def test_the_true_segments_of_clean_score_perfectly():
    sample = errors3d_v1().subsets["clean"].generate(0)
    scores = _score(sample, SegmentArrays.from_graph(sample.graph))
    assert scores.errors == []
    for row in scores.samples:
        assert row["centreline_f1"] == pytest.approx(1.0)
        assert row["junction_f1"] == pytest.approx(1.0)
        assert row["root_connected_fraction"] == pytest.approx(1.0)
    connection = [r for r in scores.events if "connected" in r]
    assert all(r["connected"] for r in connection)
    junctions = [r for r in scores.events if r["event_type"] == "junction"]
    assert {r["fit_class"] for r in junctions} == {"reaches"}
    assert {r["class"] for r in junctions} == {"connected"}


# -- planted errors --------------------------------------------------------------------


def test_a_split_tube_breaks_one_continuous_window():
    broken = [r for r in _events("tube", "split", "window") if not r["connected"]]
    assert len(broken) == 1 and broken[0]["continuous"]
    assert broken[0]["q_join"] == pytest.approx(2.0, abs=0.05)  # 2 r / r
    assert broken[0]["q_bridge"] == pytest.approx(0.0, abs=0.05)  # the tubes touch
    assert broken[0]["fit_covered"]
    ends = _errors("tube", "split", "raw", "break_end")
    assert [e["cause"] for e in ends] == ["unexplained", "unexplained"]
    assert all(e["window_id"] == broken[0]["event_id"] for e in ends)
    assert _sample_row("tube", "split", "raw")["n_algorithmic_breaks"] == 2
    # clean-up bridges it
    assert all(r["connected"] for r in _events("tube", "split", "window", "clean"))
    assert _errors("tube", "split", "clean") == []
    # the root-connected fraction weighs the break by what it cuts off
    assert _sample_row("tube", "split", "raw")["root_connected_fraction"] < 0.5
    assert _sample_row("tube", "split", "clean")["root_connected_fraction"] == 1


@pytest.mark.parametrize("stage", STAGES)
def test_a_shifted_tube_is_off_centre_by_its_segments(stage):
    errors = _errors("tube", "shift", stage)
    assert {e["type"] for e in errors} == {"off_centre"}
    assert {e["source"] for e in errors} == {"segment"}
    for error in errors:
        assert error["max_offset_over_r"] == pytest.approx(0.4, abs=0.02)
    deviations = [r["deviation"] for r in _events("tube", "shift", "window", stage)]
    assert deviations == pytest.approx([0.4] * len(deviations), abs=0.02)


def test_break_ends_at_a_gap_are_caused_by_it():
    ends = _errors("tube-gap", "split", "raw", "break_end")
    assert [e["cause"] for e in ends] == ["gap#0", "gap#0"]
    # the mask itself has lost the stretch in the gap
    (missed,) = _errors("tube-gap", "split", "raw", "missed")
    assert missed["cause"] == "mask_lost"
    assert missed["in_main_component"] < 0.5
    # where the mask holds the tube, a missed run is not mask_lost
    (missed,) = _errors("tube", "split", "raw", "missed")
    assert missed["cause"] == "unexplained"
    assert missed["in_main_component"] == 1


def test_a_break_end_finds_its_cause_along_the_edge():
    # ends retracted 9 voxels from the gap's faces: beyond 2 r of the gap, but
    # within the break's own length along the tube
    case = build_case("tube-gap")
    start, end, _ = _TUBE
    half = (2.0 + 9.0) / (end[0] - start[0])
    arrays = _from_edges(_tube_graph(), {0: [(0.0, 0.5 - half), (0.5 + half, 1.0)]})
    scores = _score(case.sample, arrays)
    ends = [e for e in scores.errors if e["type"] == "break_end"]
    assert len(ends) == 4  # both stages: clean-up cannot bridge 22 voxels
    assert all(e["break_length"] == pytest.approx(22.0, abs=0.5) for e in ends)
    assert {e["cause"] for e in ends} == {"gap#0"}


@pytest.mark.parametrize("case", ["y-thin", "y-thin-gap"])
def test_the_thin_child(case):
    (thin,) = [r for r in _events(case, "true", "junction") if r["edge"] == 2]
    assert thin["fit_class"] == "reaches" and thin["class"] == "connected"
    (thin,) = [r for r in _events(case, "retract", "junction") if r["edge"] == 2]
    assert thin["fit_class"] == "short"
    assert thin["class"] == "detached" and thin["anchors_matched"]
    assert 1.3 < thin["q_join"] < 1.9  # above join: assembly cannot link it
    cause = "junction_gap#0" if case == "y-thin-gap" else "near_junction"
    assert [e["cause"] for e in _errors(case, "retract", "raw", "break_end")] == [cause]
    # clean-up bridges it, off the centreline: no segment deviates there
    sources = {e["source"] for e in _errors(case, "retract", "clean", "off_centre")}
    assert sources == {"node"}


def test_detached_needs_both_anchors():
    case = build_case("y-thin")
    true = TrueCentreline(case.sample.graph)
    (event,) = [e for e in extract_events(case.sample) if e.event_id == "junction#1-2"]
    parent_anchor = true.points[event.anchors[1]]
    # drop the parent's segments whose axis passes the parent's anchor
    arrays = case.predictions["retract"]
    vector = arrays.end - arrays.start
    t = np.clip(
        ((parent_anchor - arrays.start) * vector).sum(1) / (vector**2).sum(1), 0, 1
    )
    closest = arrays.start + t[:, None] * vector
    keep = np.linalg.norm(closest - parent_anchor, axis=1) > 3.0
    assert (~keep).any()
    scores = _score(case.sample, arrays.select(np.flatnonzero(keep)))
    (row,) = [
        r
        for r in scores.events
        if r["event_id"] == "junction#1-2" and r["stage"] == "raw"
    ]
    assert not row["anchors_matched"]
    assert row["class"] == "missing"


def test_a_spur_into_a_bump():
    (bump,) = _events("tube-bump", "true", "bump")
    assert not bump["spur"] and not bump["fit_spur"] and bump["deviation"] < 0.25
    for stage in STAGES:
        (bump,) = _events("tube-bump", "spur", "bump", stage)
        assert bump["spur"] and bump["fit_spur"]
        assert bump["spur_leaf_length"] > 0
    assert bump["q_prune_spur"] == pytest.approx(0.99)
    assert bump["q_protrusion"] == pytest.approx(2.0, abs=0.15)  # 4 outside, r 2
    assert bump["tau_protrusion"] == GraphConfig().clean.min_protrusion
    (protrusion,) = _events("tube-bump", "spur", "protrusion")
    assert protrusion["spur"] and protrusion["fit_spur"]
    assert protrusion["bump_inside"] == ["bump#0"]
    ends = _errors("tube-bump", "spur", "raw", "spur_end")
    assert [e["cause"] for e in ends] == ["bump#0"]


def test_a_false_join_at_a_contact():
    for stage in STAGES:
        errors = _errors("contact", "join", stage)
        assert not [e for e in errors if e["type"] == "break_end"]
        assert [e for e in errors if e["type"] == "junction_extra"]
        assert {e["cause"] for e in errors} == {"contact#0"}
        assert _sample_row("contact", "join", stage)["n_components"] == 1
    # the true graph has two components: the root's holds half of it
    row = _sample_row("contact", "true", "raw")
    assert row["root_connected_fraction"] == pytest.approx(0.5, abs=0.02)
    # the join is not a spur: it leaves no end outside the true tubes
    (protrusion,) = _events("contact", "join", "protrusion")
    assert not protrusion["fit_spur"]


# -- tables ----------------------------------------------------------------------------


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("prediction", ["true", "planted"])
def test_ends_agree_with_compare_graphs(case, prediction):
    if prediction == "planted":
        prediction = next(p for p in build_case(case).predictions if p != "true")
    for row in _scored(case, prediction).samples:
        assert row["n_spur_end"] == row["n_spur_ends"]
        assert row["n_break_end"] == row["n_break_ends"]


def test_one_event_row_per_stage_with_shared_fit_outcomes():
    scores = _scored("y-thin", "retract")
    raw = [r for r in scores.events if r["stage"] == "raw"]
    clean = [r for r in scores.events if r["stage"] == "clean"]
    assert [r["event_id"] for r in raw] == [r["event_id"] for r in clean]
    shared = ("fit_class", "fit_covered", "fit_spur", "q_join", "q_prune_connect")
    for a, b in zip(raw, clean, strict=True):
        for key in shared:
            assert (a.get(key) == b.get(key)) or (np.isnan(a[key]) and np.isnan(b[key]))


def test_error_ids_count_per_stage_and_type():
    errors = _scored("tube", "split").errors
    ids = [e["error_id"] for e in errors]
    assert len(set(ids)) == len(ids)
    assert "raw/break_end#0" in ids and "raw/break_end#1" in ids


def test_scores_without_segments():
    case = build_case("tube")
    sample = case.sample
    raw, clean = graph_stage(
        None, None, sample.mask, sample.spacing, GraphConfig("skeletonize")
    )
    scores = score_sample(sample, raw, clean)
    assert not any("fit_covered" in r or "q_join" in r for r in scores.events)
    assert all(e.get("source") is None for e in scores.errors)
    metrics = compare_graphs(raw, sample.graph)
    assert scores.samples[0]["centreline_f1"] == metrics.centreline_f1


# -- decision quantities ---------------------------------------------------------------


def test_bottleneck_values():
    inf = np.inf
    # a chain 0 - 1 - 2 with links 0.3 and 0.8, and a direct link 0 - 2 of 2.0
    weight = np.array([[inf, 0.3, 2.0], [0.3, inf, 0.8], [2.0, 0.8, inf]])
    minimax = _bottleneck(weight, minimise=True)
    assert minimax[0, 2] == pytest.approx(0.8)
    assert minimax[0, 1] == pytest.approx(0.3)
    assert np.diag(minimax).tolist() == [0, 0, 0]
    # widest path by opacity: 0 - 2 directly (0.5) beats 0 - 1 - 2 (0.2)
    width = np.array([[0, 0.2, 0.5], [0.2, 0, 0.9], [0.5, 0.9, 0]])
    widest = _bottleneck(width, minimise=False)
    assert widest[0, 2] == pytest.approx(0.5)
    assert widest[0, 1] == pytest.approx(0.5)  # 0 - 2 - 1


def test_link_ratios_of_endpoints_and_t_junctions():
    segments = SegmentArrays(
        start=np.array([[0.0, 0, 0], [12.0, 0, 0], [6.0, 3.0, 0]]),
        end=np.array([[10.0, 0, 0], [20.0, 0, 0], [6.0, 10.0, 0]]),
        radius=np.full((3, 2), 2.0),
        opacity=np.array([0.99, 0.99, 0.05]),
    )
    links = segment_links(segments, prune_threshold=0.1, join=1.0, min_radius=0.75)
    assert links.ratio[0, 1] == pytest.approx(1.0)  # ends 2 apart, r 2
    assert links.ratio[0, 2] == pytest.approx(1.5)  # an end 3 from 0's interior
    assert links.alive.tolist() == [True, True, False]
    assert np.isinf(links.minimax[0, 2])  # 2 is pruned
    assert links.widest[0, 2] == 0  # their link is above join
