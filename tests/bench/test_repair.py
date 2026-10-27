import dataclasses

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("pandas")

from spleleton.bench.cases import build_cases
from spleleton.bench.probe import probe_spec
from spleleton.bench.repair import (
    classify,
    find_repairs,
    fitted,
    repair_rows,
)
from spleleton.bench.variants import get_variant
from spleleton.pipeline import graph_stage
from spleleton.primitives import SegmentArrays

PHASE5 = get_variant("phase5")
SPEC = probe_spec(PHASE5, slim=True)
CONFIG = PHASE5.variant.graph


@pytest.fixture(scope="module")
def cases():
    return {case.name: case for case in build_cases()}


def _raw(arrays: SegmentArrays) -> dict[str, np.ndarray]:
    return {
        "start": arrays.start,
        "end": arrays.end,
        "log_radius": np.log(arrays.radius),
        "opacity_logit": np.log(arrays.opacity / (1 - arrays.opacity)),
    }


def _fitted(case, arrays, config=CONFIG):
    sample = case.sample
    raw, clean = graph_stage(arrays, None, sample.mask, sample.spacing, CONFIG)
    return fitted(sample, _raw(arrays), {"raw": raw, "clean": clean}, config)


def _prediction(cases, name, prediction):
    case = cases[name]
    if prediction != "spur-empty":
        return case, case.predictions[prediction]
    # tube-bump's spur, on the tube with no bump in the mask
    bump = cases["tube-bump"].predictions["spur"]
    spur = SegmentArrays(
        bump.start[-1:], bump.end[-1:], radius=bump.radius[-1:],
        opacity=bump.opacity[-1:],
    )  # fmt: skip
    return case, case.predictions["true"].concatenate(spur)


@pytest.mark.parametrize(
    ("name", "prediction", "kinds"),
    [
        ("tube", "spur-empty", {"remove"}),
        ("tube", "split", {"join"}),
        ("tube", "shift", {"recentre"}),
        ("y-thin", "retract", {"join", "reattach", "attach"}),
    ],
)
def test_the_known_answers_are_search_errors(cases, name, prediction, kinds):
    case, arrays = _prediction(cases, name, prediction)
    rows = repair_rows(_fitted(case, arrays), SPEC, seed=name, check=True)
    ran = [r for r in rows if "margin" in r]
    assert {r["repair"] for r in ran} == kinds
    for row in ran:
        assert row["class"] == "search", row
        assert row["valid"] and not row["escaped"]
        # the crop's margins are the full volume's
        assert row["crop_error"] <= 1e-11
        assert np.isfinite(row["margin"]) and row["margin_voxels"] < 0


def test_every_error_has_a_class_or_a_reason(cases):
    case, arrays = _prediction(cases, "y-thin", "retract")
    fit = _fitted(case, arrays)
    rows = repair_rows(fit, SPEC, cap=1, seed="cap")
    assert {r["error_id"] for r in rows} >= {e["error_id"] for e in fit.scores.errors}
    for row in rows:
        assert row["error_class"] is not None or row["reason"], row
    statuses = {r["status"] for r in rows}
    assert "over the cap" in statuses and "no repair for this type" in statuses
    assert len({r["repair_id"] for r in rows if "margin" in r}) == 1


def test_an_error_with_q_below_tau_is_classed_assembly(cases):
    # with the join radius raised, the split tube's two pieces link in the
    # fitted segments, so the graph's break is assembly's
    case, arrays = _prediction(cases, "tube", "split")
    loose = dataclasses.replace(CONFIG, join=10.0)
    fit = _fitted(case, arrays, config=loose)
    repairs = [r for r in find_repairs(fit) if r.error_type == "break_end"]
    assert repairs and all(r.status == "assembly" and r.q < r.tau for r in repairs)
    rows = repair_rows(fit, SPEC, seed="assembly")
    breaks = [r for r in rows if r["error_type"] == "break_end"]
    assert breaks and all(r["error_class"] == "assembly" for r in breaks)
    assert not any("margin" in r for r in breaks)
    # with the default join they do not link, and the join runs
    default = [r for r in find_repairs(_fitted(case, arrays)) if r.kind == "join"]
    assert default and all(r.status == "ok" and r.q >= r.tau for r in default)


def test_classify_reads_the_margin_against_the_floor():
    assert classify(-1.0, 0.1, 10) == ("search", 0.1)
    assert classify(1.0, 0.1, 10) == ("loss", 0.1)
    assert classify(0.05, 0.01, 10) == ("undetermined", 0.1)


def test_the_repair_classes_plot(cases, tmp_path):
    import pandas as pd

    from spleleton.bench.plots import plot_repair_classes

    case, arrays = _prediction(cases, "tube", "split")
    rows = repair_rows(_fitted(case, arrays), SPEC, seed="plot")
    frame = pd.DataFrame(rows).assign(subset="tube", seed=0)
    path = plot_repair_classes(frame, tmp_path / "classes.png")
    assert path.stat().st_size > 0
