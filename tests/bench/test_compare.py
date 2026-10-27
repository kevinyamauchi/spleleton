"""Statistics, comparisons, the report and the Phase F bundle (step 0.8)."""

import json

import numpy as np
import pytest

pd = pytest.importorskip("pandas")
pytest.importorskip("matplotlib")

from spleleton.bench.collect import budget, collect  # noqa: E402
from spleleton.bench.compare import (  # noqa: E402
    compare,
    compare_runs,
    event_flips,
    match_errors,
    read_tables,
)
from spleleton.bench.runner import run_variant  # noqa: E402
from spleleton.bench.stages import record_path  # noqa: E402
from spleleton.bench.stats import (  # noqa: E402
    POOLED,
    cluster_bootstrap,
    flagged,
    minimum_floor,
    noise_floor,
    null_flip_rates,
    paired_difference,
)
from spleleton.bench.store import Store, write_json  # noqa: E402
from spleleton.bench.suites import build, proto_cases  # noqa: E402
from spleleton.bench.variants import get_variant, sweep  # noqa: E402

SUBSETS = ["tube-bump", "y-thin", "contact"]
SCOPE = {"suite": "proto-cases", "split": "cases", "subsets": SUBSETS}


# -- statistics ------------------------------------------------------------------------


def test_paired_difference():
    out = paired_difference([0.1, 0.2, -0.05, 0.15], higher_is_better=True)
    assert out["n"] == 4 and out["mean"] == pytest.approx(0.1)
    assert (out["wins"], out["ties"], out["losses"]) == (3, 0, 1)
    assert out["low"] < out["mean"] < out["high"]
    # lower is better: the same differences are losses
    assert paired_difference([0.1, 0.2], higher_is_better=False)["losses"] == 2
    # one sample has no interval, so nothing is flagged on it
    one = paired_difference([0.5])
    assert np.isnan(one["low"]) and not flagged(one["mean"], one["low"], one["high"], 0)
    zero = paired_difference([0.0, 0.0, 0.0])
    assert zero["mean"] == 0 and zero["wilcoxon_p"] == 1.0
    assert not flagged(zero["mean"], zero["low"], zero["high"], 0)


def test_cluster_bootstrap_needs_two_clusters_when_asked():
    mean, low, high = cluster_bootstrap([1.0, 0.0, 1.0], ["a", "a", "a"])
    assert mean == pytest.approx(2 / 3) and low == high == pytest.approx(2 / 3)
    _, low, high = cluster_bootstrap([1.0, 0.0], ["a", "a"], min_clusters=2)
    assert np.isnan(low) and np.isnan(high)
    _, low, high = cluster_bootstrap([1.0, 0.0, 1.0, 1.0], ["a", "a", "b", "b"])
    assert 0.5 <= low <= high <= 1.0


def _replicate(seed, offsets):
    rows = [
        {"subset": s, "seed": k, "stage": "raw", "centreline_f1": 0.9 + o}
        for (s, k), o in zip([("a", 0), ("a", 1), ("b", 0)], offsets, strict=True)
    ]
    return pd.DataFrame(rows)


def test_noise_floor_from_replicates():
    a = _replicate(0, [0.0, 0.0, 0.0])
    b = _replicate(1, [0.01, -0.01, 0.01])
    table = noise_floor([a, b], ["centreline_f1"])
    pooled = table[table["subset"] == POOLED].iloc[0]
    # one pair: sigma_w = sqrt(mean(d^2) / 2), floor = sqrt(2) sigma_w = |d|
    assert pooled["sigma_within"] == pytest.approx(0.01 / np.sqrt(2))
    assert pooled["floor"] == pytest.approx(0.01)
    assert pooled["n"] == 3
    assert set(table["subset"]) == {"a", "b", POOLED}
    # identical replicates: a floor of 0
    assert noise_floor([a, a], ["centreline_f1"])["floor"].max() == 0


def test_null_flip_rates():
    def events(values):
        return pd.DataFrame(
            {
                "subset": "a", "seed": 0, "stage": "raw", "event_type": "junction",
                "event_id": [f"junction#{i}" for i in range(len(values))],
                "connected": values,
            }
        )  # fmt: skip

    table = null_flip_rates(
        [events([True, True, False, True]), events([True, False, False, True])],
        {"junction": "connected"},
    )
    row = table[table["subset"] == POOLED].iloc[0]
    assert (row["events"], row["flips"]) == (4, 1)
    assert row["flip_rate"] == pytest.approx(0.25)


def test_minimum_floor_is_one_event_per_volume():
    assert minimum_floor("count") == 1
    assert minimum_floor("rate", [4, 2, 0]) == pytest.approx((1 / 4 + 1 / 2) / 2)
    assert minimum_floor("continuous") == 0
    assert flagged(0.5, 0.2, 0.8, floor=1.0) is False  # inside the floor
    assert flagged(1.5, 1.2, 1.8, floor=1.0) is True


# -- runs ------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    store = Store(tmp_path_factory.mktemp("bench") / "bench_quick", quick=True)
    build(store, proto_cases(), SUBSETS, ["cases"])
    return store


@pytest.fixture(scope="module")
def runs(store):
    names = ["skeletonize+clean", "skeletonize"]
    out = {
        name: run_variant(store, get_variant(name), progress=None, **SCOPE)
        for name in names
    }
    for named in sweep("skeletonize+clean", "graph.clean.min_protrusion", [1.0, 3.0]):
        out[named.name] = run_variant(store, named, progress=None, **SCOPE)
    return out


def test_a_run_compared_with_itself_differs_in_nothing(store, runs):
    base = runs["skeletonize+clean"]
    result = compare(store, [base, base], base)
    table = result.comparisons
    assert len(table) and (table["mean"] == 0).all()
    assert not table["flagged"].any()
    assert (result.flips["flips"] == 0).all() and result.flipped.empty
    assert set(result.errors["status"]) == {"shared"}
    assert (result.error_flips[["fixed", "new"]] == 0).all().all()


def test_events_join_across_runs_by_id(store, runs):
    a = read_tables(runs["skeletonize+clean"])
    b = read_tables(runs["skeletonize"])
    keys = ["subset", "seed", "stage", "event_type", "event_id"]
    ids_a = a.events[keys].drop_duplicates()
    ids_b = b.events[keys].drop_duplicates()
    assert len(ids_a) == len(a.events)  # unique
    assert (
        ids_a.sort_values(keys)
        .reset_index(drop=True)
        .equals(ids_b.sort_values(keys).reset_index(drop=True))
    )
    flips, flipped = event_flips(a, b)
    pooled = flips[flips["subset"] == POOLED]
    assert set(pooled["event_type"]) >= {"junction", "window"}
    assert (pooled["fixed"] + pooled["new"] == pooled["flips"]).all()
    assert len(flipped) == pooled["flips"].sum()


def test_errors_match_by_type_and_position():
    def errors(positions, kind="spur_end"):
        return pd.DataFrame(
            {
                "subset": "a", "seed": 0, "stage": "raw", "type": kind,
                "error_id": [f"raw/{kind}#{i}" for i in range(len(positions))],
                "position": [list(p) for p in positions], "cause": "unexplained",
            }
        )  # fmt: skip

    matched = match_errors(
        errors([(0, 0, 0), (10, 0, 0)]), errors([(1, 0, 0), (30, 0, 0)])
    )
    assert sorted(matched["status"]) == ["fixed", "new", "shared"]
    shared = matched[matched["status"] == "shared"].iloc[0]
    assert (
        shared["error_id"] == "raw/spur_end#0" and shared["match"] == "raw/spur_end#0"
    )


def test_the_report_has_every_plot(store, runs, tmp_path):
    order = list(runs.values())
    report = compare_runs(store, order, tmp_path, baseline=order[0])
    text = report.read_text()
    figures = {p.name for p in (tmp_path / "figures").glob("*.png")}
    expected = {
        "scorecard_clean.png", "scorecard_raw.png", "forest_clean.png",
        "forest_raw.png", "error_modes_raw.png", "error_modes_clean.png",
        "paired_scatter.png", "tradeoff_raw.png", "tradeoff_clean.png",
        "flips_raw.png", "flips_clean.png",
        "sweep_skeletonize_clean_graph_clean_min_protrusion.png",
    }  # fmt: skip
    assert expected <= figures
    for figure in expected:
        assert f"figures/{figure}" in text
    for section in ("## Scorecard", "## Paired comparisons", "## Flips", "## Errors",
                    "## Noise floor", "## Error-mode curves"):  # fmt: skip
        assert section in text
    for table in ("samples", "events", "errors", "comparisons", "flips"):
        assert (tmp_path / "tables" / f"{table}.parquet").exists()


def test_collect_writes_the_bundle(store, runs, tmp_path):
    # a stand-in for the baselines stage's record, with these runs
    write_json(
        record_path(store, "baselines"),
        {"name": "baselines", "status": "done", "src_hash": "x", "commit": "y",
         "elapsed_s": 60.0, "checks": {"no sample failed": True},
         "runs": [str(p) for p in runs.values()], "probes": []},
    )  # fmt: skip
    report = collect(store, tmp_path)
    text = report.read_text()
    assert (tmp_path / "baselines.md").exists()
    for section in ("## Stages", "## Baselines", "## Screening", "## Budget (F13)"):
        assert section in text
    digest = json.loads((tmp_path / "digest.json").read_text())
    assert {f"F{i}" for i in range(1, 14)} <= set(digest)
    assert digest["F1"]["pooled"]  # the baselines' comparison
    assert digest["F5"] == {"pending": "no replicates of the baseline (stage S4)"}
    assert "pending" in digest["F6"] and "pending" in digest["F8"]
    assert list((tmp_path / "figures").glob("*.png"))
    record_path(store, "baselines").unlink()


def test_budget_projects_every_stage(store, runs):
    result = budget(store)
    stages = {row["label"]: row for row in result["stages"]}
    assert set(stages) == {"S1", "S2", "S3", "S4", "S5", "S6", "S7"}
    for row in stages.values():
        assert 0 < row["low_s"] <= row["high_s"]
    # proto-cases volumes are 64^3: they are not timed, so stage 0's units hold
    units = {u["unit"]: u for u in result["units"]}
    assert units["skeletonize"]["n"] == 0
    assert units["skeletonize"]["source"].startswith("stage 0")


def test_loss_curves_come_from_the_stored_fits(store, tmp_path):
    pytest.importorskip("jax")
    short = ["fit.n_rounds=2", "fit.fit.n_iterations=40"]
    run = run_variant(
        store, get_variant("phase5", short), progress=None,
        suite="proto-cases", split="cases", subsets=["tube-bump"],
    )  # fmt: skip
    report = compare_runs(store, [run], tmp_path)
    assert (tmp_path / "figures" / "loss_curves.png").exists()
    assert "figures/loss_curves.png" in report.read_text()
