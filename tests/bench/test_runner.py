import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("jax")
pd = pytest.importorskip("pandas")

from spleleton.bench import METRICS_VERSION  # noqa: E402
from spleleton.bench.runner import (  # noqa: E402
    REPLICATE_JITTER,
    Limits,
    check_jitter,
    keys_for,
    list_runs,
    load_fit,
    read_run,
    resolve_run,
    run_variant,
    score_run,
    verify_run,
)
from spleleton.bench.store import Store  # noqa: E402
from spleleton.bench.suites import build, load_split, proto_cases  # noqa: E402
from spleleton.bench.variants import get_variant  # noqa: E402
from spleleton.pipeline._pipeline_jax import jitter_segments  # noqa: E402
from spleleton.primitives import SegmentArrays  # noqa: E402

#: A short Phase 5 fit, for speed.
SHORT = ["fit.n_rounds=2", "fit.fit.n_iterations=40"]
SCOPE = {"suite": "proto-cases", "split": "cases", "subsets": ["tube-bump", "y-thin"]}


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    store = Store(tmp_path_factory.mktemp("bench") / "bench_quick", quick=True)
    build(store, proto_cases(), ["tube-bump", "y-thin"], ["cases"])
    return store


@pytest.fixture(scope="module")
def phase5_run(store):
    return run_variant(store, get_variant("phase5", SHORT), progress=None, **SCOPE)


def _fits(store):
    return sorted((store.root / "cache" / "fit").glob("*.npz"))


def test_a_run_records_every_sample(store, phase5_run):
    meta = read_run(phase5_run)
    assert set(meta["samples"]) == {"tube-bump_0000", "y-thin_0000"}
    for entry in meta["samples"].values():
        assert entry["status"] == "done"
        assert entry["fit_cached"] is False
        # timing, load average and power source, per sample
        assert entry["fit"]["timing"]["wall_s"] > 0
        assert "load_1m" in entry["host_end"] and "power" in entry["host_end"]
        rounds = entry["fit"]["rounds"]
        assert len(rounds) == 2
        assert {"loss_change", "grad_norm", "steady_ms"} <= set(rounds[0])
    assert meta["provenance"]["src_hash"]
    rows = pd.read_parquet(phase5_run / "samples.parquet")
    assert set(rows["stage"]) == {"raw", "clean"} and len(rows) == 4


def test_a_run_writes_the_three_score_tables(phase5_run):
    tables = {
        t: pd.read_parquet(phase5_run / f"{t}.parquet")
        for t in ("samples", "events", "errors")
    }
    for frame in tables.values():
        assert (frame["metrics_version"] == METRICS_VERSION).all()
    samples, events = tables["samples"], tables["events"]
    assert {"root_connected_fraction", "n_alive", "fit_s", "n_idle_rounds"} <= set(
        samples.columns
    )
    assert (samples["n_rounds"] == 2).all()
    # one row per (event, stage), joinable by ID
    assert set(events["stage"]) == {"raw", "clean"}
    per_stage = events.groupby("stage")["event_id"].apply(sorted)
    assert per_stage["raw"] == per_stage["clean"]
    assert {"bump#0", "junction#1-2"} <= set(events["event_id"])


def test_scoring_again_reproduces_the_tables(store, phase5_run):
    before = {
        t: pd.read_parquet(phase5_run / f"{t}.parquet")
        for t in ("samples", "events", "errors")
    }
    score_run(store, phase5_run, progress=None)
    for table, frame in before.items():
        again = pd.read_parquet(phase5_run / f"{table}.parquet")
        pd.testing.assert_frame_equal(frame, again, check_exact=True)


def test_a_second_run_is_a_no_op(store, phase5_run):
    before = read_run(phase5_run)["samples"]
    fits = _fits(store)
    lines = []
    again = run_variant(
        store, get_variant("phase5", SHORT), progress=lines.append, **SCOPE
    )
    assert again == phase5_run
    assert read_run(again)["samples"] == before
    assert _fits(store) == fits
    assert lines == [f"{again.name.split('__')[0]}: 0 samples to run (2 done)"]


def test_an_assembly_only_variant_reuses_the_fits(store, phase5_run):
    fits = _fits(store)
    other = run_variant(
        store, get_variant("phase5", [*SHORT, "graph.join=1.5"]), progress=None, **SCOPE
    )
    assert other != phase5_run
    samples = read_run(other)["samples"]
    assert all(s["fit_cached"] for s in samples.values())
    assert _fits(store) == fits
    first = read_run(phase5_run)["samples"]
    assert all(samples[k]["fit_key"] == first[k]["fit_key"] for k in samples)
    assert all(samples[k]["graph_key"] != first[k]["graph_key"] for k in samples)


def test_verify_confirms_determinism(store, phase5_run):
    result = verify_run(store, phase5_run, n=1, progress=None)
    assert result["ok"]
    assert resolve_run(store, f"{read_run(phase5_run)['name']}@latest") == phase5_run
    assert any(row["directory"] == str(phase5_run) for row in list_runs(store))


def test_a_stored_fit_reproduces_its_loss_in_a_fresh_process(store, phase5_run):
    meta = read_run(phase5_run)
    entry = meta["samples"]["y-thin_0000"]
    code = f"""
import json
from pathlib import Path
from spleleton.bench.runner import check_fit
from spleleton.bench.store import Store
from spleleton.bench.suites import load_split
from spleleton.bench.variants import Variant
store = Store(Path({str(store.root)!r}), quick=True)
(loaded,) = load_split(store, "proto-cases", "y-thin", "cases", [0])
variant = Variant.from_dict(json.loads({json.dumps(json.dumps(meta["spec"]))}))
print(json.dumps(check_fit(store.fit_path({entry["fit_key"]!r}), loaded, variant)))
"""
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    result = json.loads(out.stdout.strip().splitlines()[-1])
    assert result["final"]["bitwise"], result
    assert result["last_iteration"]["bitwise"], result


@pytest.fixture(scope="module")
def replicate_run(store, phase5_run):
    return run_variant(
        store, get_variant("phase5", SHORT), replicate=1, progress=None, **SCOPE
    )


def test_replicates_get_their_own_fits(store, phase5_run, replicate_run):
    variant = get_variant("phase5", SHORT).variant
    first = keys_for(variant, "sha", 0)[0]
    jittered = replicate_run
    meta = read_run(jittered)
    assert meta["name"].endswith("~r1")
    assert meta["spec"]["init_jitter"] == pytest.approx(1e-3)
    base = read_run(phase5_run)["samples"]
    for label, entry in meta["samples"].items():
        assert entry["fit_key"] != base[label]["fit_key"]
        assert entry["fit_cached"] is False
    assert keys_for(variant, "sha", 1)[0] != first


def test_the_jitter_moves_every_endpoint_a_little():
    arrays = SegmentArrays(np.zeros((50, 3)), np.ones((50, 3)))
    moved = jitter_segments(arrays, 1e-3, seed=4)
    shifts = np.linalg.norm(moved.start - arrays.start, axis=1)
    assert (shifts >= 0.5e-3 - 1e-12).all() and (shifts <= 1e-3 + 1e-12).all()
    again = jitter_segments(arrays, 1e-3, seed=4)
    assert np.array_equal(moved.start, again.start)
    assert jitter_segments(arrays, 0.0, seed=4) is arrays


def test_a_jittered_start_moves_a_little_and_rebuilds_from_its_seed(
    store, phase5_run, replicate_run
):
    result = check_jitter(store, replicate_run)
    assert result["jitter"] == pytest.approx(REPLICATE_JITTER)
    assert set(result["samples"]) == {"tube-bump_0000", "y-thin_0000"}
    for row in result["samples"].values():
        assert row["n_endpoints"] > 0
        assert 0.5 * REPLICATE_JITTER - 1e-12 <= row["min_shift"]
        assert row["max_shift"] <= REPLICATE_JITTER + 1e-12
        assert row["rebuilds"]
    # replicate 0 starts where the fit stage starts, unmoved
    for row in check_jitter(store, phase5_run)["samples"].values():
        assert row["max_shift"] == 0.0


def test_replicate_runs_pair_only_the_same_variant(store, phase5_run, replicate_run):
    from spleleton.bench.compare import read_tables, replicate_runs

    assert replicate_runs(store, read_tables(phase5_run)) == [phase5_run, replicate_run]
    # a replicate's own list starts with it and holds no other replicate 0
    assert replicate_runs(store, read_tables(replicate_run)) == [replicate_run]
    # a replicate of another spec under the same name is not paired
    meta_path = replicate_run / "run.json"
    meta = json.loads(meta_path.read_text())
    changed = json.loads(json.dumps(meta))
    changed["spec"]["fit"]["connection_reach"] += 1.0
    other = replicate_run.with_name(replicate_run.name + "-changed")
    other.mkdir()
    (other / "run.json").write_text(json.dumps(changed))
    try:
        assert other not in replicate_runs(store, read_tables(phase5_run))
    finally:
        (other / "run.json").unlink()
        other.rmdir()


def test_the_floor_stage_on_two_replicates(store, phase5_run, replicate_run):
    from spleleton.bench.stages import StageContext, floor_runs

    lines = []
    ctx = StageContext(store, True, lines.append)
    floor_runs(ctx, get_variant("phase5", SHORT), SCOPE, replicates=2)
    assert ctx.runs == [str(phase5_run), str(replicate_run)]
    assert all(ctx.checks.values()), {k: ctx.details.get(k) for k in ctx.checks}
    assert len(ctx.checks) == 7
    report = (store.report_dir("floor") / "report.md").read_text()
    assert "From 2 replicates" in report and "**Null flip rates**" in report


def test_the_diagnose_stage_repairs_and_starts_from_the_truth(store, phase5_run):
    from spleleton.bench.repair import summary
    from spleleton.bench.runner import ground_truth_start
    from spleleton.bench.stages import StageContext, diagnose_runs

    ctx = StageContext(store, True, lambda line: None)
    truth = get_variant("phase5-ground-truth", SHORT)
    diagnose_runs(ctx, get_variant("phase5", SHORT), truth, SCOPE, SCOPE, cap=2)
    assert all(ctx.checks.values()), {k: ctx.details.get(k) for k in ctx.checks}
    assert len(ctx.checks) == 7
    repairs = pd.read_parquet(phase5_run / "repairs.parquet")
    assert summary(repairs)["repairs_run"] >= 1
    # a second call redoes no volume
    parts = sorted((phase5_run / "parts").glob("*.repairs.parquet"))
    stamps = [p.stat().st_mtime_ns for p in parts]
    diagnose_runs(ctx, get_variant("phase5", SHORT), truth, SCOPE, SCOPE, cap=2)
    assert [p.stat().st_mtime_ns for p in parts] == stamps
    # the start is the subdivided true graph
    (loaded,) = load_split(store, "proto-cases", "y-thin", "cases", [0])
    assert len(ground_truth_start(loaded.sample)) > loaded.sample.graph.n_edges


def _implicit(store, run, label, method):
    from spleleton.bench.sensitivity import DEFAULT_PARAMS, implicit_sample
    from spleleton.bench.variants import Variant

    meta = read_run(run)
    entry = meta["samples"][label]
    subset, seed = label.rsplit("_", 1)
    (loaded,) = load_split(store, meta["suite"], subset, meta["split"], [int(seed)])
    record = load_fit(store.fit_path(entry["fit_key"]))
    return implicit_sample(
        Variant.from_dict(meta["spec"]), loaded.sample, record, DEFAULT_PARAMS,
        method=method, check_width=True,
    )  # fmt: skip


def test_implicit_derivatives_at_a_stored_fit(store, phase5_run):
    result = _implicit(store, phase5_run, "y-thin_0000", "explicit")
    info = result["info"]
    # the JAX decision quantities are the numpy ones
    assert info["events"] > 0 and info["q_error"] <= 2e-7
    # the width's mixed term (a jvp of the gradient) is its central difference
    assert info["width_error"] <= 1e-4
    assert info["spectrum"]["largest"] > 0
    rows = pd.DataFrame(result["rows"])
    assert set(rows["parameter"]) >= {
        "fit.connection_weight",
        "fit.fit.render.wall_width",
    }
    assert np.isfinite(rows["dq"]).all() and np.isfinite(rows["damping_spread"]).all()


@pytest.mark.slow
def test_explicit_and_minres_solves_agree(store, phase5_run):
    result = _implicit(store, phase5_run, "y-thin_0000", "both")
    rows = pd.DataFrame(result["rows"])
    explicit, minres = rows["dq/explicit/1e-05"], rows["dq/minres/1e-05"]
    scale = float(explicit.abs().max())
    assert float((explicit - minres).abs().max()) <= 0.02 * scale
    solves = pd.DataFrame(result["solves"])
    assert solves[solves["solver"] == "minres"]["converged"].all()


def test_the_sensitivity_and_validate_stages(store, phase5_run):
    from spleleton.bench.stages import StageContext, sensitivity_runs, validate_runs

    named = get_variant("phase5", SHORT)
    ctx = StageContext(store, True, lambda line: None)
    sensitivity_runs(ctx, named, SCOPE, check_width=True)
    assert all(ctx.checks.values()), {k: ctx.details.get(k) for k in ctx.checks}
    assert len(ctx.checks) == 4
    directory = phase5_run / "sensitivity"
    for name in ("thresholds", "grid", "agreement", "implicit", "solves"):
        assert (directory / f"{name}.parquet").exists()
    assert Path(ctx.details["report"]).exists()
    ctx = StageContext(store, True, lambda line: None)
    validate_runs(ctx, named, SCOPE, (0.2,), second=False)
    assert all(ctx.checks.values()), ctx.details
    assert ctx.details["parameters"] == ["fit.connection_weight"]
    assert len(ctx.runs) == 2
    validation = pd.read_parquet(ctx.details["validation"])
    assert {"observed", "predicted"} <= set(validation.columns)
    assert set(validation["factor"].round(6)) == {0.8, 1.2}


def test_limits_fail_a_sample_with_its_reason(store):
    named = get_variant("phase5", [*SHORT, "fit.n_rounds=1"])
    directory = run_variant(
        store, named, limits=Limits(segments=3), progress=None, **SCOPE
    )
    samples = read_run(directory)["samples"]
    assert all(s["status"] == "failed" for s in samples.values())
    assert all(s["reason"].startswith("segment limit") for s in samples.values())
    # failed samples are not run again, unless retried
    lines = []
    run_variant(store, named, limits=Limits(segments=3), progress=lines.append, **SCOPE)
    assert "0 samples to run" in lines[0]
    run_variant(
        store, named, limits=Limits(segments=3), retry=True, progress=lines.append,
        **SCOPE,
    )  # fmt: skip
    assert "2 samples to run" in lines[1]
    slow = run_variant(
        store, get_variant("phase5", [*SHORT, "fit.fit.n_iterations=41"]),
        limits=Limits(wall_s=0.0), progress=None, **SCOPE,
    )  # fmt: skip
    reasons = [s["reason"] for s in read_run(slow)["samples"].values()]
    assert all(r.startswith("time limit") for r in reasons)


def test_an_interrupted_run_resumes(store):
    named = get_variant("phase5", [*SHORT, "fit.fit.n_iterations=42"])

    def interrupt(line: str) -> None:
        if line.endswith("s") and ": done" in line:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_variant(store, named, progress=interrupt, **SCOPE)
    lines = []
    directory = run_variant(store, named, progress=lines.append, **SCOPE)
    assert "1 samples to run (1 done)" in lines[0]
    assert all(s["status"] == "done" for s in read_run(directory)["samples"].values())


def test_a_stored_fit_round_trips(store, phase5_run):
    entry = read_run(phase5_run)["samples"]["tube-bump_0000"]
    record = load_fit(store.fit_path(entry["fit_key"]))
    assert record.segments.start.dtype == np.float32
    assert len(record.history) == 2 * 40
    assert record.meta["sample_sha256"] == entry["sample_sha256"]
    assert record.meta["fit_spec"]["replicate"] == 0


def test_skeletonize_runs_without_a_fit(store):
    directory = run_variant(
        store, get_variant("skeletonize+clean"), progress=None, **SCOPE
    )
    samples = read_run(directory)["samples"]
    assert all(s["status"] == "done" and "fit" not in s for s in samples.values())


@pytest.mark.slow
def test_workers_give_the_same_graphs(store, phase5_run):
    named = get_variant("phase5", SHORT)
    other = run_variant(
        store, named, fresh=True, workers=2, progress=None,
        suite="proto-cases", split="cases", subsets=["tube-bump", "y-thin"],
    )  # fmt: skip
    assert other == phase5_run
    samples = read_run(other)["samples"]
    assert all(s["fit_cached"] is False for s in samples.values())
    first = {k: v["graph_hashes"] for k, v in samples.items()}
    assert all(first[k] == v["graph_hashes"] for k, v in samples.items())
