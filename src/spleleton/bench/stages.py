"""The stages: one command per long run, with prerequisites and records.

(Plan, Sections 15 and 14.3.) Each stage is a recipe of runs, probes and
checks. ``bench stage NAME`` runs it after checking that its prerequisites
are done, and writes ``stages/<name>.json`` (status, the runs and probes it
made, its checks, timings, and the code and host it ran on) and a log,
``stages/<name>.log``. A stage that is run again resumes: its runs and
probes skip the samples they have finished.

- A **full** stage refuses to start on a dirty tree (uncommitted changes
  under ``src/``, ``scripts/`` or ``pyproject.toml``), so that every stage
  runs from a commit; a **quick** one (``--quick``, in ``bench_quick/``)
  only warns.
- A stage warns at its start when the Mac is on battery or the load is
  above :data:`MAX_LOAD`; every sample records both anyway.

The order is S1, S3, S4, S5, S6, S7, S2 (:data:`ORDER`). The stages whose
building blocks come in later steps of Phase 0 are declared, so that
``status`` and ``--dry-run`` show them, but refuse to run until then.
"""

import json
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spleleton.bench.store import Store, check_tree, src_hash, write_json
from spleleton.bench.timing import host_state

#: A load average above this makes timings unreliable (plan, Section 6.5).
MAX_LOAD = 13.0
#: The full suite.
SUITE = "errors3d-v1"


class StageUnavailable(RuntimeError):
    """The stage's building blocks are not built yet."""


class PrerequisiteMissing(RuntimeError):
    """A stage this one needs is not done."""


@dataclass
class StageContext:
    """What a stage's recipe works with, and what it records."""

    store: Store
    quick: bool
    log: Callable[[str], None]
    runs: list[str] = field(default_factory=list)
    probes: list[str] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    def check(self, name: str, ok: bool, detail: Any = None) -> None:
        """Record a check (and log it)."""
        self.checks[name] = bool(ok)
        if detail is not None:
            self.details[name] = detail
        self.log(f"check {'PASS' if ok else 'FAIL'}: {name}")


@dataclass(frozen=True)
class Stage:
    """A stage: its label, prerequisites, recipe and what it runs.

    Attributes
    ----------
    name : str
        ``data``, ``baselines``, ...
    label : str
        S1-S7.
    needs : tuple of str
        Stages that must be done first.
    describe : str
        What it runs (for ``status`` and ``--dry-run``).
    projected : str
        Its projected time, full and quick.
    commands : callable
        ``commands(quick)``: the equivalent command lines.
    recipe : callable, optional
        ``recipe(context)``; ``None`` while its building blocks are missing.
    built_in : str, optional
        The Phase 0 step that builds it, while ``recipe`` is ``None``.
    """

    name: str
    label: str
    needs: tuple[str, ...]
    describe: str
    projected: str
    commands: Callable[[bool], list[str]]
    recipe: Callable[[StageContext], None] | None = None
    built_in: str | None = None


# -- recipes ---------------------------------------------------------------------------


def _data(ctx: StageContext) -> None:
    from spleleton.bench.suites import QUICK_SCOPE, build, get_suite, load_split

    scope = (
        {
            name: (subsets, (split,), seeds)
            for name, (subsets, split, seeds) in QUICK_SCOPE.items()
        }
        if ctx.quick
        else {SUITE: (None, ("tune", "test"), None)}
    )
    written = checked = 0
    for name, (subsets, splits, seeds) in scope.items():
        report = build(ctx.store, get_suite(name), subsets, splits, seeds)
        written += report.n_written
        checked += report.n_checked
        ctx.log(f"{name}: wrote {report.n_written}, checked {report.n_checked}")
    ctx.details["built"] = {"written": written, "checked": checked}
    loaded = 0
    for name, (subsets, splits, seeds) in scope.items():
        suite = get_suite(name)
        for subset in subsets or suite.subsets:
            for split in splits:
                loaded += len(load_split(ctx.store, suite, subset, split, seeds))
    ctx.check("every sample's hash verifies", loaded == written + checked, loaded)
    again = sum(
        build(ctx.store, get_suite(name), subsets, splits, seeds).n_written
        for name, (subsets, splits, seeds) in scope.items()
    )
    ctx.check("a second build writes nothing", again == 0, again)


def _run_scope(quick: bool) -> dict[str, Any]:
    """The suite, split, subsets and seeds a run stage covers."""
    from spleleton.bench.suites import QUICK_SCOPE

    if quick:
        subsets, split, seeds = QUICK_SCOPE[SUITE]
        return {"suite": SUITE, "split": split, "subsets": subsets, "seeds": seeds}
    return {"suite": SUITE, "split": "tune", "subsets": None, "seeds": None}


def _baselines(ctx: StageContext) -> None:
    from spleleton.bench.runner import read_run, run_variant, verify_run
    from spleleton.bench.variants import BASELINES, get_variant

    scope = _run_scope(ctx.quick)
    directories = {}
    for name in BASELINES:
        directory = run_variant(ctx.store, get_variant(name), progress=ctx.log, **scope)
        directories[name] = directory
        ctx.runs.append(str(directory))
    failed = {
        name: {
            label: s.get("reason")
            for label, s in read_run(d)["samples"].items()
            if s["status"] == "failed"
        }
        for name, d in directories.items()
    }
    failed = {name: f for name, f in failed.items() if f}
    ctx.check("no sample failed", not failed, failed)
    result = verify_run(
        ctx.store, directories["phase5"], n=1 if ctx.quick else 3, progress=ctx.log
    )
    ctx.check("verify matches (phase5)", result["ok"], result["samples"])


def _screen(ctx: StageContext) -> None:
    from spleleton.bench.probe import (
        load_probe,
        loss_correct_rates,
        probe_spec,
        run_probe,
    )
    from spleleton.bench.suites import QUICK_SCOPE, get_suite
    from spleleton.bench.variants import SCREENED, get_variant

    names = ["phase5", SCREENED[0]] if ctx.quick else ["phase5", *SCREENED]
    if ctx.quick:
        plan = [
            ("proto-cases", list(get_suite("proto-cases").subsets), "cases", None),
            (
                SUITE,
                ["asym-clean", "bumps-h1.5"],
                QUICK_SCOPE[SUITE][1],
                QUICK_SCOPE[SUITE][2],
            ),
        ]
        max_events = 3
    else:
        plan = [(SUITE, list(get_suite(SUITE).subsets), "smoke", None)]
        max_events = None
    events = []
    for name in names:
        named = get_variant(name)
        spec = probe_spec(named, slim=True)
        for suite, subsets, split, seeds in plan:
            directory = run_probe(
                ctx.store, spec, f"{name}-slim", suite, subsets, split, seeds,
                floor=name == "phase5", max_events=max_events, progress=ctx.log,
            )  # fmt: skip
            ctx.probes.append(str(directory))
            frame = load_probe(directory).events.assign(probe=name, suite=suite)
            events.append(frame)
            rates = loss_correct_rates(frame)
            ctx.log(rates.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    import pandas as pd

    events = pd.concat(events, ignore_index=True)
    # bounded compiling: a batch whose shapes its probe has already seen
    # compiles nothing (a new shape compiles its three programs once)
    repeated = []
    for _, frame in events.groupby("probe", sort=False):
        seen = set()
        for row in frame.sort_values("started").itertuples():
            if row.shape in seen and row.time_programs > 0:
                repeated.append((row.probe, row.event_id, row.family, row.shape))
            seen.add(row.shape)
    shapes = events.groupby("probe")["shape"].nunique().to_dict()
    ctx.check(
        "compiling is bounded (a batch of shapes already seen compiles nothing)",
        not repeated,
        {
            "recompiled": repeated,
            "distinct_shapes": shapes,
            "compile_s": float(events["time_compile_s"].sum()),
        },
    )
    escaped = events[events["escaped"].astype(str) != ""]
    ctx.check(
        "no relaxed segment leaves its crop",
        escaped.empty,
        escaped[["probe", "event_id", "family", "escaped"]].to_dict("records"),
    )
    missing = events["class_profile"].isna().sum()
    undetermined = int((events["class_profile"] == "undetermined").sum())
    ctx.check("every event gets a class", missing == 0, {"undetermined": undetermined})


#: Replicates of the floor stage, the baselines' run included (Section 8.3).
REPLICATES = {"full": 3, "quick": 2}


def _floor(ctx: StageContext) -> None:
    from spleleton.bench.variants import get_variant

    replicates = REPLICATES["quick" if ctx.quick else "full"]
    floor_runs(ctx, get_variant("phase5"), _run_scope(ctx.quick), replicates)


def floor_runs(ctx: StageContext, named, scope: dict[str, Any], replicates: int):
    """The floor stage's work: replicates of a variant, their checks, a report.

    Replicate 0 is the variant's own run (the baselines stage's, which is
    then only read); replicates 1 to ``replicates - 1`` start jittered.
    The report (``reports/floor/report.md``) compares replicate 0 with
    itself, so that it shows the floor and the null flip rates.
    """
    from spleleton.bench.compare import (
        FLIP_OUTCOMES,
        METRICS,
        compare_runs,
        measure_floors,
        read_tables,
        replicate_runs,
    )
    from spleleton.bench.runner import (
        REPLICATE_JITTER,
        check_jitter,
        read_run,
        run_variant,
    )
    from spleleton.bench.stats import POOLED

    directories = []
    for replicate in range(replicates):
        directory = run_variant(
            ctx.store, named, replicate=replicate, progress=ctx.log, **scope
        )
        directories.append(directory)
        ctx.runs.append(str(directory))
    failed = {
        str(d.name): {
            label: s.get("reason")
            for label, s in read_run(d)["samples"].items()
            if s["status"] == "failed"
        }
        for d in directories
    }
    failed = {name: f for name, f in failed.items() if f}
    ctx.check("no sample failed", not failed, failed)
    # the jittered starts: at most the jitter, never nothing, and the same
    # start again from the same seed
    shifts = {d.name: check_jitter(ctx.store, d)["samples"] for d in directories[1:]}
    rows = [row for samples in shifts.values() for row in samples.values()]
    tolerance = 1e-9
    ctx.check(
        f"each jittered start moves by at most {REPLICATE_JITTER:g} voxel, not by 0",
        bool(rows)
        and all(
            0 < r["min_shift"] and r["max_shift"] <= REPLICATE_JITTER + tolerance
            for r in rows
        ),
        shifts,
    )
    ctx.check(
        "each jittered start rebuilds identically from its seed",
        bool(rows) and all(r["rebuilds"] for r in rows),
    )
    base = read_tables(directories[0])
    found = replicate_runs(ctx.store, base)
    ctx.check(
        "the comparisons find every replicate",
        {Path(d) for d in found} == set(directories),
        [str(d) for d in found],
    )
    floors = measure_floors([read_tables(d) for d in found])
    metrics = floors.metrics
    pooled = metrics[metrics["subset"] == POOLED] if len(metrics) else metrics
    wanted = {m for m in METRICS if m in base.samples}
    missing = {
        stage: sorted(
            wanted
            - (
                set(pooled["metric"][pooled["stage"] == stage])
                if len(pooled)
                else set()
            )
        )
        for stage in ("raw", "clean")
    }
    zero = sorted(set(pooled["metric"][pooled["floor"] == 0])) if len(pooled) else []
    ctx.check(
        "every metric has a floor",
        not any(missing.values()),
        {"missing": missing, "measured_zero": zero},
    )
    kinds = set(base.events["event_type"]) & set(FLIP_OUTCOMES)
    flips = floors.flips
    ctx.check(
        "the null flip rates are computed",
        bool(kinds) and kinds <= (set(flips["event_type"]) if len(flips) else set()),
        flips[flips["subset"] == POOLED].to_dict("records") if len(flips) else [],
    )
    output = ctx.store.report_dir("floor")
    report = compare_runs(ctx.store, [directories[0]], output)
    text = Path(report).read_text()
    ctx.details["report"] = str(report)
    ctx.check(
        "the report shows the floor and the null flip rates",
        f"From {replicates} replicates" in text and "**Null flip rates**" in text,
    )


#: The ground-truth start's subsets in the quick diagnose stage (Section 14.3).
QUICK_GROUND_TRUTH = ["all"]


def _diagnose(ctx: StageContext) -> None:
    from spleleton.bench.repair import MAX_REPAIRS
    from spleleton.bench.variants import get_variant

    scope = _run_scope(ctx.quick)
    truth_scope = {**scope, "subsets": QUICK_GROUND_TRUTH} if ctx.quick else scope
    diagnose_runs(
        ctx, get_variant("phase5"), get_variant("phase5-ground-truth"), scope,
        truth_scope, MAX_REPAIRS["quick" if ctx.quick else "full"],
    )  # fmt: skip


def diagnose_runs(
    ctx: StageContext,
    named,
    truth,
    scope: dict[str, Any],
    truth_scope: dict[str, Any],
    cap: int,
) -> None:
    """The diagnose stage's work: repairs of a run, and the ground-truth start.

    The variant's run (the baselines stage's, only read) is repaired, at
    most ``cap`` repairs per volume; ``truth`` (the variant started from the
    true graph) runs on ``truth_scope`` and is compared with it sample by
    sample.
    """
    import numpy as np
    import pandas as pd

    from spleleton.bench.compare import read_tables
    from spleleton.bench.repair import repair_run, summary
    from spleleton.bench.runner import (
        ground_truth_start,
        load_fit,
        read_run,
        run_variant,
    )
    from spleleton.bench.suites import load_split

    base = run_variant(ctx.store, named, progress=ctx.log, **scope)
    ctx.runs.append(str(base))
    path = repair_run(ctx.store, base, cap=cap, progress=ctx.log)
    repairs = pd.read_parquet(path)
    result = summary(repairs)
    ctx.details["repairs"] = {"table": str(path), **result}
    errors = read_tables(base).errors
    keys = ["subset", "seed", "error_id"]
    listed = (
        set(map(tuple, repairs[keys].astype(str).to_numpy())) if len(repairs) else set()
    )
    missing = [
        k for k in map(tuple, errors[keys].astype(str).to_numpy()) if k not in listed
    ]
    unclassed = repairs[repairs["error_class"].isna() & repairs["reason"].isna()]
    ctx.check(
        "every error has a class or a reason why it has none",
        not missing and unclassed.empty,
        {"missing": missing[:20], "unclassed": unclassed[keys].to_dict("records")},
    )
    ran = repairs.dropna(subset=["margin"]) if "margin" in repairs else repairs.iloc[:0]
    ran = ran.drop_duplicates(["subset", "seed", "repair_id"])
    ctx.check(
        "every repair run returns a finite margin",
        bool(np.isfinite(ran["margin"].to_numpy(float)).all()),
        ran.groupby("repair").size().to_dict() if len(ran) else {},
    )
    below = repairs.dropna(subset=["q", "tau"]) if "q" in repairs else repairs.iloc[:0]
    below = below[below["q"].astype(float) < below["tau"].astype(float)]
    ctx.check(
        "an error with q < tau is classed assembly",
        bool((below["class"] == "assembly").all()),
        len(below),
    )
    escaped = ran[ran["escaped"].astype(bool)] if len(ran) else ran
    ctx.check(
        "relaxed segments stay inside their crops",
        escaped.empty,
        escaped[["subset", "seed", "repair_id"]].to_dict("records"),
    )
    # the ground-truth start
    started = run_variant(ctx.store, truth, progress=ctx.log, **truth_scope)
    ctx.runs.append(str(started))
    meta, base_meta = read_run(started), read_run(base)
    failed = {
        k: v.get("reason")
        for k, v in meta["samples"].items()
        if v["status"] == "failed"
    }
    ctx.check("no ground-truth sample failed", not failed, failed)
    first = {}
    for label, entry in meta["samples"].items():
        if entry["status"] != "done":
            continue
        subset, seed = label.rsplit("_", 1)
        (loaded,) = load_split(
            ctx.store, meta["suite"], subset, meta["split"], [int(seed)]
        )
        rounds = load_fit(ctx.store.fit_path(entry["fit_key"])).meta["rounds"]
        first[label] = (rounds[0]["n_segments"], len(ground_truth_start(loaded.sample)))
    ctx.check(
        "the ground-truth start begins from the subdivided true graph",
        bool(first) and all(a == b for a, b in first.values()),
        first,
    )
    mine, theirs = read_tables(started), read_tables(base)
    ids = ["subset", "seed", "stage", "event_type", "event_id"]
    a = set(map(tuple, mine.events[ids].astype(str).to_numpy()))
    samples = set(map(tuple, mine.events[["subset", "seed"]].astype(str).to_numpy()))
    b = {
        k
        for k in map(tuple, theirs.events[ids].astype(str).to_numpy())
        if k[:2] in samples
    }
    ctx.check(
        "the ground-truth start is scored with the same events as the normal run",
        bool(a) and a == b,
        {"events": len(a), "only_ground_truth": len(a - b), "only_normal": len(b - a)},
    )
    metrics = ["centreline_f1", "junction_f1", "root_connected_fraction"]
    joined = mine.samples.merge(
        theirs.samples, on=["subset", "seed", "stage"], suffixes=("_truth", "_normal")
    )
    losses = {}
    for label, entry in meta["samples"].items():
        normal = base_meta["samples"].get(label, {})
        if entry["status"] == "done" and normal.get("status") == "done":
            losses[label] = {
                "truth": load_fit(ctx.store.fit_path(entry["fit_key"])).meta[
                    "final_terms"
                ]["loss"],
                "normal": load_fit(ctx.store.fit_path(normal["fit_key"])).meta[
                    "final_terms"
                ]["loss"],
            }
    ctx.details["ground_truth"] = {
        "metrics": joined[
            ["subset", "seed", "stage"]
            + [f"{m}_{w}" for m in metrics for w in ("truth", "normal")]
        ].to_dict("records"),
        "final_loss": losses,
    }


#: The quick sensitivity stage's implicit derivatives, and the quick validate
#: stage's re-fits, run on these volumes (Section 14.3).
QUICK_IMPLICIT = ["all", "bumps-h1.5"]
#: Validation: the scale factors (f) of the re-fits at lambda (1 +- f).
VALIDATE_F = {"full": (0.1, 0.2), "quick": (0.2,)}


def _sensitivity(ctx: StageContext) -> None:
    from spleleton.bench.variants import get_variant

    scope = _run_scope(ctx.quick)
    sensitivity_runs(
        ctx, get_variant("phase5"), scope,
        implicit_subsets=QUICK_IMPLICIT if ctx.quick else None,
    )  # fmt: skip


def sensitivity_runs(
    ctx: StageContext,
    named,
    scope,
    implicit_subsets=None,
    params=None,
    check_width: bool | None = None,
) -> None:
    """The sensitivity stage's work: thresholds, the grid and implicit derivatives.

    On the variant's run (the baselines stage's, only read): critical
    thresholds and the restricted grid on every volume, implicit derivatives
    for :data:`~spleleton.bench.sensitivity.DEFAULT_PARAMS` on
    ``implicit_subsets`` (all by default), and the report.
    """
    import numpy as np
    import pandas as pd

    from spleleton.bench import sensitivity
    from spleleton.bench.runner import run_variant

    base = run_variant(ctx.store, named, progress=ctx.log, **scope)
    ctx.runs.append(str(base))
    paths = sensitivity.thresholds_run(ctx.store, base, progress=ctx.log)
    agreement = pd.read_parquet(paths["agreement"])
    low = agreement[agreement["share"] < sensitivity.GRID_AGREEMENT]
    ctx.check(
        "the step functions agree with the grid where the anchors match",
        len(agreement) > 0 and low.empty,
        agreement.to_dict("records"),
    )
    params = params or sensitivity.DEFAULT_PARAMS
    check_width = ctx.quick if check_width is None else check_width
    paths |= sensitivity.implicit_run(
        ctx.store, base, params, subsets=implicit_subsets, check_width=check_width,
        progress=ctx.log,
    )  # fmt: skip
    implicit = pd.read_parquet(paths["implicit"])
    info = pd.read_parquet(paths["implicit_info"])
    ctx.check(
        "every solve records its damping spread",
        len(implicit) > 0 and bool(np.isfinite(implicit["damping_spread"]).all()),
        {"rows": len(implicit),
         "median_spread": float(implicit["damping_spread"].abs().median())
         if len(implicit) else None},
    )  # fmt: skip
    ctx.check(
        "the JAX q equals the numpy q to 2e-7",
        bool((info["q_error"] <= 2e-7).all()),
        info[["subset", "seed", "q_error"]].to_dict("records"),
    )
    if check_width and "width_error" in info:
        ctx.check(
            "the width's mixed term matches central differences of the gradient",
            bool((info["width_error"] <= 1e-4).all()),
            info[["subset", "seed", "width_error"]].to_dict("records"),
        )
    report, digest = sensitivity.sensitivity_report(
        base, ctx.store.report_dir("sensitivity")
    )
    ctx.details |= {"report": str(report), "second_weight":
                    digest.get("F10", {}).get("second_weight")}  # fmt: skip


def _validate(ctx: StageContext) -> None:
    from spleleton.bench.variants import get_variant

    scope = _run_scope(ctx.quick)
    if ctx.quick:
        scope = {**scope, "subsets": QUICK_IMPLICIT}
    validate_runs(
        ctx, get_variant("phase5"), scope,
        VALIDATE_F["quick" if ctx.quick else "full"], second=not ctx.quick,
    )  # fmt: skip


def validate_runs(
    ctx: StageContext, named, scope, fs, second: bool = True, params=None
) -> None:
    """The validate stage's work: paired re-fits at lambda (1 +- f).

    The connection weight always; with ``second``, also the BCE or Dice
    weight with the larger elasticity in the sensitivity stage's
    derivatives, chosen and logged before any fit (Section 11.4); or
    ``params``, as given. Writes ``sensitivity/validate/validation.parquet``.
    """
    import pandas as pd

    from spleleton.bench import sensitivity
    from spleleton.bench.compare import read_tables
    from spleleton.bench.runner import read_run, run_variant
    from spleleton.bench.store import write_json

    base = run_variant(ctx.store, named, progress=ctx.log, **scope)
    implicit_path = base / "sensitivity" / "implicit.parquet"
    if not implicit_path.exists():
        raise PrerequisiteMissing("validate needs the sensitivity stage's derivatives")
    tables = read_tables(base)
    implicit = sensitivity.with_outcomes(pd.read_parquet(implicit_path), tables.events)
    elasticity = sensitivity.elasticities(implicit)
    chosen = sensitivity.second_weight(elasticity)
    if params is None:
        params = ["fit.connection_weight"]
        if second and chosen["parameter"] is not None:
            params.append(chosen["parameter"])
    params = list(params)
    ctx.log(f"validating {', '.join(params)}; second weight by elasticity: {chosen}")
    ctx.details["second_weight"] = chosen
    ctx.details["parameters"] = params
    rows, missing = [], {}
    subsets = scope.get("subsets")
    for param in params:
        for f in fs:
            for factor in (1 + f, 1 - f):
                refit = sensitivity.validation_variant(named, param, factor)
                directory = run_variant(ctx.store, refit, progress=ctx.log, **scope)
                ctx.runs.append(str(directory))
                meta = read_run(directory)
                bad = [k for k, v in meta["samples"].items() if v["status"] != "done"]
                if bad:
                    missing[str(directory.name)] = bad
                other = read_tables(directory)
                mine = (
                    implicit
                    if not subsets
                    else implicit[implicit["subset"].isin(subsets)]
                )
                rows += sensitivity.validation_rows(
                    tables.events, other.events, mine, param, factor
                )
    validation = pd.DataFrame(rows)
    out = ctx.store.sensitivity_dir("validate")
    out.mkdir(parents=True, exist_ok=True)
    validation.to_parquet(out / "validation.parquet", index=False)
    summary = sensitivity.validation_agreement(validation)
    write_json(out / "summary.json", summary)
    ctx.details |= {"validation": str(out / "validation.parquet"), "summary": summary}
    paired = bool(len(validation)) and set(validation["parameter"]) == set(params)
    ctx.check("every re-fit is scored and paired with the baseline's run",
              not missing and paired, missing)  # fmt: skip


def _unavailable(step: str) -> None:
    raise StageUnavailable(f"built in step {step}")


# -- the stages ------------------------------------------------------------------------


def _commands_data(quick: bool) -> list[str]:
    return [
        "bench --quick data build"
        if quick
        else "bench data build errors3d-v1 --splits tune test"
    ]


def _commands_baselines(quick: bool) -> list[str]:
    scope = "--quick" if quick else "--suite errors3d-v1 --split tune"
    return [
        f"bench run phase5 phase4 skeletonize skeletonize+clean {scope}",
        f"bench verify phase5@latest --n {1 if quick else 3}",
    ]


def _commands_screen(quick: bool) -> list[str]:
    from spleleton.bench.variants import SCREENED

    if quick:
        return [
            "bench --quick probe phase5 --slim --floor",
            f"bench --quick probe {SCREENED[0]} --slim",
        ]
    return ["bench probe phase5 --split smoke --slim --floor"] + [
        f"bench probe {name} --split smoke --slim" for name in SCREENED
    ]


def _commands_floor(quick: bool) -> list[str]:
    scope = "--quick" if quick else "--suite errors3d-v1 --split tune"
    replicates = REPLICATES["quick" if quick else "full"]
    compare = "bench --quick compare" if quick else "bench compare"
    root = "bench_quick" if quick else "bench"
    return [
        f"bench run phase5 --replicates {replicates} {scope}",
        f"{compare} phase5@latest -o {root}/reports/floor",
    ]


def _commands_diagnose(quick: bool) -> list[str]:
    if quick:
        return [
            "bench --quick repair phase5@latest --max 3",
            "bench --quick run phase5-ground-truth --subsets all",
        ]
    return [
        "bench repair phase5@latest --max 10",
        "bench run phase5-ground-truth --suite errors3d-v1 --split tune",
    ]


def _commands_sensitivity(quick: bool) -> list[str]:
    prefix = "bench --quick" if quick else "bench"
    subsets = f" --subsets {' '.join(QUICK_IMPLICIT)}" if quick else ""
    return [
        f"{prefix} sensitivity thresholds phase5@latest",
        f"{prefix} sensitivity implicit phase5@latest{subsets}",
    ]


def _commands_validate(quick: bool) -> list[str]:
    prefix = "bench --quick" if quick else "bench"
    fs = " ".join(map(str, VALIDATE_F["quick" if quick else "full"]))
    second = "" if quick else " (and the BCE or Dice weight, chosen from S6)"
    return [(f"{prefix} sensitivity validate phase5@latest "
            f"--params fit.connection_weight{second} --f {fs}")]  # fmt: skip


STAGES: dict[str, Stage] = {
    stage.name: stage
    for stage in (
        Stage(
            "data", "S1", (),
            "builds errors3d-v1, tune and test (308 samples)",
            "1.5-2.5 min (quick: 10 s)", _commands_data, _data,
        ),
        Stage(
            "baselines", "S3", ("data",),
            "phase5, phase4, skeletonize and skeletonize+clean on tune; verify",
            "2.1-2.5 h (quick: 6 min)", _commands_baselines, _baselines,
        ),
        Stage(
            "floor", "S4", ("baselines",),
            "two jittered replicates of phase5 on tune, the floor and its report",
            "2.3-2.7 h (quick: 3 min)", _commands_floor, _floor,
        ),
        Stage(
            "diagnose", "S5", ("baselines",),
            "bench repair on phase5; a ground-truth-start phase5 run on tune",
            "1.9-3.9 h (quick: 3 min)", _commands_diagnose, _diagnose,
        ),
        Stage(
            "sensitivity", "S6", ("baselines",),
            "critical thresholds, the restricted grid, implicit derivatives",
            "2.0-2.7 h (quick: 3 min)", _commands_sensitivity, _sensitivity,
        ),
        Stage(
            "validate", "S7", ("sensitivity",),
            "paired re-fits at λ(1 ± f) for two loss weights",
            "8.7-11.4 h (quick: 3 min)", _commands_validate, _validate,
        ),
        Stage(
            "screen", "S2", ("data",),
            "the slim probe on smoke: phase5 (with the floor) and 7 candidates",
            "5.9-8.3 h (quick: about 15 min)", _commands_screen, _screen,
        ),
    )
}  # fmt: skip
#: The order the stages run in.
ORDER = ("data", "baselines", "floor", "diagnose", "sensitivity", "validate", "screen")


# -- running ---------------------------------------------------------------------------


def record_path(store: Store, name: str) -> Path:
    """``stages/<name>.json``."""
    return store.stages_dir / f"{name}.json"


def read_record(store: Store, name: str) -> dict[str, Any] | None:
    """A stage's record, or ``None`` if it never ran."""
    path = record_path(store, name)
    return json.loads(path.read_text()) if path.exists() else None


def resolve(names: list[str]) -> list[str]:
    """Stage names (or labels, or ``all``) in the order they run."""
    wanted = []
    labels = {stage.label.lower(): name for name, stage in STAGES.items()}
    for name in names:
        if name == "all":
            wanted += list(ORDER)
            continue
        key = labels.get(name.lower(), name)
        if key not in STAGES:
            raise KeyError(f"unknown stage {name!r}; known: {', '.join(ORDER)}, all")
        wanted.append(key)
    return [name for name in ORDER if name in wanted]


def dry_run(name: str, quick: bool, projection: dict | None = None) -> list[str]:
    """What a stage would run: its commands and projected time.

    ``projection`` is a :func:`spleleton.bench.collect.budget` result: the
    full stage's time projected from measured units of work.
    """
    stage = STAGES[name]
    lines = [f"{stage.label} {name}: {stage.describe}"]
    lines += [f"  needs: {', '.join(stage.needs) or 'nothing'}"]
    lines += [f"  $ {command}" for command in stage.commands(quick)]
    lines += [f"  planned: {stage.projected}"]
    if projection is not None:
        from spleleton.bench.collect import duration

        row = next(r for r in projection["stages"] if r["stage"] == name)
        measured = sum(u["n"] > 0 for u in projection["units"])
        lines += [
            (
                f"  projected (full, from {measured} measured units of "
                f"{projection.get('store', 'this store')}): {duration(row['low_s'])} "
                f"to {duration(row['high_s'])} ({row['units']})"
            )
        ]
    if stage.recipe is None:
        lines += [f"  not available until step {stage.built_in}"]
    return lines


def run_stage(store: Store, name: str, quick: bool, progress=print) -> dict[str, Any]:
    """Run one stage and write its record (see the module's docstring).

    Raises
    ------
    StageUnavailable, PrerequisiteMissing, DirtyTree
        Before anything runs.
    Exception
        Whatever the recipe raised, after recording the stage as failed.
    """
    stage = STAGES[name]
    if stage.recipe is None:
        _unavailable(stage.built_in)
    for need in stage.needs:
        record = read_record(store, need)
        if record is None or record["status"] != "done":
            raise PrerequisiteMissing(f"{name} needs {need} to be done first")
    tree = check_tree(full=not quick)
    host = host_state()
    warnings = []
    if host.get("on_mains") is False:
        warnings.append(f"on {host['power']}: timings may be throttled")
    if host["load_1m"] > MAX_LOAD:
        warnings.append(f"load average {host['load_1m']:.1f} (> {MAX_LOAD:g})")
    store.stages_dir.mkdir(parents=True, exist_ok=True)
    log_path = store.stages_dir / f"{name}.log"

    def log(line: str) -> None:
        stamp = time.strftime("%H:%M:%S")
        with log_path.open("a") as file:
            file.write(f"{stamp} {line}\n")
        if progress:
            progress(line)

    for warning in warnings:
        log(f"warning: {warning}")
    record = {
        "name": name,
        "label": stage.label,
        "quick": quick,
        "status": "running",
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": tree["commit"],
        "dirty": tree["dirty"],
        "src_hash": src_hash(),
        "host": host,
        "warnings": warnings,
    }
    write_json(record_path(store, name), record)
    log(f"{stage.label} {name}: started")
    ctx = StageContext(store, quick, log)
    started = time.perf_counter()
    try:
        stage.recipe(ctx)
    except BaseException as error:
        record.update(
            status="failed",
            error=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
        raise
    else:
        record["status"] = "done" if all(ctx.checks.values()) else "failed"
    finally:
        record.update(
            ended=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            elapsed_s=time.perf_counter() - started,
            runs=ctx.runs,
            probes=ctx.probes,
            checks=ctx.checks,
            details=ctx.details,
        )
        write_json(record_path(store, name), record)
        log(
            f"{stage.label} {name}: {record['status']} "
            f"in {record['elapsed_s'] / 60:.1f} min"
        )
    return record


def status(store: Store) -> dict[str, Any]:
    """Every stage's status, failed samples, the next command and code drift."""
    from spleleton.bench.runner import read_run

    rows, failed, hashes, commits = [], {}, set(), set()
    next_stage = None
    done = set()
    for name in ORDER:
        stage = STAGES[name]
        record = read_record(store, name)
        state = record["status"] if record else "pending"
        if stage.recipe is None and state == "pending":
            state = f"not built (step {stage.built_in})"
        rows.append(
            {
                "stage": name,
                "label": stage.label,
                "status": state,
                "elapsed_min": None
                if not record
                else (record.get("elapsed_s") or 0) / 60,
                "checks": None if not record else record.get("checks"),
            }
        )
        if record and record["status"] == "done":
            done.add(name)
            hashes.add(record["src_hash"])
            commits.add(record["commit"])
        for run in (record or {}).get("runs", []):
            if Path(run, "run.json").exists():
                meta = read_run(Path(run))
                bad = {
                    k: v.get("reason")
                    for k, v in meta["samples"].items()
                    if v["status"] == "failed"
                }
                if bad:
                    failed[meta["name"]] = bad
    for name in ORDER:
        stage = STAGES[name]
        if name not in done and stage.recipe is not None and set(stage.needs) <= done:
            next_stage = name
            break
    warnings = []
    if len(hashes) > 1:
        warnings.append(f"the done stages ran with {len(hashes)} different src/ hashes")
    if len(commits) > 1:
        warnings.append(f"the done stages ran from {len(commits)} different commits")
    return {
        "stages": rows,
        "failed_samples": failed,
        "next": None if next_stage is None else f"bench stage {next_stage}",
        "warnings": warnings,
    }
