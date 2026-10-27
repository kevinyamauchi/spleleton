"""Phase F's bundle and the time budget (plan, Sections 15.3 and 16).

``bench collect`` (:func:`collect`) gathers what the stages recorded into one
directory, for Claude to read:

- ``report.md``: the stages and their checks; the baselines compared with
  ``phase5`` (``bench compare``'s report, with the noise floor once S4 has
  run); the screening table; the sections of the stages still to come
  (repairs, sensitivity, validation) or why they are empty; the re-measured
  budget;
- ``figures/``: every plot, as PNGs;
- ``tables/``: the merged tables (parquet) and small CSV extracts;
- ``digest.json``: the numbers each question of Section 16.2 needs, keyed
  by question (``F1`` ... ``F13``), small enough to read whole. A question
  whose stage has not run holds ``{"pending": <why>}``.

``bench budget`` (:func:`budget`) re-measures Section 15.3: the units of work
from the stages' own timings (fits, graphs, scores, probe references and
batches), leaving out samples measured on battery or beside another job
(1-minute load above :data:`LOAD_LIMIT`), and the full stages' projected
times from them. Units nothing has measured yet keep stage 0's values, and
say so.
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spleleton.bench.stats import POOLED
from spleleton.bench.store import Store, write_json

#: Timings with a higher 1-minute load average are left out (P10).
LOAD_LIMIT = 13.0
#: Volumes of ``errors3d-v1``: 11 subsets, 8 on tune, 20 on test, 2 on smoke.
TUNE, TEST, SMOKE = 88, 220, 22
#: Probe specs on S2 (the current loss and 7 candidates).
PROBE_SPECS = 8
#: Validation runs (Section 11.4): 2 weights x 2 values of f x (1 +- f).
VALIDATION_RUNS = 8
#: Parameters per alive segment of the implicit derivatives.
PARAMETERS_PER_SEGMENT = 9
#: Stage 0's units (Section 15.3): name -> (median, 90th percentile, source).
STAGE0_UNITS: dict[str, tuple[float, float, str]] = {
    "sample": (0.28, 0.48, "P0"),
    "phase5": (44.0, 57.0, "P1"),
    "phase4": (37.0, 39.0, "P1"),
    "skeletonize": (0.2, 0.2, "P1"),
    "replicate": (47.0, 53.0, "P4"),
    "ground_truth": (45.0, 45.0, "P7"),
    "graph": (0.024, 0.030, "P3"),
    "score": (0.21, 0.56, "P3"),
    "grid_point": (0.027, 0.052, "P3"),
    "reference": (26.0, 27.0, "P6"),
    "probe_batch": (3.9, 5.9, "P6 --slim"),
    "repair": (4.2, 13.0, "P7"),
    "repair_volume": (0.26, 0.51, "P7"),
    "product": (0.21, 0.21, "P9"),
}
#: Suites whose volumes are not the benchmark's size (64^3): not timed.
SMALL_SUITES = ("proto-cases",)
#: Units summed many times per volume: their high end is the mean, not the
#: 90th percentile (the sum of many batches tends to their mean).
SUMMED_UNITS = ("probe_batch",)
#: Stage 0's counts per volume (Section 15.3), and the cap its repairs ran under.
STAGE0_COUNTS = {"probe_batches": 23.9, "repairs": 7.5, "alive": (40.0, 53.0)}
STAGE0_REPAIR_CAP = 10


# -- the budget --------------------------------------------------------------------


@dataclass
class Unit:
    """A unit of work: its measurements, or stage 0's values."""

    name: str
    values: list[float] = field(default_factory=list)
    dropped: int = 0

    @property
    def measured(self) -> bool:
        """Whether the stages measured it."""
        return bool(self.values)

    def quantile(self, high: bool) -> float:
        """The median (``high=False``) or the 90th percentile (mean if summed)."""
        if self.values:
            if high and self.name in SUMMED_UNITS:
                return float(np.mean(self.values))
            return float(np.percentile(self.values, 90 if high else 50))
        median, p90, _ = STAGE0_UNITS[self.name]
        return p90 if high else median

    def source(self) -> str:
        """Where its numbers come from."""
        if self.values:
            return f"{len(self.values)} measured"
        return f"stage 0 ({STAGE0_UNITS[self.name][2]})"


def _usable(host: dict[str, Any] | None) -> bool:
    """A timing on mains, without another job beside it."""
    if not host:
        return True
    load = host.get("load_1m")
    return host.get("on_mains", True) is not False and (
        load is None or load <= LOAD_LIMIT
    )


def measure_units(store: Store) -> tuple[dict[str, Unit], dict[str, Any]]:
    """The units of work and the counts per volume, from the store's records.

    Returns
    -------
    units : dict
        Name -> :class:`Unit` (every name of :data:`STAGE0_UNITS`).
    counts : dict
        ``probe_batches`` and ``repairs`` per volume, ``alive`` (median and
        90th percentile of alive segments per Phase 5 fit), each with its
        source.
    """
    units = {name: Unit(name) for name in STAGE0_UNITS}
    alive: list[float] = []
    for path in sorted(store.root.glob("runs/*/*/run.json")):
        meta = json.loads(path.read_text())
        if meta["suite"] in SMALL_SUITES:
            continue
        spec = meta["spec"]
        if spec["pipeline"] == "skeletonize":
            fit_unit = "skeletonize"
        elif spec.get("init") == "ground_truth":
            fit_unit = "ground_truth"
        elif spec.get("init_jitter", 0) > 0:
            fit_unit = "replicate"
        else:
            fit_unit = "phase5" if spec["pipeline"] == "scratch_fit" else "phase4"
        for entry in meta["samples"].values():
            if entry.get("status") != "done":
                continue
            fit = entry.get("fit") or {}
            usable = _usable(entry.get("host_start")) and _usable(fit.get("host"))
            if "score_s" in entry:
                units["score"].values.append(entry["score_s"])
            graph_s = None
            if not entry.get("graph_cached"):
                graph_s = entry.get("graph_s", _graph_time(store, entry))
            if fit_unit == "skeletonize":  # its graph stage skeletonises
                if graph_s is not None:
                    target = units["skeletonize"]
                    if usable:
                        target.values.append(graph_s)
                    else:
                        target.dropped += 1
                continue
            if graph_s is not None:
                units["graph"].values.append(graph_s)
            if fit and not entry.get("fit_cached"):
                if usable:
                    units[fit_unit].values.append(fit["timing"]["wall_s"])
                else:
                    units[fit_unit].dropped += 1
    for path in sorted(store.root.glob("runs/*/*/samples.parquet")):
        frame = pd.read_parquet(path)
        meta = json.loads((path.parent / "run.json").read_text())
        if meta["suite"] in SMALL_SUITES:
            continue
        if meta["spec"]["pipeline"] == "scratch_fit" and "n_alive" in frame:
            alive += frame[frame["stage"] == "raw"]["n_alive"].dropna().tolist()
    # repairs: each repair's relaxation and checks, and the rest of a volume
    repairs_per_volume, repairs_capped = [], False
    for path in sorted(store.root.glob("runs/*/*/repairs.parquet")):
        meta = json.loads((path.parent / "run.json").read_text())
        if meta["suite"] in SMALL_SUITES:
            continue
        settings = path.parent / "repairs.json"
        cap = json.loads(settings.read_text()).get("cap") if settings.exists() else None
        repairs_capped |= cap is not None and cap < STAGE0_REPAIR_CAP
        frame = pd.read_parquet(path)
        if "time_s" not in frame:
            continue
        for _, volume in frame.groupby(["subset", "seed"]):
            ran = volume.dropna(subset=["time_s"]).drop_duplicates("repair_id")
            units["repair"].values += ran["time_s"].tolist()
            if "volume_s" in volume:
                rest = float(volume["volume_s"].iloc[0]) - float(ran["time_s"].sum())
                units["repair_volume"].values.append(max(rest, 0.0))
            runnable = volume[volume["status"].isin(["ok", "over the cap"])]
            repairs_per_volume.append(runnable["repair_id"].nunique())
    # the sensitivity stage: grid points and Hessian products
    for path in sorted(store.root.glob("runs/*/*/sensitivity/grid.parquet")):
        meta = json.loads((path.parents[1] / "run.json").read_text())
        if meta["suite"] in SMALL_SUITES:
            continue
        frame = pd.read_parquet(path, columns=["subset", "seed", "setting", "time_s"])
        units["grid_point"].values += frame.drop_duplicates(
            ["subset", "seed", "setting"]
        )["time_s"].tolist()
    for path in sorted(store.root.glob("runs/*/*/sensitivity/implicit_info.parquet")):
        meta = json.loads((path.parents[1] / "run.json").read_text())
        if meta["suite"] in SMALL_SUITES:
            continue
        frame = pd.read_parquet(path)
        if {"hessian_s", "parameters"} <= set(frame.columns):
            per = (frame["hessian_s"] / frame["parameters"]).dropna()
            units["product"].values += per.tolist()
    batches, capped = [], False
    for path in sorted(store.root.glob("probes/*/*/probe.json")):
        meta = json.loads(path.read_text())
        if meta["suite"] in SMALL_SUITES:
            continue
        for volume in meta.get("volumes", {}).values():
            if volume.get("status") != "done":
                continue
            capped |= volume.get("max_events") is not None
            reference = volume.get("reference", {}).get("time_s")
            if reference is not None:
                units["reference"].values.append(reference)
            batches.append(volume.get("events", 0))
        events = path.parent / "events.parquet"
        if events.exists():
            frame = pd.read_parquet(events)
            if {"time_total_s", "time_compile_s"} <= set(frame.columns):
                net = frame["time_total_s"] - frame["time_compile_s"].fillna(0)
                units["probe_batch"].values += net.dropna().tolist()
    counts = {
        "probe_batches": (
            (float(np.mean(batches)), "measured")
            if batches and not capped
            else (STAGE0_COUNTS["probe_batches"], "stage 0 (the probes were capped)")
        ),
        "repairs": (
            (float(np.mean(repairs_per_volume)), "measured")
            if repairs_per_volume and not repairs_capped
            else (
                STAGE0_COUNTS["repairs"],
                "stage 0 (the repairs were capped)"
                if repairs_per_volume
                else "stage 0",
            )
        ),
        "alive": (
            (float(np.median(alive)), float(np.percentile(alive, 90)), "measured")
            if alive
            else (*STAGE0_COUNTS["alive"], "stage 0")
        ),
    }
    return units, counts


def _graph_time(store: Store, entry: dict[str, Any]) -> float | None:
    """A graph stage's time, from its cached file (runs from before ``graph_s``)."""
    path = store.graph_path(entry["graph_key"]) if entry.get("graph_key") else None
    if path is None or not path.exists():
        return None
    with np.load(path) as data:
        return json.loads(str(data["meta"])).get("time_s")


def stage_costs(units: dict[str, Unit], counts: dict[str, Any]) -> list[dict[str, Any]]:
    """The full stages' projected times (low: medians; high: 90th percentiles)."""
    from spleleton.bench.sensitivity import grid_settings
    from spleleton.bench.variants import get_variant

    batches = counts["probe_batches"][0]
    repairs = counts["repairs"][0]
    alive_median, alive_p90 = counts["alive"][:2]
    settings = len(grid_settings(get_variant("phase5").variant.graph))

    def run(n: int, fit: str):
        return lambda u, h: n * (u(fit) + u("graph") + u("score"))

    plans = [
        ("data", "S1", f"{TUNE + TEST} samples", lambda u, h: (TUNE + TEST) * u("sample")),  # noqa: E501
        (
            "baselines", "S3", f"4 baselines x {TUNE} volumes, verify 3",
            lambda u, h: run(TUNE, "phase5")(u, h) + run(TUNE, "phase4")(u, h)
            + 2 * TUNE * (u("skeletonize") + u("score")) + 3 * u("phase5"),
        ),
        ("floor", "S4", f"2 x {TUNE} jittered volumes", run(2 * TUNE, "replicate")),
        (
            "diagnose", "S5", f"{TUNE} volumes x {repairs:.1f} repairs; ground truth",
            lambda u, h: TUNE * (u("repair_volume") + repairs * u("repair"))
            + run(TUNE, "ground_truth")(u, h),
        ),
        (
            "sensitivity", "S6",
            f"{settings} grid points; {PARAMETERS_PER_SEGMENT} x alive products",
            lambda u, h: TUNE * settings * u("grid_point")
            + TUNE * PARAMETERS_PER_SEGMENT * (alive_p90 if h else alive_median)
            * u("product"),
        ),
        ("validate", "S7", f"{VALIDATION_RUNS} runs x {TUNE} volumes",
         run(VALIDATION_RUNS * TUNE, "phase5")),
        (
            "screen", "S2", f"{PROBE_SPECS} specs x {SMOKE} volumes x {batches:.1f} batches",  # noqa: E501
            lambda u, h: PROBE_SPECS * SMOKE * (u("reference") + batches * u("probe_batch")),  # noqa: E501
        ),
    ]  # fmt: skip
    rows = []
    for name, label, what, cost in plans:
        low = cost(lambda n: units[n].quantile(False), False)
        high = cost(lambda n: units[n].quantile(True), True)
        rows.append(
            {"stage": name, "label": label, "units": what, "low_s": low, "high_s": high}
        )
    return rows


def budget(store: Store) -> dict[str, Any]:
    """Units, counts and the stages' projected and recorded times."""
    from spleleton.bench.stages import read_record

    units, counts = measure_units(store)
    rows = stage_costs(units, counts)
    for row in rows:
        record = read_record(store, row["stage"])
        row["recorded_s"] = (record or {}).get("elapsed_s")
    return {
        "units": [
            {
                "unit": u.name,
                "n": len(u.values),
                "dropped": u.dropped,
                "median_s": u.quantile(False),
                "p90_s": u.quantile(True),
                "source": u.source(),
            }
            for u in units.values()
        ],
        "counts": counts,
        "stages": rows,
    }


def duration(seconds: float | None) -> str:
    """Seconds as s, min or h."""
    if seconds is None or not np.isfinite(seconds):
        return "-"
    if seconds < 90:
        return f"{seconds:.2g} s" if seconds < 10 else f"{seconds:.0f} s"
    if seconds < 90 * 60:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def budget_markdown(result: dict[str, Any]) -> list[str]:
    """The budget's two tables, in the plans' format."""
    from spleleton.bench.compare import _table

    lines = [
        (
            "Units of work (timings on battery or with a load above "
            f"{LOAD_LIMIT:g} left out):"
        ),
        "",
    ]
    lines += _table(
        ["unit", "n", "left out", "median", "90th percentile", "source"],
        [
            [u["unit"], u["n"], u["dropped"], duration(u["median_s"]),
             duration(u["p90_s"]), u["source"]]
            for u in result["units"]
        ],
    )  # fmt: skip
    counts = result["counts"]
    lines += [
        (
            f"Counts per volume: {counts['probe_batches'][0]:.1f} probe batches "
            f"({counts['probe_batches'][1]}), {counts['repairs'][0]:.1f} repairs "
            f"({counts['repairs'][1]}), alive segments per Phase 5 fit "
            f"{counts['alive'][0]:.0f} (median) to {counts['alive'][1]:.0f} (90th "
            f"percentile; {counts['alive'][2]})."
        ),
        "",
    ]
    lines += _table(
        ["stage", "", "units", "projected", "recorded (this store)"],
        [
            [r["label"], r["stage"], r["units"],
             f"{duration(r['low_s'])} to {duration(r['high_s'])}",
             duration(r["recorded_s"])]
            for r in result["stages"]
        ],
    )  # fmt: skip
    return lines


# -- the bundle --------------------------------------------------------------------


def _stage_runs(store: Store, name: str) -> list[Path]:
    from spleleton.bench.stages import read_record

    record = read_record(store, name) or {}
    return [Path(r) for r in record.get("runs", []) if Path(r, "run.json").exists()]


def _stage_probes(store: Store, name: str) -> list[Path]:
    from spleleton.bench.stages import read_record

    record = read_record(store, name) or {}
    return [Path(p) for p in record.get("probes", []) if Path(p, "probe.json").exists()]


def _demote(text: str) -> list[str]:
    """Markdown with every heading one level down, without its title."""
    lines = text.splitlines()
    out = []
    for line in lines[1:]:
        out.append("#" + line if line.startswith("#") else line)
    return out


def screening(probes: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loss-correct rates per candidate and family, and each candidate's screen.

    The probes of one spec on several suites are pooled; the current loss
    (``phase5``) is the baseline, with its rate floor where it has one.
    """
    from spleleton.bench.probe import load_probe, loss_correct_rates
    from spleleton.bench.probe._analysis import rate_floor, screen

    by_name: dict[str, list[pd.DataFrame]] = {}
    for probe in probes:
        name = probe.name.split("__")[0].removesuffix("-slim")
        by_name.setdefault(name, []).append(load_probe(probe).events)
    events = {k: pd.concat(v, ignore_index=True) for k, v in by_name.items()}
    rates = []
    for name, frame in events.items():
        rates.append(loss_correct_rates(frame).assign(candidate=name))
    rates = pd.concat(rates, ignore_index=True) if rates else pd.DataFrame()
    screens = []
    base = events.get("phase5")
    if base is not None:
        floor = rate_floor(base)
        for name, frame in events.items():
            if name == "phase5":
                continue
            table, passes = screen(frame, base, floor)
            screens.append(table.assign(candidate=name, passes=passes))
    return rates, pd.concat(screens, ignore_index=True) if screens else pd.DataFrame()


def collect(store: Store, output: Path | str | None = None) -> Path:
    """Write the Phase F bundle (see the module's docstring).

    Parameters
    ----------
    store : Store
        The store whose stages are collected (``bench_quick/`` with
        ``--quick``).
    output : Path or str, optional
        The bundle's directory (``reports/collect`` in the store by default).

    Returns
    -------
    Path
        ``report.md``.
    """
    from spleleton.bench import plots
    from spleleton.bench.compare import _fmt, _table, compare, compare_runs
    from spleleton.bench.stages import STAGES, read_record, status

    output = Path(output) if output else store.root / "reports" / "collect"
    (output / "tables").mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    state = status(store)
    digest: dict[str, Any] = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "store": str(store.root),
        "quick": store.quick,
        "stages": {row["stage"]: row for row in state["stages"]},
        "warnings": state["warnings"],
        "failed_samples": state["failed_samples"],
    }
    lines = [
        "# Benchmark results bundle",
        "",
        f"*{time.strftime('%d %B %Y')}. Store `{store.root}`"
        + (" (quick)" if store.quick else "")
        + ". `digest.json` holds the numbers by question (Section 16.2).*",
        "",
        "## Stages",
        "",
    ]
    lines += _table(
        ["stage", "", "status", "elapsed", "checks"],
        [
            [STAGES[r["stage"]].label, r["stage"], r["status"],
             duration(None if r["elapsed_min"] is None else r["elapsed_min"] * 60),
             "-" if not r["checks"]
             else ", ".join(f"{k}: {'ok' if v else 'FAILED'}" for k, v in r["checks"].items())]  # noqa: E501
            for r in state["stages"]
        ],
    )  # fmt: skip
    lines += [f"> **Warning:** {w}" for w in state["warnings"]]
    for name, samples in state["failed_samples"].items():
        lines += [f"Failed in `{name}`: " + ", ".join(samples), ""]

    # F1-F4: the baselines; F5: the floor (once S4 has run)
    runs = _stage_runs(store, "baselines")
    by_name = {json.loads((r / "run.json").read_text())["name"]: r for r in runs}
    lines += ["## Baselines (F1 to F5)", ""]
    if by_name:
        # phase5 is the baseline (the first run if the stage has none)
        first = "phase5" if "phase5" in by_name else next(iter(by_name))
        order = [by_name[first], *[r for n, r in by_name.items() if n != first]]
        result = compare(store, order, by_name[first])
        report = compare_runs(store, order, output, order[0], "baselines.md", result)
        lines += _demote(report.read_text())
        digest |= _baseline_digest(result)
        for name in ("comparisons", "flips"):
            frame = getattr(result, name)
            if len(frame):
                frame[frame["subset"] == POOLED].to_csv(
                    output / "tables" / f"{name}_pooled.csv", index=False
                )
    else:
        lines += ["Stage S3 (baselines) has not run.", ""]
        digest |= {f"F{i}": {"pending": "stage S3 has not run"} for i in range(1, 6)}

    # F6, F7: the screen
    lines += ["## Screening (F6, F7)", ""]
    probes = _stage_probes(store, "screen")
    if probes:
        rates, screens = screening(probes)
        if len(rates):
            rates.to_csv(output / "tables" / "screening_rates.csv", index=False)
            figure = plots.plot_screening(
                rates, "phase5", output / "figures" / "screening.png"
            )
            lines += _table(
                ["candidate", "family", "events", "volumes", "loss-correct",
                 "95% interval"],
                [
                    [r.candidate, r.family, r.n_events, r.n_volumes, _fmt(r.rate),
                     f"[{_fmt(r.low)}, {_fmt(r.high)}]"]
                    for r in rates.itertuples()
                ],
            )  # fmt: skip
            lines += [f"![screening]({figure.relative_to(output).as_posix()})", ""]
        if len(screens):
            screens.to_csv(output / "tables" / "screens.csv", index=False)
            lines += ["Against `phase5` (the screen's rule, Section 10.7):", ""]
            lines += _table(
                ["candidate", "family", "events", "difference", "95% interval",
                 "floor", "gain", "loss", "passes"],
                [
                    [r.candidate, r.family, r.n_events, _fmt(r.difference),
                     f"[{_fmt(r.low)}, {_fmt(r.high)}]", _fmt(r.floor),
                     r.gain, r.loss, r.passes]
                    for r in screens.itertuples()
                ],
            )  # fmt: skip
        digest["F6"] = {
            "rates": rates.to_dict("records") if len(rates) else [],
        }
        digest["F7"] = {
            "screens": screens.to_dict("records") if len(screens) else [],
            "passing": sorted(
                set(screens[screens["passes"]]["candidate"]) if len(screens) else []
            ),
        }
    else:
        lines += ["Stage S2 (screen) has not run.", ""]
        digest |= {k: {"pending": "stage S2 has not run"} for k in ("F6", "F7")}

    # F8: repairs and the ground-truth start
    lines += ["## Repairs and the ground-truth start (F8)", ""]
    record = read_record(store, "diagnose")
    if record is not None and record["status"] == "done":
        section, digest["F8"] = _diagnose_section(record, output)
        lines += section
    else:
        state_of = digest["stages"]["diagnose"]["status"]
        lines += [f"Stage S5 (diagnose): {state_of}.", ""]
        digest["F8"] = {"pending": f"stage S5 (diagnose): {state_of}"}

    # F9, F10: sensitivity; F11, F12: validation
    lines += ["## Sensitivity (F9, F10)", ""]
    record = read_record(store, "sensitivity")
    if record is not None and record["status"] == "done":
        from spleleton.bench.sensitivity import sensitivity_report

        report, found = sensitivity_report(
            Path(record["runs"][0]), output / "sensitivity"
        )
        lines += _demote(report.read_text())
        digest["F9"] = found.get("F9", {"pending": "no threshold tables"})
        digest["F10"] = found.get("F10", {"pending": "no implicit derivatives"})
    else:
        state_of = digest["stages"]["sensitivity"]["status"]
        lines += [f"Stage S6 (sensitivity): {state_of}.", ""]
        digest |= {k: {"pending": f"stage S6 (sensitivity): {state_of}"}
                   for k in ("F9", "F10")}  # fmt: skip
    lines += ["## Validation (F11, F12)", ""]
    record = read_record(store, "validate")
    if record is not None and record["status"] == "done":
        section, digest["F11"], digest["F12"] = _validate_section(record, output)
        lines += section
    else:
        state_of = digest["stages"]["validate"]["status"]
        lines += [f"Stage S7 (validate): {state_of}.", ""]
        digest |= {k: {"pending": f"stage S7 (validate): {state_of}"}
                   for k in ("F11", "F12")}  # fmt: skip

    # F13: the budget
    measured = budget(store)
    lines += ["## Budget (F13)", ""]
    lines += budget_markdown(measured)
    digest["F13"] = {"budget": measured, **digest.pop("_idle", {})}
    pd.DataFrame(measured["units"]).to_csv(output / "tables" / "budget_units.csv")
    pd.DataFrame(measured["stages"]).to_csv(output / "tables" / "budget_stages.csv")
    write_json(output / "digest.json", _jsonable(digest))
    report = output / "report.md"
    report.write_text("\n".join(lines) + "\n")
    return report


def _validate_section(record: dict[str, Any], output: Path) -> tuple[list, dict, dict]:
    """F11's and F12's report section and digest, from the validate stage's record."""
    from spleleton.bench import plots
    from spleleton.bench.compare import _fmt, _table
    from spleleton.bench.sensitivity import validation_agreement

    details = record["details"]
    table = pd.read_parquet(details["validation"])
    table.to_csv(output / "tables" / "validation.csv", index=False)
    pooled = table[table["subset"] == POOLED]
    chosen = details.get("second_weight") or {}
    second = (
        f"`{chosen['parameter']}` (elasticity {_fmt(chosen.get('elasticity'))} on "
        f"{chosen.get('outcome')}, {chosen.get('subset')})"
        if chosen.get("parameter")
        else "none"
    )
    validated = ", ".join(f"`{p}`" for p in details.get("parameters", []))
    alone = len(details.get("parameters", [])) < 2
    lines = [
        (
            f"Validated: {validated}. The second weight by S6's elasticities: "
            f"{second}{' (not validated here)' if alone else ''}."
        ),
        "",
    ]
    lines += _table(
        ["parameter", "factor", "outcome", "events", "observed", "predicted"],
        [[r.parameter, _fmt(r.factor), r.outcome, r.events, _fmt(r.observed),
          _fmt(r.predicted)] for r in pooled.itertuples()],
    )  # fmt: skip
    if len(pooled):
        figure = plots.plot_validation(table, output / "figures" / "validation.png")
        lines += [f"![validation]({figure.relative_to(output).as_posix()})", ""]
    summary = validation_agreement(table)
    morris = summary.pop("morris_needed", None)
    rule = summary.pop("rule", None)
    lines += [
        (
            f"Sign agreement {_fmt(summary.get('sign_agreement'))}, within 2x "
            f"{_fmt(summary.get('within_2x'))}, ranking agrees: "
            f"{summary.get('ranking_agrees')}. Morris screening needed: {morris}."
        ),
        "",
    ]
    f12 = {"morris_needed": morris, "rule": rule}
    if morris is None:
        f12["why"] = "fewer than two validated weights: the ranking is undefined"
    return lines, summary, f12


def _diagnose_section(record: dict[str, Any], output: Path) -> tuple[list, dict]:
    """F8's report section and digest, from the diagnose stage's record."""
    from spleleton.bench import plots
    from spleleton.bench.compare import _fmt, _table
    from spleleton.bench.repair import REPAIRS, summary

    lines: list[str] = []
    table = pd.read_parquet(record["details"]["repairs"]["table"])
    result = summary(table)
    errors = table.drop_duplicates(["subset", "seed", "error_id"])
    errors = errors[errors["error_type"].isin(list(REPAIRS))].assign(
        label=lambda f: f["error_class"].fillna("none"),
        cause_kind=lambda f: f["cause"].astype(str).str.split("#").str[0],
    )
    shares = {}
    for column in ("error_type", "cause_kind"):
        counts = errors.groupby([column, "label"]).size().unstack(fill_value=0)
        shares[column] = {
            str(k): {c: int(v) for c, v in row.items()} for k, row in counts.iterrows()
        }
        lines += [f"Errors by class, per {column.replace('_kind', '')}:", ""]
        labels = list(counts.columns)
        lines += _table(
            [column.replace("_kind", ""), *labels, "n"],
            [[k, *(int(row[c]) for c in labels), int(row.sum())]
             for k, row in counts.iterrows()],
        )  # fmt: skip
    if len(errors):
        figure = plots.plot_repair_classes(
            table, output / "figures" / "repair_classes.png"
        )
        lines += [f"![repair classes]({figure.relative_to(output).as_posix()})", ""]
    table.to_csv(output / "tables" / "repairs.csv", index=False)
    lines += [
        (
            f"{result['repairs_run']} repairs run ({result['repairs_valid']} valid, "
            f"{result['escaped']} escaped their crop); {result['classified']} of "
            f"{result['repairable']} repairable errors classed."
        ),
        "",
    ]
    truth = record["details"].get("ground_truth", {})
    frame = pd.DataFrame(truth.get("metrics", []))
    comparison = {}
    if len(frame):
        means = frame.groupby("stage").mean(numeric_only=True)
        metrics = sorted({c.rsplit("_", 1)[0] for c in means if c != "seed"})
        lines += ["The ground-truth start against the normal start (means):", ""]
        rows = []
        for stage, row in means.iterrows():
            for metric in metrics:
                a, b = row[f"{metric}_normal"], row[f"{metric}_truth"]
                rows.append([stage, metric, _fmt(a), _fmt(b), _fmt(b - a)])
                comparison[f"{stage}/{metric}"] = {"normal": a, "truth": b}
        lines += _table(
            ["stage", "metric", "normal", "ground truth", "difference"], rows
        )
    losses = truth.get("final_loss", {})
    if losses:
        normal = float(np.mean([v["normal"] for v in losses.values()]))
        ground = float(np.mean([v["truth"] for v in losses.values()]))
        lines += [
            (
                f"Final loss: {normal:.6f} (normal) against {ground:.6f} "
                f"(ground truth), mean over {len(losses)} volumes."
            ),
            "",
        ]
        comparison["final_loss"] = {"normal": normal, "truth": ground}
    digest = {
        "repairs": result,
        "classes": shares,
        "ground_truth": comparison or {"pending": "no ground-truth sample"},
    }
    return lines, digest


def _baseline_digest(result: Any) -> dict[str, Any]:
    """F1 to F5 and the idle rounds of F13, from the baselines' comparison."""
    comparisons = result.comparisons
    pooled = (
        comparisons[comparisons["subset"] == POOLED]
        if len(comparisons)
        else comparisons
    )
    base = result.baseline
    events = base.events
    out: dict[str, Any] = {}
    out["F1"] = {
        "baseline": base.name,
        "pooled": pooled[
            ["run", "stage", "metric", "value_baseline", "value", "mean", "low",
             "high", "floor", "flagged"]
        ].to_dict("records") if len(pooled) else [],
        "flagged_per_subset": comparisons[comparisons["flagged"]][
            ["run", "subset", "stage", "metric", "mean", "low", "high"]
        ].to_dict("records") if len(comparisons) else [],
    }  # fmt: skip
    rates: dict[str, Any] = {}
    if len(events):
        for stage in ("raw", "clean"):
            mine = events[events["stage"] == stage]
            junctions = mine[
                (mine["event_type"] == "junction") & mine["thin"].astype(bool)
            ]
            windows = mine[
                (mine["event_type"] == "window") & mine["continuous"].astype(bool)
            ]
            bumps = mine[mine["event_type"] == "bump"]
            rates[stage] = {
                "thin_junctions": len(junctions),
                "thin_connected_by_fit_class": junctions.groupby("fit_class")[
                    "connected"
                ].agg(["size", "mean"]).reset_index().to_dict("records")
                if "fit_class" in junctions and len(junctions) else [],
                "continuous_windows": len(windows),
                "continuous_window_break_rate": float(
                    1 - windows["connected"].astype(bool).mean()
                ) if len(windows) else None,
                "bump_spur_by_height": bumps.groupby(
                    bumps["height_over_r"].round(1)
                )["spur"].agg(["size", "mean"]).reset_index().to_dict("records")
                if len(bumps) else [],
            }  # fmt: skip
    out["F2"] = {"run": base.name, **rates}
    errors = base.errors
    volumes = max(base.samples[["subset", "seed"]].drop_duplicates().shape[0], 1)
    mix: dict[str, Any] = {}
    if len(errors):
        for stage in ("raw", "clean"):
            mine = errors[errors["stage"] == stage]
            kinds = mine["cause"].str.split("#").str[0]
            missed = mine[mine["type"] == "missed"]
            noisy = missed[
                missed["subset"].astype(str).str.startswith(("noise", "all"))
            ]
            mix[stage] = {
                "per_volume_by_type": (mine["type"].value_counts() / volumes).to_dict(),
                "per_volume_by_cause": (kinds.value_counts() / volumes).to_dict(),
                "off_centre_sources": mine[mine["type"] == "off_centre"]["source"]
                .value_counts().to_dict(),
                "unexplained_share": float((kinds == "unexplained").mean()),
                "mask_lost_share_of_missed_length_noisy": float(
                    noisy[noisy["cause"] == "mask_lost"]["size"].sum()
                    / max(noisy["size"].sum(), 1e-9)
                ) if len(noisy) else None,
            }  # fmt: skip
    out["F3"] = {"run": base.name, **mix}
    out["F4"] = _contacts(base)
    floors = result.floors
    out["F5"] = (
        {
            "replicates": floors.replicates,
            "floors": floors.metrics[floors.metrics["subset"] == POOLED].to_dict(
                "records"
            ),
            "null_flip_rates": floors.flips[floors.flips["subset"] == POOLED].to_dict(
                "records"
            ),
        }
        if floors.replicates > 1
        else {"pending": "no replicates of the baseline (stage S4)"}
    )
    rounds = [
        r
        for entry in base.meta["samples"].values()
        for r in (entry.get("fit") or {}).get("rounds", [])[1:]
    ]
    out["_idle"] = {
        "later_rounds": len(rounds),
        "idle_later_rounds": sum(bool(r.get("idle")) for r in rounds),
    }
    return out


#: F4's pairs: (subset, control), paired by seed within one run.
CONTACT_PAIRS = (
    ("contacts", "tortuous-clean"),
    ("tortuous-clean", "clean"),
    ("all-tortuous", "tortuous-clean"),
    ("all", "clean"),
)
CONTACT_METRICS = (
    "n_cycles", "n_junction_extra", "n_components", "centreline_f1", "junction_f1",
    "root_connected_fraction",
)  # fmt: skip


def _contacts(run: Any) -> dict[str, Any]:
    """F4: subsets against their controls, paired by seed, within the baseline."""
    from spleleton.bench.stats import paired_difference

    samples = run.samples
    out: dict[str, Any] = {}
    for subset, control in CONTACT_PAIRS:
        a = samples[samples["subset"] == control]
        b = samples[samples["subset"] == subset]
        if a.empty or b.empty:
            continue
        joined = a.merge(b, on=["seed", "stage"], suffixes=("_a", "_b"))
        for stage, group in joined.groupby("stage"):
            for metric in CONTACT_METRICS:
                if f"{metric}_a" not in group:
                    continue
                d = group[f"{metric}_b"].astype(float) - group[f"{metric}_a"].astype(
                    float
                )
                stats = paired_difference(d.to_numpy())
                out[f"{subset} - {control}/{stage}/{metric}"] = {
                    k: stats[k] for k in ("n", "mean", "low", "high")
                }
    return out or {"pending": "the contact subsets are not in the baseline run"}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, Path):
        return str(value)
    return value
