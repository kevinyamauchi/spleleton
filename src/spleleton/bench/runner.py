"""Running variants: fit, graph and score per sample, each cached (plan, Section 6).

A **run** is a variant (and a replicate index) on some samples of a suite's
split. It lives in ``runs/<suite>/<name>__<key8>/``, where the key hashes the
resolved spec, the replicate, the stage versions and the numerics; a name
pointed at another spec gets another key, so nothing is overwritten.

Per sample, three stages, each cached on disk:

- **fit** (``cache/fit/<key>.npz``, shared by every run): the fitted segments
  as their raw float32 parameters, the segments before the last Adam step,
  each prior's state, the bin capacity and render config of the last
  iteration (with which :func:`check_fit` reproduces the final loss and
  ``history[-1]`` bit for bit, in any process), the layout, the loss
  history, and per round its segments, convergence and timing. Its key is
  the sample's hash and :meth:`Variant.fit_spec` (pipeline, target, fit,
  init, jitter, replicate), so variants that differ only in their graph
  settings share fits;
- **graph** (``cache/graph/<key>.npz``): the raw and the clean graph;
- **score**: the run's own tables (:func:`spleleton.bench.events.score_sample`):
  ``samples.parquet`` (per sample and stage: the graph metrics, the
  root-connected fraction, event summaries, error counts by type and cause,
  and the fit's size and rounds), ``events.parquet`` (per event and stage:
  covariates, outcomes, fit-stage outcomes and decision quantities) and
  ``errors.parquet`` (every error, located, with its cause). Every row
  carries ``metrics_version``. :func:`score_run` scores a run's stored graphs
  again, without refitting.

**Run state.** ``run.json`` holds the resolved spec, provenance (code,
versions, host) and per sample its status (``done`` or ``failed`` with the
reason and traceback), timing, load average and power source, cache keys and
graph hashes. It is written after every sample, so an interrupted run
resumes where it stopped; finished and failed samples are not run again
(``retry`` reruns the failed ones, ``fresh`` ignores the caches).

**Limits.** A sample whose fit runs longer than ``Limits.wall_s`` or holds
more than ``Limits.segments`` segments is stopped (from the fit's callback)
and recorded as failed, with the reason.
"""

import json
import time
import traceback
from collections.abc import Callable, Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from spleleton._serialise import canonical_json, from_dict, to_dict
from spleleton.bench import FIT_VERSION, GRAPH_VERSION, METRICS_VERSION
from spleleton.bench.store import (
    Store,
    array_hash,
    fit_key,
    graph_key,
    key_hash,
    numerics,
    provenance,
    score_key,
    write_json,
)
from spleleton.bench.suites import LoadedSample, get_suite, load_split
from spleleton.bench.timing import enable_compilation_cache, host_state, stopwatch
from spleleton.bench.variants import NamedVariant, Variant
from spleleton.graph import SkeletonGraph

#: The fitted parameters, as stored.
SEGMENT_FIELDS = ("start", "end", "log_radius", "opacity_logit")
#: A run's score tables, each gathered from ``parts/<sample>.<table>.parquet``.
TABLES = ("samples", "events", "errors")
#: Size of the initial jitter of replicates (plan, Section 8.3), physical units.
REPLICATE_JITTER = 1e-3


@dataclass(frozen=True)
class Limits:
    """Per-sample limits (plan, Section 6.5)."""

    wall_s: float = 220.0
    segments: int = 150


class LimitExceeded(RuntimeError):
    """A fit went over a per-sample limit."""


# -- keys and names --------------------------------------------------------------------


def replicate_variant(variant: Variant, replicate: int) -> Variant:
    """Replicate 0 is the variant itself; the others start jittered."""
    if replicate == 0:
        return variant.resolved()
    import dataclasses

    return dataclasses.replace(variant, init_jitter=REPLICATE_JITTER).resolved()


def run_key(variant: Variant, replicate: int = 0) -> str:
    """The key of a run: spec, replicate, stage versions and numerics."""
    return key_hash(
        {
            "variant": variant.resolved(),
            "replicate": replicate,
            "versions": [FIT_VERSION, GRAPH_VERSION, METRICS_VERSION],
            "numerics": numerics(),
        }
    )


def run_label(name: str, replicate: int = 0) -> str:
    """A run's directory name before its key: ``name``, or ``name~r<k>``."""
    return name if replicate == 0 else f"{name}~r{replicate}"


def run_directory(store: Store, suite: str, named: NamedVariant, replicate: int = 0):
    """``runs/<suite>/<name>__<key8>``."""
    variant = replicate_variant(named.variant, replicate)
    return store.run_dir(
        suite, run_label(named.name, replicate), run_key(variant, replicate)
    )


def jitter_seed(sample_sha: str, replicate: int) -> int:
    """The jitter seed of a sample's replicate (each sample draws its own)."""
    return int(key_hash({"sample": sample_sha, "replicate": replicate})[:15], 16)


def sample_label(subset: str, seed: int) -> str:
    """A sample's label in a run: ``<subset>_<seed>``."""
    return f"{subset}_{seed:04d}"


def keys_for(variant: Variant, sample_sha: str, replicate: int) -> tuple[str, str]:
    """The fit and graph cache keys of a sample."""
    spec = variant.fit_spec(replicate)
    fit = (
        fit_key(sample_sha, spec, numerics())
        if spec is not None
        else key_hash({"sample": sample_sha, "fit": None})
    )
    return fit, graph_key(fit, variant.graph)


# -- the fit stage ---------------------------------------------------------------------


@dataclass
class FitRecord:
    """A stored fit, read back."""

    segments: Any
    pre_last: Any
    prior_states: tuple
    capacity: Any
    render_config: Any
    layout: Any
    history: list[dict[str, float]]
    meta: dict[str, Any]


def _regularisers(variant: Variant, n_voxels: int, layout) -> list:
    from spleleton.priors._priors_jax import JaxConnectionLoss, JaxFreeEndCost

    fit = variant.fit
    if variant.pipeline == "scratch_fit":
        return [
            JaxConnectionLoss(n_voxels, fit.connection_weight, fit.connection_reach)
        ]
    if fit.free_end_weight > 0:
        return [JaxFreeEndCost(layout, n_voxels, fit.free_end_weight)]
    return []


def _field(variant: Variant, mask: np.ndarray, spacing) -> np.ndarray:
    from spleleton.targets import target_field

    return target_field(variant.target, mask, spacing)


def ground_truth_start(sample) -> Any:
    """The true graph cut into pieces of about 2 local radii (plan, Section 10.6).

    The probe's reference cuts it the same way, so both starts have the same
    capacity.
    """
    from spleleton.bench.probe._setup import subdivide

    return subdivide(sample.graph, {})[0]


def run_fit(
    variant: Variant,
    loaded: LoadedSample,
    replicate: int,
    limits: Limits,
    serial_timing: bool = False,
    extra_callback: Callable[[int, Any], None] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Fit one sample, within the limits; return the output and its record.

    ``extra_callback`` is called after the limits are checked, at every
    iteration, as the fit's callback (e.g. a :class:`FitRecorder`).

    Raises
    ------
    LimitExceeded
        If the fit runs longer, or holds more segments, than allowed.
    """
    from spleleton.fit._fit_jax import loss_terms_jax
    from spleleton.pipeline import scratch_fit_stage_jax, skeleton_fit_stage_jax

    sample = loaded.sample
    extra = {}
    if variant.init == "ground_truth":
        extra["initial"] = ground_truth_start(sample)
    stage = (
        scratch_fit_stage_jax
        if variant.pipeline == "scratch_fit"
        else skeleton_fit_stage_jax
    )
    stamps: list[tuple[int, float]] = []
    started = time.perf_counter()

    def callback(iteration: int, segments) -> None:
        now = time.perf_counter()
        stamps.append((iteration, now))
        if now - started > limits.wall_s:
            raise LimitExceeded(
                f"time limit: over {limits.wall_s:g} s (round {_round(stamps)}, "
                f"iteration {iteration})"
            )
        if len(segments) > limits.segments:
            raise LimitExceeded(
                f"segment limit: {len(segments)} segments, over {limits.segments}"
            )
        if extra_callback is not None:
            extra_callback(iteration, segments)

    with stopwatch() as watch:
        output = stage(
            sample.mask,
            sample.spacing,
            variant.fit,
            callback=callback,
            target=variant.target,
            init_jitter=variant.init_jitter,
            jitter_seed=jitter_seed(loaded.sha256, replicate),
            diagnostics=True,
            **extra,
        )
    # the loss at the end, and at the last step's segments (history[-1]),
    # as the fit's own step computes them
    field = _field(variant, sample.mask, sample.spacing)
    regularisers = _regularisers(variant, sample.mask.size, output.layout)
    final, overflow = loss_terms_jax(
        output.segments,
        field,
        output.state,
        sample.spacing,
        variant.fit.fit,
        regularisers,
    )
    pre_last, _ = loss_terms_jax(
        output.state.pre_last, field, output.state, sample.spacing, variant.fit.fit,
        regularisers,
    )  # fmt: skip
    rounds = _round_timing(stamps, output.rounds)
    record = {
        "timing": {k: v for k, v in watch.items() if k != "host"},
        "host": watch["host"],
        "rounds": rounds,
        "final_terms": final,
        "final_overflow": overflow,
        "pre_last_terms": pre_last,
        "n_segments": len(output.segments),
    }
    if serial_timing:
        record["iteration_s"] = np.diff([t for _, t in stamps]).tolist()
    return output, record


def _round(stamps: list[tuple[int, float]]) -> int:
    return sum(1 for iteration, _ in stamps if iteration == 0) - 1


def _round_timing(stamps, rounds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Each round's first-iteration and steady time per iteration, from the callback."""
    starts = [i for i, (iteration, _) in enumerate(stamps) if iteration == 0]
    out = []
    for index, info in enumerate(rounds):
        first = starts[index]
        last = starts[index + 1] if index + 1 < len(starts) else len(stamps)
        times = np.diff([t for _, t in stamps[first:last]])
        out.append(
            {
                **info,
                "first_iteration_s": float(times[0]) if len(times) else None,
                "steady_ms": float(np.median(times[1:]) * 1e3)
                if len(times) > 1
                else None,
                "idle": bool(
                    index > 0
                    and info.get("added", 0) == 0
                    and abs(info.get("loss_change", 1.0)) < 1e-4
                ),
            }
        )
    return out


def save_fit(path: Path, output, record: dict[str, Any], meta: dict[str, Any]) -> None:
    """Write a fit to the cache (atomically)."""
    import jax

    arrays: dict[str, np.ndarray] = {}
    for name, segments in (
        ("segments", output.segments),
        ("pre_last", output.state.pre_last),
    ):
        for field in SEGMENT_FIELDS:
            arrays[f"{name}__{field}"] = np.asarray(getattr(segments, field))
    for i, state in enumerate(output.state.prior_states):
        for j, leaf in enumerate(jax.tree_util.tree_leaves(state)):
            arrays[f"state__{i}__{j}"] = np.asarray(leaf)
    if output.layout is not None:
        for key, value in output.layout.to_dict().items():
            arrays[f"layout__{key}"] = np.asarray(value)
    for name in output.history[0]:
        arrays[f"history__{name}"] = np.array([row[name] for row in output.history])
    if "iteration_s" in record:
        arrays["iteration_s"] = np.asarray(record["iteration_s"])
    meta = {
        **meta,
        **{k: v for k, v in record.items() if k != "iteration_s"},
        "capacity": list(output.state.capacity),
        "render_config": to_dict(output.state.render_config),
        "last_history_terms": output.history[-1],
        "n_priors": len(output.state.prior_states),
    }
    arrays["meta"] = np.array(json.dumps(meta, default=_jsonable))
    _save_npz(path, arrays)


def load_fit(path: Path) -> FitRecord:
    """A cached fit, with its segments as float32 ``JaxLineSegments``."""
    import jax.numpy as jnp

    from spleleton.initialise import SegmentLayout
    from spleleton.primitives._primitives_jax import JaxLineSegments
    from spleleton.render._config import RenderConfig
    from spleleton.render._render_jax import BinCapacity

    with np.load(path) as data:
        meta = json.loads(str(data["meta"]))
        segments, pre_last = (
            JaxLineSegments(
                **{f: jnp.asarray(data[f"{name}__{f}"]) for f in SEGMENT_FIELDS}
            )
            for name in ("segments", "pre_last")
        )
        states = []
        for i in range(meta["n_priors"]):
            leaves = sorted(
                (k for k in data.files if k.startswith(f"state__{i}__")),
                key=lambda k: int(k.rsplit("__", 1)[1]),
            )
            states.append(tuple(jnp.asarray(data[k]) for k in leaves))
        layout_keys = [k for k in data.files if k.startswith("layout__")]
        layout = (
            SegmentLayout.from_dict(
                {k.removeprefix("layout__"): data[k] for k in layout_keys}
            )
            if layout_keys
            else None
        )
        names = [
            k.removeprefix("history__") for k in data.files if k.startswith("history__")
        ]
        columns = {n: data[f"history__{n}"] for n in names}
    n = len(next(iter(columns.values()))) if columns else 0
    history = [{name: float(columns[name][i]) for name in names} for i in range(n)]
    return FitRecord(
        segments,
        pre_last,
        tuple(states),
        BinCapacity(*meta["capacity"]),
        from_dict(RenderConfig, meta["render_config"]),
        layout,
        history,
        meta,
    )


def check_fit(path: Path, loaded: LoadedSample, variant: Variant) -> dict[str, Any]:
    """Re-evaluate a stored fit's loss from the file alone.

    The final loss and ``history[-1]`` (at the segments before the last
    step) are recomputed with the fit's own step and its stored state, and
    compared with the stored values bit for bit.

    Returns
    -------
    dict
        ``final`` and ``last_iteration``: whether every term is equal, and
        the largest relative difference.
    """
    from spleleton.fit._fit_jax import FitState, loss_terms_jax

    record = load_fit(path)
    if record.meta["sample_sha256"] != loaded.sha256:
        raise ValueError(f"{path} was fitted on another sample")
    sample = loaded.sample
    state = FitState(
        record.pre_last, record.prior_states, record.capacity, record.render_config
    )
    field = _field(variant, sample.mask, sample.spacing)
    regularisers = _regularisers(variant, sample.mask.size, record.layout)

    def compare(segments, expected):
        found, _ = loss_terms_jax(
            segments, field, state, sample.spacing, variant.fit.fit, regularisers
        )
        worst = max(
            abs(found[k] - expected[k]) / max(abs(expected[k]), 1e-30) for k in expected
        )
        return {
            "bitwise": all(found[k] == expected[k] for k in expected),
            "worst": worst,
        }

    return {
        "final": compare(record.segments, record.meta["final_terms"]),
        "last_iteration": compare(record.pre_last, record.meta["last_history_terms"]),
    }


# -- the graph and score stages --------------------------------------------------------


def save_graphs(
    path: Path, raw: SkeletonGraph, clean: SkeletonGraph, meta: dict
) -> None:
    """Write a graph stage's output to the cache."""
    arrays = {f"raw__{k}": v for k, v in raw.to_dict().items()}
    arrays |= {f"clean__{k}": v for k, v in clean.to_dict().items()}
    arrays["meta"] = np.array(json.dumps(meta, default=_jsonable))
    _save_npz(path, arrays)


def load_graphs(path: Path) -> tuple[SkeletonGraph, SkeletonGraph, dict[str, Any]]:
    """A cached graph stage: the raw graph, the clean graph and its metadata."""
    with np.load(path) as data:
        parts = {
            stage: SkeletonGraph.from_dict(
                {
                    k.removeprefix(f"{stage}__"): data[k]
                    for k in data.files
                    if k.startswith(f"{stage}__")
                }
            )
            for stage in ("raw", "clean")
        }
        meta = json.loads(str(data["meta"]))
    return parts["raw"], parts["clean"], meta


def graph_hash(graph: SkeletonGraph) -> str:
    """SHA-256 of a graph's arrays."""
    return array_hash(graph.to_dict())


def score_tables(
    sample,
    raw: SkeletonGraph,
    clean: SkeletonGraph,
    segments: Any,
    graph: Any,
    fit_meta: dict[str, Any] | None,
) -> dict[str, list[dict[str, Any]]]:
    """The sample's rows of the three score tables.

    Parameters
    ----------
    sample : Sample
        The sample.
    raw, clean : SkeletonGraph
        The graph stage's output.
    segments : JaxLineSegments, optional
        The stored fitted segments (``None`` for variants that fit nothing).
    graph : GraphConfig
        The graph stage's settings.
    fit_meta : dict, optional
        The stored fit's summary (``_fit_summary``), for the samples table's
        fit columns.

    Returns
    -------
    dict
        ``samples``, ``events`` and ``errors``: lists of rows.
    """
    from spleleton.bench.events import score_sample

    scores = score_sample(sample, raw, clean, segments, graph)
    fit_columns = {}
    if fit_meta is not None:
        rounds = fit_meta["rounds"]
        fit_columns = {
            "fit_s": fit_meta["timing"]["wall_s"],
            "n_segments": fit_meta["n_segments"],
            "n_rounds": len(rounds),
            "n_idle_rounds": sum(bool(r.get("idle")) for r in rounds),
            "last_loss_change": rounds[-1].get("loss_change") if rounds else None,
            "last_grad_norm": rounds[-1].get("grad_norm") if rounds else None,
        }
    version = {"metrics_version": METRICS_VERSION}
    return {
        "samples": [{**row, **fit_columns, **version} for row in scores.samples],
        "events": [{**row, **version} for row in scores.events],
        "errors": [{**row, **version} for row in scores.errors],
    }


# -- one sample ------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleJob:
    """Everything a worker needs to process one sample (picklable)."""

    store_root: str
    quick: bool
    variant: dict[str, Any]
    suite: str
    subset: str
    split: str
    seed: int
    replicate: int
    limits: Limits
    fresh: bool
    serial_timing: bool


def process_sample(
    job: SampleJob, loaded: LoadedSample | None = None
) -> dict[str, Any]:
    """Fit (or reuse), assemble (or reuse) and score one sample.

    Returns its status entry for ``run.json``, with its score rows under
    ``rows``. Failures are returned, not raised.
    """
    store = Store(Path(job.store_root), job.quick)
    variant = Variant.from_dict(job.variant)
    started = time.perf_counter()
    entry: dict[str, Any] = {"status": "running", "host_start": host_state()}
    try:
        if loaded is None:
            (loaded,) = load_split(store, job.suite, job.subset, job.split, [job.seed])
        sample = loaded.sample
        fit, graph = keys_for(variant, loaded.sha256, job.replicate)
        entry.update(sample_sha256=loaded.sha256, fit_key=fit, graph_key=graph)
        fit_path, graph_path = store.fit_path(fit), store.graph_path(graph)
        segments = layout = None
        if variant.pipeline != "skeletonize":
            if fit_path.exists() and not job.fresh:
                record = load_fit(fit_path)
                segments, layout = record.segments, record.layout
                entry["fit_cached"] = True
                entry["fit"] = _fit_summary(record.meta)
            else:
                output, record = run_fit(
                    variant, loaded, job.replicate, job.limits, job.serial_timing
                )
                save_fit(
                    fit_path,
                    output,
                    record,
                    {
                        "sample_sha256": loaded.sha256,
                        "fit_spec": json.loads(
                            canonical_json(variant.fit_spec(job.replicate))
                        ),
                        "numerics": numerics(),
                        "fit_version": FIT_VERSION,
                        "provenance": _short_provenance(),
                    },
                )
                segments, layout = output.segments, output.layout
                entry["fit_cached"] = False
                entry["fit"] = _fit_summary(record)
        if graph_path.exists() and not job.fresh:
            raw, clean, graph_meta = load_graphs(graph_path)
            entry["graph_cached"] = True
        else:
            from spleleton.pipeline import graph_stage

            with stopwatch() as watch:
                raw, clean = graph_stage(
                    segments, layout, sample.mask, sample.spacing, variant.graph
                )
            graph_meta = {
                "fit_key": fit,
                "graph": to_dict(variant.graph),
                "graph_version": GRAPH_VERSION,
                "time_s": watch["wall_s"],
                "hashes": {"raw": graph_hash(raw), "clean": graph_hash(clean)},
            }
            save_graphs(graph_path, raw, clean, graph_meta)
            entry["graph_cached"] = False
            entry["graph_s"] = watch["wall_s"]
        entry["graph_hashes"] = graph_meta["hashes"]
        with stopwatch() as watch:
            rows = score_tables(
                sample, raw, clean, segments, variant.graph, entry.get("fit")
            )
        entry.update(
            status="done",
            score_key=score_key(graph),
            score_s=watch["wall_s"],
            rows=rows,
        )
    except LimitExceeded as error:
        entry.update(status="failed", reason=str(error))
    except Exception as error:  # noqa: BLE001 (recorded, never dropped)
        entry.update(
            status="failed",
            reason=f"{type(error).__name__}: {error}",
            traceback=traceback.format_exc(),
        )
    entry["wall_s"] = time.perf_counter() - started
    entry["host_end"] = host_state()
    return entry


def _fit_summary(record: dict[str, Any]) -> dict[str, Any]:
    """What run.json keeps of a fit: timing, segments, rounds."""
    return {
        "timing": record["timing"],
        "host": record.get("host"),
        "n_segments": record["n_segments"],
        "rounds": record["rounds"],
    }


def _short_provenance() -> dict[str, Any]:
    full = provenance()
    return {k: full[k] for k in ("commit", "dirty", "src_hash", "host", "time")}


def _work(job: SampleJob) -> dict[str, Any]:
    """A worker process's job (spawn start method)."""
    enable_compilation_cache(Store(Path(job.store_root), job.quick).jax_cache)
    return process_sample(job)


# -- runs ------------------------------------------------------------------------------


def run_variant(
    store: Store,
    named: NamedVariant,
    suite: str,
    split: str,
    subsets: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
    replicate: int = 0,
    limits: Limits | None = None,
    fresh: bool = False,
    retry: bool = False,
    workers: int = 1,
    serial_timing: bool = False,
    progress: Callable[[str], None] | None = print,
) -> Path:
    """Run a variant on a suite's split, resuming a run that was interrupted.

    Parameters
    ----------
    store : Store
        Where the suite is built and the run is written.
    named : NamedVariant
        The variant and its name.
    suite : str
        The suite.
    split : str
        A split or a view (``smoke``).
    subsets : iterable of str, optional
        The subsets (all by default).
    seeds : iterable of int, optional
        Restrict the seeds.
    replicate : int
        The replicate index; replicates above 0 start jittered (Section 8.3).
    limits : Limits, optional
        Per-sample limits.
    fresh : bool
        Ignore the caches (and refit every sample).
    retry : bool
        Run failed samples again.
    workers : int
        Worker processes (spawned); 1 runs in this process.
    serial_timing : bool
        Store every iteration's time with the fit (implies one worker).
    progress : callable, optional
        Called with a line per sample.

    Returns
    -------
    Path
        The run's directory.
    """
    limits = limits or Limits()
    enable_compilation_cache(store.jax_cache)
    suite_obj = get_suite(suite)
    variant = replicate_variant(named.variant, replicate)
    directory = run_directory(store, suite, named, replicate)
    meta_path = directory / "run.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    else:
        meta = {
            "name": run_label(named.name, replicate),
            "variant_name": named.name,
            "key": run_key(variant, replicate),
            "spec": variant.to_dict(),
            "spec_hash": variant.spec_hash(),
            "sweep": None if named.sweep is None else to_dict(named.sweep),
            "replicate": replicate,
            "suite": suite,
            "split": split,
            "stage_versions": {
                "fit": FIT_VERSION,
                "graph": GRAPH_VERSION,
                "metrics": METRICS_VERSION,
            },
            "numerics": numerics(),
            "samples": {},
        }
    meta["provenance"] = provenance()
    meta.setdefault("starts", []).append(meta["provenance"]["time"])
    jobs, loaded_of, skipped = [], {}, {}
    for subset in subsets or list(suite_obj.subsets):
        for loaded in load_split(store, suite_obj, subset, split, seeds):
            label = sample_label(subset, loaded.seed)
            status = meta["samples"].get(label, {}).get("status")
            skip = (status == "done" and not fresh) or (
                status == "failed" and not (retry or fresh)
            )
            if skip:
                skipped[status] = skipped.get(status, 0) + 1
                continue
            job = SampleJob(
                str(store.root), store.quick, variant.to_dict(), suite, subset,
                loaded.split, loaded.seed, replicate, limits, fresh, serial_timing,
            )  # fmt: skip
            jobs.append((label, job))
            loaded_of[label] = loaded
    if progress:
        line = (
            f"{meta['name']}: {len(jobs)} samples to run ({skipped.get('done', 0)} done"
        )
        if skipped.get("failed"):
            line += f", {skipped['failed']} failed"
        progress(line + ")")

    def record(label: str, entry: dict[str, Any]) -> None:
        tables = entry.pop("rows", None)
        if tables is not None:
            _write_tables(directory, label, tables)
        meta["samples"][label] = entry
        meta["ended"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        write_json(meta_path, meta)
        if progress:
            fit = entry.get("fit", {})
            line = f"  {label}: {entry['status']}"
            if entry["status"] == "failed":
                line += f" ({entry.get('reason')})"
            else:
                cached = "cached" if entry.get("fit_cached") else "fitted"
                if fit:
                    line += f", {cached}, {fit['n_segments']} segments"
                line += f", {entry['wall_s']:.1f} s"
            progress(line)

    write_json(meta_path, meta)
    if workers > 1 and not serial_timing and len(jobs) > 1:
        context = get_context("spawn")
        with ProcessPoolExecutor(workers, mp_context=context) as pool:
            futures = {label: pool.submit(_work, job) for label, job in jobs}
            for label, future in futures.items():
                record(label, future.result())
    else:
        for label, job in jobs:
            record(label, process_sample(job, loaded_of[label]))
    assemble(directory)
    return directory


def _write_tables(
    directory: Path, label: str, tables: dict[str, list[dict[str, Any]]]
) -> None:
    """Write one sample's score tables to ``parts/<label>.<table>.parquet``."""
    for table in TABLES:
        _write_rows(
            directory / "parts" / f"{label}.{table}.parquet", tables[table], label
        )


def _write_rows(path: Path, rows: list[dict[str, Any]], label: str) -> None:
    import pandas as pd

    subset, seed = label.rsplit("_", 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        [{"subset": subset, "seed": int(seed), **_cells(row)} for row in rows]
    )
    frame.to_parquet(path, index=False)


def _cells(row: dict[str, Any]) -> dict[str, Any]:
    """A row's values as parquet stores them: arrays as lists, floats as floats."""
    out = {}
    for key, value in row.items():
        if isinstance(value, np.ndarray):
            value = value.tolist()
        elif isinstance(value, np.generic):
            value = value.item()
        out[key] = value
    return out


def assemble(directory: Path) -> None:
    """Gather the samples' rows into ``samples``, ``events`` and ``errors.parquet``.

    Columns that some event or error types lack hold nulls there; a column
    with values of one kind and nulls keeps that kind (a boolean outcome
    stays boolean).
    """
    import pandas as pd

    for table in TABLES:
        parts = sorted((directory / "parts").glob(f"*.{table}.parquet"))
        frames = [pd.read_parquet(p) for p in parts]
        frames = [f for f in frames if len(f)]
        if frames:
            frame = pd.concat(frames, ignore_index=True)
            frame.to_parquet(directory / f"{table}.parquet", index=False)


def read_run(directory: Path) -> dict[str, Any]:
    """A run's ``run.json``."""
    return json.loads((Path(directory) / "run.json").read_text())


def list_runs(store: Store, suite: str | None = None) -> list[dict[str, Any]]:
    """Every run in a store: name, suite, status counts, versions, ``src/`` hash."""
    rows = []
    for path in sorted(store.root.glob("runs/*/*/run.json")):
        meta = json.loads(path.read_text())
        if suite and meta["suite"] != suite:
            continue
        statuses = [s["status"] for s in meta["samples"].values()]
        rows.append(
            {
                "directory": str(path.parent),
                "name": meta["name"],
                "suite": meta["suite"],
                "split": meta["split"],
                "done": statuses.count("done"),
                "failed": statuses.count("failed"),
                "stage_versions": meta["stage_versions"],
                "src_hash": meta["provenance"]["src_hash"][:12],
                "commit": (meta["provenance"]["commit"] or "")[:10],
            }
        )
    return rows


def score_run(
    store: Store,
    directory: Path,
    progress: Callable[[str], None] | None = print,
) -> Path:
    """Score a run's stored graphs again, without refitting (``bench score``).

    Every done sample is scored from its cached graphs (and fit) with the
    current metrics, and its tables, ``run.json`` entry (``score_key``,
    ``score_s``) and the run's tables are rewritten. With the metrics
    unchanged this reproduces the tables bit for bit.

    Returns
    -------
    Path
        The run's directory.
    """
    directory = Path(directory)
    meta_path = directory / "run.json"
    meta = read_run(directory)
    variant = Variant.from_dict(meta["spec"])
    for label, entry in meta["samples"].items():
        if entry["status"] != "done":
            continue
        subset, seed = label.rsplit("_", 1)
        (loaded,) = load_split(store, meta["suite"], subset, meta["split"], [int(seed)])
        graph_path = store.graph_path(entry["graph_key"])
        if not graph_path.exists():
            raise FileNotFoundError(
                f"{label}: its graphs are not cached ({graph_path})"
            )
        raw, clean, _ = load_graphs(graph_path)
        segments = None
        if variant.pipeline != "skeletonize":
            segments = load_fit(store.fit_path(entry["fit_key"])).segments
        with stopwatch() as watch:
            tables = score_tables(
                loaded.sample, raw, clean, segments, variant.graph, entry.get("fit")
            )
        _write_tables(directory, label, tables)
        entry.update(score_key=score_key(entry["graph_key"]), score_s=watch["wall_s"])
        if progress:
            progress(f"  scored {label} ({watch['wall_s']:.2f} s)")
    write_json(meta_path, meta)
    assemble(directory)
    return directory


def verify_run(
    store: Store,
    directory: Path,
    n: int = 3,
    progress: Callable[[str], None] | None = print,
) -> dict[str, Any]:
    """Re-run a few done samples from scratch and compare their graph hashes.

    Nothing is read from or written to the caches.

    Returns
    -------
    dict
        Per sample: whether both graphs match, and the time; and ``ok``.
    """
    from spleleton.pipeline import graph_stage

    meta = read_run(directory)
    variant = Variant.from_dict(meta["spec"])
    done = [label for label, s in meta["samples"].items() if s["status"] == "done"][:n]
    results = {}
    for label in done:
        subset, seed = label.rsplit("_", 1)
        (loaded,) = load_split(store, meta["suite"], subset, meta["split"], [int(seed)])
        started = time.perf_counter()
        segments = layout = None
        if variant.pipeline != "skeletonize":
            output, _ = run_fit(variant, loaded, meta["replicate"], Limits(1e9, 10**9))
            segments, layout = output.segments, output.layout
        sample = loaded.sample
        raw, clean = graph_stage(
            segments, layout, sample.mask, sample.spacing, variant.graph
        )
        expected = meta["samples"][label]["graph_hashes"]
        found = {"raw": graph_hash(raw), "clean": graph_hash(clean)}
        results[label] = {
            "match": found == expected,
            "time_s": time.perf_counter() - started,
        }
        if progress:
            progress(f"  verify {label}: {'match' if found == expected else 'DIFFERS'}")
    return {
        "samples": results,
        "ok": bool(results) and all(r["match"] for r in results.values()),
    }


def check_jitter(store: Store, directory: Path) -> dict[str, Any]:
    """How far a replicate run's jittered starts moved (plan, Section 8.3).

    Per done sample, the fit's start is built as the fit stage builds it
    (:func:`~spleleton.pipeline._pipeline_jax.fit_start`), without jitter
    and twice with the replicate's seed. Distances are in voxels (each axis
    divided by its spacing). Nothing is fitted.

    Returns
    -------
    dict
        Per sample: ``min_shift`` and ``max_shift``, the smallest and largest
        distance an endpoint moved, and ``rebuilds``, whether the second
        build equals the first bit for bit; and ``jitter``, the run's
        ``init_jitter``.
    """
    from spleleton.pipeline._pipeline_jax import fit_start, jitter_segments

    meta = read_run(directory)
    variant = Variant.from_dict(meta["spec"])
    if variant.pipeline == "skeletonize":
        raise ValueError("a skeletonize run fits nothing, so it has no start")
    results = {}
    for label, entry in meta["samples"].items():
        if entry["status"] != "done":
            continue
        subset, seed = label.rsplit("_", 1)
        (loaded,) = load_split(store, meta["suite"], subset, meta["split"], [int(seed)])
        sample = loaded.sample
        spacing = np.asarray(sample.spacing, dtype=float)
        seed_ = jitter_seed(loaded.sha256, meta["replicate"])
        if variant.init == "ground_truth":
            plain = ground_truth_start(sample)
            starts = [
                jitter_segments(plain, variant.init_jitter, seed_) for _ in range(2)
            ]
        else:
            plain, _ = fit_start(sample.mask, sample.spacing, variant.fit)
            starts = [
                fit_start(
                    sample.mask, sample.spacing, variant.fit, variant.init_jitter, seed_
                )[0]
                for _ in range(2)
            ]
        shifts = np.concatenate(
            [
                np.linalg.norm((starts[0].start - plain.start) / spacing, axis=1),
                np.linalg.norm((starts[0].end - plain.end) / spacing, axis=1),
            ]
        )
        same = all(
            np.array_equal(getattr(starts[0], name), getattr(starts[1], name))
            for name in ("start", "end", "radius", "opacity")
        )
        results[label] = {
            "n_endpoints": int(shifts.size),
            "min_shift": float(shifts.min()),
            "max_shift": float(shifts.max()),
            "rebuilds": bool(same),
        }
    return {"jitter": variant.init_jitter, "samples": results}


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "_asdict"):
        return value._asdict()
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def resolve_run(store: Store, text: str) -> Path:
    """A run directory from a path, or ``name@latest``.

    ``name@latest`` is the newest run of that name (by its last sample) at
    the current stage versions, in any suite.
    """
    path = Path(text)
    if (path / "run.json").exists():
        return path
    name, _, which = text.partition("@")
    if which != "latest":
        raise FileNotFoundError(f"{text} is neither a run directory nor name@latest")
    versions = {"fit": FIT_VERSION, "graph": GRAPH_VERSION, "metrics": METRICS_VERSION}
    candidates = []
    for meta_path in store.root.glob("runs/*/*/run.json"):
        meta = json.loads(meta_path.read_text())
        if meta["name"] == name and meta["stage_versions"] == versions:
            candidates.append((meta.get("ended", ""), meta_path.parent))
    if not candidates:
        raise FileNotFoundError(f"no run named {name!r} in {store.root}")
    return max(candidates)[1]
