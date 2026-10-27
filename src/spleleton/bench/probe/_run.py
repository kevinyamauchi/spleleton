"""Probing volumes, and the probe's results on disk (plan, Section 10.7).

``probes/<suite>/<probe>__<key8>/``:

- ``probe.json``: the resolved spec, its name and key, provenance, and per
  volume its status, reference and skipped events;
- ``reference/<subset>_<seed>.npz``: the relaxed reference segments;
- ``events.parquet``: per (event, family, opacity), its classes,
  loss-correctness, crop and timing;
- ``margins.parquet``: per (event, family, opacity, ε, term), the rigid and
  profile margins, the forces and the finite differences;
- ``floor.parquet``: with ``floor``, per (event, family, opacity, term), the
  profile margin between two jittered relaxations of the reference.

Volumes are written as they finish (``parts/``), so an interrupted probe
resumes where it stopped; the tables are assembled at the end.
"""

import json
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import numpy as np

from spleleton._serialise import from_dict, to_dict
from spleleton.bench.events import extract_events
from spleleton.bench.probe._crop import background
from spleleton.bench.probe._probe import probe_setup, summarise
from spleleton.bench.probe._setup import (
    ProbeEvent,
    Reference,
    make_setup,
    relax_reference,
    select_events,
)
from spleleton.bench.probe._spec import NEW_SLOT_FAMILIES, ProbeSpec
from spleleton.bench.probe._terms import check_losses
from spleleton.bench.store import Store, key_hash, numerics, provenance, write_json
from spleleton.bench.suites import Suite, get_suite, load_split
from spleleton.synthetic.dataset import Sample
from spleleton.targets import target_field


@dataclass
class VolumeProbe:
    """One volume's probe: its reference, rows and skipped events."""

    reference: Reference
    events: list[dict[str, Any]]
    margins: list[dict[str, Any]]
    floor: list[dict[str, Any]]
    skipped: list[dict[str, Any]]
    chosen: list[ProbeEvent]
    results: list[tuple[Any, dict[str, Any]]] = field(default_factory=list)
    time_s: float = 0.0


def probe_volume(
    sample: Sample,
    spec: ProbeSpec,
    seed: int,
    floor: bool = False,
    max_events: int | None = None,
    keep_results: bool = False,
    progress: Callable[[str], None] | None = None,
    reference_iterations: int | None = None,
) -> VolumeProbe:
    """Probe every chosen event of one volume.

    Parameters
    ----------
    sample : Sample
        The volume.
    spec : ProbeSpec
        The probe.
    seed : int
        Seeds the choice of windows and directions, and the floor's jitter
        (e.g. from the sample's hash).
    floor : bool
        Also measure the floor (two jittered relaxations per event).
    max_events : int, optional
        Events per family (``spec.max_events`` by default).
    keep_results : bool
        Keep each set-up and its full result (for checks).
    progress : callable, optional
        Called with a line per event.
    reference_iterations : int, optional
        Override the reference's relaxation length (tests).

    Returns
    -------
    VolumeProbe
    """
    check_losses(spec.losses)
    started = time.perf_counter()
    with jax.enable_x64(True):
        target = target_field(spec.target, sample.mask, sample.spacing)
        target = np.asarray(target, dtype=np.float64)
        events = extract_events(sample)
        chosen, skipped = select_events(sample, events, spec, seed, max_events)
        whole: dict[int, list[tuple[float, float]]] = {}
        for event in chosen:
            whole.setdefault(event.edge, []).extend(event.intervals)
        reference = relax_reference(
            sample, target, spec, whole, chosen, reference_iterations
        )
        if progress:
            info = reference.info
            progress(
                f"  reference: {info['segments']} segments, {info['held']} held, "
                f"{info['time_s']:.1f} s"
            )
        out = VolumeProbe(reference, [], [], [], skipped, chosen)
        for event in chosen:
            opacities = (
                spec.new_slot_opacities
                if event.family in NEW_SLOT_FAMILIES
                else (None,)
            )
            for opacity in opacities:
                setup = make_setup(reference, event, spec, opacity)
                back = background(reference, setup)
                result = probe_setup(
                    reference, setup, back, spec, floor=floor, seed=seed
                )
                row, margins, floors = summarise(reference, setup, result, spec)
                row["time_background_s"] = back.time_s
                out.events.append(row)
                out.margins += margins
                out.floor += floors
                if keep_results:
                    out.results.append((setup, result))
                if progress:
                    progress(
                        f"  {event.event_id} {event.family}"
                        + (f" o={opacity}" if opacity is not None else "")
                        + f": {row['class_profile']}, loss-correct "
                        f"{row['loss_correct']}, {result['timing']['total_s']:.1f} s "
                        f"(compile {result['timing']['compile_s']:.1f} s)"
                    )
    out.time_s = time.perf_counter() - started
    return out


# -- storage ---------------------------------------------------------------------------


def probe_key(spec: ProbeSpec) -> str:
    """The key of a probe: its spec and the float64 numerics."""
    return key_hash({"probe": spec, "numerics": numerics("float64")})


def probe_directory(store: Store, suite: str, name: str, spec: ProbeSpec) -> Path:
    """``probes/<suite>/<name>__<key8>``."""
    return store.probe_dir(suite, name, probe_key(spec))


def run_probe(
    store: Store,
    spec: ProbeSpec,
    name: str,
    suite: Suite | str,
    subsets: Iterable[str],
    split: str,
    seeds: Iterable[int] | None = None,
    floor: bool = False,
    max_events: int | None = None,
    progress: Callable[[str], None] | None = print,
) -> Path:
    """Probe the volumes of a suite's split, resuming finished volumes.

    Parameters
    ----------
    store : Store
        Where the suite is built and the probe is written.
    spec : ProbeSpec
        The probe.
    name : str
        Its label (usually the variant's name).
    suite : Suite or str
        The suite.
    subsets : iterable of str
        The subsets.
    split : str
        A split or view (``smoke``).
    seeds : iterable of int, optional
        Restrict the seeds.
    floor : bool
        Also measure the floor.
    max_events : int, optional
        Events per family and volume.
    progress : callable, optional
        Called with a line per volume and event.

    Returns
    -------
    Path
        The probe's directory.
    """
    suite = get_suite(suite) if isinstance(suite, str) else suite
    directory = probe_directory(store, suite.name, name, spec)
    meta_path = directory / "probe.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else None
    if meta is None:
        meta = {
            "name": name,
            "key": probe_key(spec),
            "spec": to_dict(spec),
            "weights": dict(zip(spec.terms, spec.weights, strict=True)),
            "suite": suite.name,
            "split": split,
            "volumes": {},
            "provenance": provenance(),
        }
    seeds = None if seeds is None else list(seeds)
    for subset in subsets:
        for loaded in load_split(store, suite, subset, split, seeds):
            label = f"{subset}_{loaded.seed:04d}"
            part = directory / "parts" / label
            entry = meta["volumes"].get(label, {})
            if entry.get("status") == "done" and entry.get("floor", False) >= floor:
                if progress:
                    progress(f"{label}: done, skipped")
                continue
            if progress:
                progress(f"{label}:")
            seed = int(loaded.sha256[:15], 16)
            try:
                result = probe_volume(
                    loaded.sample, spec, seed, floor, max_events, progress=progress
                )
            except Exception as error:
                meta["volumes"][label] = {"status": "failed", "error": repr(error)}
                write_json(meta_path, meta)
                raise
            keys = {
                "subset": subset,
                "seed": loaded.seed,
                "sample_sha256": loaded.sha256,
            }
            _write_rows(part.with_suffix(".events.parquet"), result.events, keys)
            _write_rows(part.with_suffix(".margins.parquet"), result.margins, keys)
            if floor:
                _write_rows(part.with_suffix(".floor.parquet"), result.floor, keys)
            reference = result.reference
            path = directory / "reference" / f"{label}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                path,
                **reference.raw,
                initial_start=reference.initial.start,
                initial_end=reference.initial.end,
                initial_radius=reference.initial.radius,
                pieces=json.dumps(reference.pieces),
                held=np.asarray(reference.held, dtype=int),
            )
            meta["volumes"][label] = {
                "status": "done",
                "floor": floor,
                "max_events": max_events,
                "seed_value": seed,
                "sample_sha256": loaded.sha256,
                "reference": reference.info,
                "events": len(result.events),
                "skipped": result.skipped,
                "time_s": result.time_s,
            }
            write_json(meta_path, meta)
    combine(directory)
    return directory


def _write_rows(path: Path, rows: list[dict[str, Any]], keys: dict[str, Any]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([{**keys, **row} for row in rows])
    frame.to_parquet(path, index=False)


def combine(directory: Path) -> None:
    """Assemble the volumes' parts into ``events``, ``margins`` and ``floor``."""
    import pandas as pd

    for table in ("events", "margins", "floor"):
        parts = sorted((directory / "parts").glob(f"*.{table}.parquet"))
        frames = [pd.read_parquet(p) for p in parts]
        frames = [f for f in frames if len(f)]
        if frames:
            pd.concat(frames, ignore_index=True).to_parquet(
                directory / f"{table}.parquet", index=False
            )


@dataclass
class ProbeResults:
    """A probe read back from disk."""

    directory: Path
    meta: dict[str, Any]
    events: Any
    margins: Any
    floor: Any | None

    @property
    def spec(self) -> ProbeSpec:
        """The probe's spec."""
        return from_dict(ProbeSpec, self.meta["spec"])

    @property
    def weights(self) -> dict[str, float]:
        """The weight of each term."""
        return self.meta["weights"]


def load_probe(directory: Path) -> ProbeResults:
    """A probe's metadata and tables."""
    import pandas as pd

    directory = Path(directory)
    meta = json.loads((directory / "probe.json").read_text())
    floor_path = directory / "floor.parquet"
    return ProbeResults(
        directory,
        meta,
        pd.read_parquet(directory / "events.parquet"),
        pd.read_parquet(directory / "margins.parquet"),
        pd.read_parquet(floor_path) if floor_path.exists() else None,
    )


def restore_volume(
    directory: Path, store: Store, subset: str, seed: int
) -> tuple[Reference, list[ProbeEvent], ProbeSpec]:
    """A probed volume's reference and events, rebuilt from disk (no relaxing).

    The events are chosen again with the same seed, so they are the probed
    ones; the reference's segments are the stored relaxed ones.
    """
    from spleleton.bench.probe._setup import capacity_for, full_static
    from spleleton.primitives import SegmentArrays

    results = load_probe(directory)
    spec = results.spec
    meta = results.meta
    label = f"{subset}_{seed:04d}"
    entry = meta["volumes"][label]
    (loaded,) = load_split(store, meta["suite"], subset, meta["split"], [seed])
    sample = loaded.sample
    with jax.enable_x64(True):
        target = np.asarray(
            target_field(spec.target, sample.mask, sample.spacing), dtype=np.float64
        )
        events = extract_events(sample)
        chosen, _ = select_events(
            sample, events, spec, entry["seed_value"], entry.get("max_events")
        )
    with np.load(directory / "reference" / f"{label}.npz") as data:
        raw = {k: data[k] for k in ("start", "end", "log_radius", "opacity_logit")}
        initial = SegmentArrays(
            data["initial_start"], data["initial_end"], radius=data["initial_radius"]
        )
        pieces = json.loads(str(data["pieces"]))
        held = data["held"].tolist()
    shape = tuple(sample.mask.shape)
    spacing = tuple(float(s) for s in sample.spacing)
    st = full_static(
        spec, shape, spacing, capacity_for(raw, shape, spacing, spec.render, 1.0)
    )
    reference = Reference(
        sample,
        target,
        pieces,
        initial,
        raw,
        held,
        st,
        entry["reference"]["grad_median"],
        entry["reference"],
    )
    return reference, chosen, spec
