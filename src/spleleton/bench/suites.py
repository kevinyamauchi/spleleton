"""Benchmark suites: definitions, building with manifests, and loading.

A **suite** is named, versioned and immutable. It maps subset names to a way
of generating a sample from a seed, and split names to seed lists. A split
can also be a **view** onto part of another (``smoke`` is seeds 0-1 of
``tune``), with no files of its own.

**Building** (:func:`build`) writes
``datasets/<suite>/<subset>/<split>/sample_<seed>.npz`` and a
``dataset.json`` manifest per split, with a SHA-256 per sample taken over its
decoded arrays and metadata (:func:`sample_hash`). Every sample is
regenerated in memory and hashed. If one that a manifest lists hashes
differently, or a subset's description differs from its manifest's, the
build fails with :class:`SuiteChanged` and leaves the store as it was: new
files are staged and only moved into place once every check has passed. A
generator change can therefore never silently change the data behind old
results; it needs a new suite version (``errors3d-v2``). New seeds are added
to a manifest, so a later ``test`` build adds to a suite.

**Loading** (:func:`load_split`) goes through the manifest and checks each
file's hash, so every result names exactly the files it was scored on.
"""

import dataclasses
import json
import shutil
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from spleleton.bench.cases import build_case, case_names
from spleleton.bench.store import (
    Store,
    array_hash,
    file_hash,
    provenance,
    sample_name,
    write_json,
)
from spleleton.synthetic.corruptions import (
    BumpConfig,
    ContactConfig,
    CorruptionConfig,
    SurfaceNoiseConfig,
)
from spleleton.synthetic.dataset import (
    Sample,
    SampleConfig,
    airway_config,
    generate_sample,
)

#: Where a split's new samples wait until the whole build has been checked.
_STAGING = ".staging"


class SuiteChanged(RuntimeError):
    """Building would change samples (or subsets) that a manifest already lists."""


class Subset(Protocol):
    """How a subset's samples are made."""

    def generate(self, seed: int) -> Sample:
        """The sample of a seed."""
        ...

    def describe(self) -> dict[str, Any]:
        """What generates the samples, for the manifest (plain JSON types)."""
        ...


@dataclass(frozen=True)
class GeneratedSubset:
    """A subset made by :func:`~spleleton.synthetic.dataset.generate_sample`."""

    config: SampleConfig

    def generate(self, seed: int) -> Sample:
        """The sample of a seed."""
        return generate_sample(self.config, seed)

    def describe(self) -> dict[str, Any]:
        """The generator and its config."""
        return {"generator": "generate_sample", "config": self.config.to_dict()}


@dataclass(frozen=True)
class CaseSubset:
    """One hand-built case of :mod:`spleleton.bench.cases` (seed 0 only)."""

    name: str

    def generate(self, seed: int) -> Sample:
        """The case's sample."""
        if seed != 0:
            raise ValueError(f"case {self.name!r} has only seed 0, not {seed}")
        return build_case(self.name).sample

    def describe(self) -> dict[str, Any]:
        """The case's name and description."""
        return {"case": self.name, "description": build_case(self.name).description}


@dataclass(frozen=True)
class Suite:
    """Named subsets and splits.

    Parameters
    ----------
    name : str
        The suite's name, with its version (``errors3d-v1``).
    subsets : dict of str to Subset
        How each subset's samples are made.
    splits : dict of str to tuple of int
        The seeds of each split that has files.
    views : dict of str to (str, tuple of int)
        Splits without files of their own: the split they view and the
        seeds they take from it.
    """

    name: str
    subsets: dict[str, Subset]
    splits: dict[str, tuple[int, ...]]
    views: dict[str, tuple[str, tuple[int, ...]]] = field(default_factory=dict)

    def resolve(self, split: str) -> tuple[str, tuple[int, ...]]:
        """The split with files and the seeds behind a split or view."""
        if split in self.splits:
            return split, self.splits[split]
        if split in self.views:
            return self.views[split]
        raise KeyError(f"suite {self.name!r} has no split {split!r}")


def errors3d_v1() -> Suite:
    """The suite of the benchmark plan's Section 4.3 (11 subsets, 128^3)."""
    base = airway_config(3, corrupted=False)
    contacts = ContactConfig(min_gap=2.0)
    realistic = dataclasses.replace(
        airway_config(3).corruptions, gaps=None, contacts=contacts
    )

    def tree(**changes) -> SampleConfig:
        return dataclasses.replace(base, tree=dataclasses.replace(base.tree, **changes))

    def corrupted(config: SampleConfig, corruptions: CorruptionConfig) -> SampleConfig:
        return dataclasses.replace(config, corruptions=corruptions)

    tortuous = tree(tortuosity=0.1)
    configs = {
        "clean": base,
        "asym-clean": tree(side_branch_probability=1.0),
    }
    for height in (0.5, 1.0, 1.5):
        bumps = BumpConfig(count=6, height=(height, height), size=(1.0, 1.0))
        configs[f"bumps-h{height}"] = corrupted(base, CorruptionConfig(bumps=bumps))
    for amplitude in (0.5, 1.0):
        noise = SurfaceNoiseConfig(amplitude=amplitude)
        configs[f"noise-a{amplitude}"] = corrupted(
            base, CorruptionConfig(surface_noise=noise)
        )
    configs["all"] = corrupted(base, realistic)
    configs["tortuous-clean"] = tortuous
    configs["contacts"] = corrupted(tortuous, CorruptionConfig(contacts=contacts))
    configs["all-tortuous"] = corrupted(tortuous, realistic)
    return Suite(
        "errors3d-v1",
        {name: GeneratedSubset(config) for name, config in configs.items()},
        splits={"tune": tuple(range(8)), "test": tuple(range(1000, 1020))},
        views={"smoke": ("tune", (0, 1))},
    )


def proto_cases() -> Suite:
    """The hand-built cases, one subset each, in the split ``cases``."""
    return Suite(
        "proto-cases",
        {name: CaseSubset(name) for name in case_names()},
        splits={"cases": (0,)},
    )


#: Suites by name.
SUITES: dict[str, Callable[[], Suite]] = {
    "errors3d-v1": errors3d_v1,
    "proto-cases": proto_cases,
}

#: The quick scope (plan, Section 4.3): per suite, its subsets, split and seeds.
QUICK_SCOPE: dict[str, tuple[tuple[str, ...] | None, str, tuple[int, ...]]] = {
    "errors3d-v1": (("clean", "asym-clean", "bumps-h1.5", "all"), "tune", (0,)),
    "proto-cases": (None, "cases", (0,)),
}


def get_suite(name: str) -> Suite:
    """A suite by name."""
    try:
        return SUITES[name]()
    except KeyError:
        raise KeyError(f"unknown suite {name!r}; known: {', '.join(SUITES)}") from None


# -- hashing --------------------------------------------------------------------------


def sample_payload(sample: Sample) -> dict[str, np.ndarray]:
    """The arrays that :meth:`Sample.save` writes (the hash is taken over these)."""
    metadata = {
        "config": sample.config.to_dict(),
        "seed": sample.seed,
        "corruptions": sample.corruptions,
    }
    graph = {f"graph_{key}": value for key, value in sample.graph.to_dict().items()}
    return {
        "clean_mask": sample.clean_mask,
        "mask": sample.mask,
        "metadata": np.array(json.dumps(metadata)),
        **graph,
    }


def sample_hash(sample: Sample) -> str:
    """SHA-256 over a sample's decoded arrays and metadata."""
    return array_hash(sample_payload(sample))


# -- building -------------------------------------------------------------------------


@dataclass
class BuildReport:
    """What a build did, per (subset, split)."""

    written: dict[tuple[str, str], list[int]] = field(default_factory=dict)
    checked: dict[tuple[str, str], list[int]] = field(default_factory=dict)

    @property
    def n_written(self) -> int:
        """Samples written."""
        return sum(len(seeds) for seeds in self.written.values())

    @property
    def n_checked(self) -> int:
        """Samples already present and checked."""
        return sum(len(seeds) for seeds in self.checked.values())


def build(
    store: Store,
    suite: Suite,
    subsets: Iterable[str] | None = None,
    splits: Iterable[str] = ("tune",),
    seeds: Iterable[int] | None = None,
    progress: Callable[[str], None] | None = None,
) -> BuildReport:
    """Build (or check) splits of a suite.

    Parameters
    ----------
    store : Store
        Where to build.
    suite : Suite
        The suite.
    subsets : iterable of str, optional
        Which subsets; all by default.
    splits : iterable of str
        Which splits (with files of their own; a view is built as the split
        it views, with its seeds).
    seeds : iterable of int, optional
        Restrict the seeds (to those of each split).
    progress : callable, optional
        Called with a line of text per sample.

    Returns
    -------
    BuildReport
        The samples written and the samples checked.

    Raises
    ------
    SuiteChanged
        If a sample regenerates differently from its manifest, a manifest's
        description differs from the subset's, or a file is damaged. Then
        nothing is written.
    """
    names = list(suite.subsets) if subsets is None else list(subsets)
    unknown = sorted(set(names) - set(suite.subsets))
    if unknown:
        raise KeyError(f"suite {suite.name!r} has no subsets {unknown}")
    wanted = None if seeds is None else set(seeds)
    report = BuildReport()
    staged: list[tuple[Path, dict[str, Any], list[dict[str, Any]], Path]] = []
    touched: list[Path] = []
    try:
        for split_name in splits:
            split, split_seeds = suite.resolve(split_name)
            chosen = [s for s in split_seeds if wanted is None or s in wanted]
            for name in names:
                touched.append(store.split_dir(suite.name, name, split) / _STAGING)
                entry = _stage_split(
                    store, suite, name, split, chosen, report, progress
                )
                if entry is not None:
                    staged.append(entry)
    except BaseException:
        for staging in touched:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    for directory, manifest, new, staging in staged:
        for entry in new:
            (staging / entry["file"]).replace(directory / entry["file"])
        shutil.rmtree(staging, ignore_errors=True)
        write_json(directory / "dataset.json", manifest)
    return report


def _stage_split(
    store: Store,
    suite: Suite,
    subset: str,
    split: str,
    seeds: list[int],
    report: BuildReport,
    progress: Callable[[str], None] | None,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], Path] | None:
    """Check one split and stage its new samples (see :func:`build`)."""
    directory = store.split_dir(suite.name, subset, split)
    description = {
        "suite": suite.name,
        "subset": subset,
        **suite.subsets[subset].describe(),
    }
    description = json.loads(json.dumps(description))  # as the manifest stores it
    manifest = read_manifest(directory)
    if manifest is not None and manifest["description"] != description:
        raise SuiteChanged(
            f"{directory}: the subset's definition differs from its manifest. "
            "Bump the suite's version instead of rebuilding into it."
        )
    known = {} if manifest is None else {e["seed"]: e for e in manifest["samples"]}
    staging = directory / _STAGING
    shutil.rmtree(staging, ignore_errors=True)  # left over by an interrupted build
    new: list[dict[str, Any]] = []
    for seed in seeds:
        sample = suite.subsets[subset].generate(seed)
        payload = sample_payload(sample)
        digest = array_hash(payload)
        label = f"{suite.name}/{subset}/{split}/{sample_name(seed)}"
        if seed in known:
            if known[seed]["sha256"] != digest:
                raise SuiteChanged(
                    f"{label} regenerates differently from its manifest. The "
                    "generator or the subset changed: bump the suite's version "
                    "instead of rebuilding into it."
                )
            if file_hash(directory / known[seed]["file"]) != digest:
                raise SuiteChanged(f"{label} is damaged (its file's hash differs)")
            report.checked.setdefault((subset, split), []).append(seed)
            if progress:
                progress(f"checked {label}")
            continue
        staging.mkdir(parents=True, exist_ok=True)
        path = staging / sample_name(seed)
        np.savez_compressed(path, **payload)
        if file_hash(path) != digest:
            raise RuntimeError(f"{path}: the written file hashes differently")
        entry = {
            "seed": seed,
            "file": path.name,
            "sha256": digest,
            "bytes": path.stat().st_size,
        }
        new.append(entry)
        known[seed] = entry
        report.written.setdefault((subset, split), []).append(seed)
        if progress:
            progress(f"wrote   {label}")
    if not new:
        return None
    manifest = {
        "description": description,
        "samples": [known[seed] for seed in sorted(known)],
        "provenance": provenance(),
    }
    return directory, manifest, new, staging


def read_manifest(directory: Path) -> dict[str, Any] | None:
    """A split's manifest, or ``None`` if it has not been built."""
    path = directory / "dataset.json"
    return json.loads(path.read_text()) if path.exists() else None


# -- loading --------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedSample:
    """A sample loaded through its manifest."""

    suite: str
    subset: str
    split: str
    seed: int
    sha256: str
    path: Path
    sample: Sample


def load_split(
    store: Store,
    suite: Suite | str,
    subset: str,
    split: str,
    seeds: Iterable[int] | None = None,
) -> list[LoadedSample]:
    """Load samples through their manifest, checking every file's hash.

    Parameters
    ----------
    store : Store
        Where the suite was built.
    suite : Suite or str
        The suite (or its name).
    subset : str
        The subset.
    split : str
        A split or a view (``smoke``).
    seeds : iterable of int, optional
        Restrict the seeds; they must belong to the split or view.

    Returns
    -------
    list of LoadedSample
        In seed order.

    Raises
    ------
    FileNotFoundError
        If the split, or one of the seeds, has not been built.
    SuiteChanged
        If a file does not match its manifest's hash.
    """
    suite = get_suite(suite) if isinstance(suite, str) else suite
    stored, view_seeds = suite.resolve(split)
    wanted = set(view_seeds) if seeds is None else set(seeds)
    outside = sorted(wanted - set(view_seeds))
    if outside:
        raise ValueError(f"seeds {outside} are not in {suite.name}'s {split!r}")
    directory = store.split_dir(suite.name, subset, stored)
    manifest = read_manifest(directory)
    if manifest is None:
        raise FileNotFoundError(f"{directory} has not been built")
    entries = {e["seed"]: e for e in manifest["samples"]}
    missing = sorted(wanted - set(entries))
    if missing:
        raise FileNotFoundError(f"{directory}: seeds {missing} have not been built")
    loaded = []
    for seed in sorted(wanted):
        entry = entries[seed]
        path = directory / entry["file"]
        if file_hash(path) != entry["sha256"]:
            raise SuiteChanged(f"{path} does not match its manifest's hash")
        loaded.append(
            LoadedSample(
                suite.name,
                subset,
                stored,
                seed,
                entry["sha256"],
                path,
                Sample.load(path),
            )
        )
    return loaded


def build_quick(
    store: Store, progress: Callable[[str], None] | None = None
) -> dict[str, BuildReport]:
    """Build the quick scope (plan, Section 4.3) into a store."""
    reports = {}
    for name, (subsets, split, seeds) in QUICK_SCOPE.items():
        reports[name] = build(
            store, get_suite(name), subsets, (split,), seeds, progress=progress
        )
    return reports
