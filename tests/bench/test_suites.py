import dataclasses
import json

import numpy as np
import pytest

from spleleton.bench.cases import build_case, case_names
from spleleton.bench.store import Store
from spleleton.bench.suites import (
    QUICK_SCOPE,
    GeneratedSubset,
    Suite,
    SuiteChanged,
    build,
    build_quick,
    errors3d_v1,
    load_split,
    sample_hash,
)
from spleleton.synthetic.corruptions import BumpConfig, CorruptionConfig
from spleleton.synthetic.dataset import airway_config


def _small(**corruptions):
    config = airway_config(2, shape=(64, 64), corrupted=False)
    return dataclasses.replace(config, corruptions=CorruptionConfig(**corruptions))


def _suite(**subsets):
    subsets = subsets or {"clean": _small(), "bumps": _small(bumps=BumpConfig())}
    return Suite(
        "toy-v1",
        {name: GeneratedSubset(config) for name, config in subsets.items()},
        splits={"tune": (0, 1, 2)},
        views={"smoke": ("tune", (0, 1))},
    )


def _files(store: Store) -> dict[str, float]:
    return {
        str(path.relative_to(store.root)): path.stat().st_mtime_ns
        for path in sorted(store.root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "bench_quick", quick=True)


def test_build_writes_manifests_and_rebuilding_writes_nothing(store):
    suite = _suite()
    report = build(store, suite)
    assert report.n_written == 6
    manifest = json.loads(
        (store.split_dir("toy-v1", "bumps", "tune") / "dataset.json").read_text()
    )
    assert [e["seed"] for e in manifest["samples"]] == [0, 1, 2]
    assert manifest["description"]["subset"] == "bumps"
    assert "src_hash" in manifest["provenance"]
    before = _files(store)
    report = build(store, suite)
    assert (report.n_written, report.n_checked) == (0, 6)
    assert _files(store) == before


def test_new_seeds_are_added(store):
    build(store, _suite(), seeds=[0])
    report = build(store, _suite())
    assert report.written == {("clean", "tune"): [1, 2], ("bumps", "tune"): [1, 2]}
    assert report.n_checked == 2
    assert [s.seed for s in load_split(store, _suite(), "clean", "tune")] == [0, 1, 2]


def test_a_changed_generator_is_refused_and_writes_nothing(store):
    build(store, _suite(), seeds=[0, 1])
    before = _files(store)

    @dataclasses.dataclass(frozen=True)
    class Changed(GeneratedSubset):
        def generate(self, seed):
            sample = super().generate(seed)
            sample.mask[0, 0] = not sample.mask[0, 0]
            return sample

    changed = _suite()
    changed.subsets["bumps"] = Changed(changed.subsets["bumps"].config)
    # seed 2 of "clean" would be new: it must not be written either
    with pytest.raises(SuiteChanged, match="regenerates differently"):
        build(store, changed)
    assert _files(store) == before


def test_a_changed_subset_is_refused(store):
    build(store, _suite(), seeds=[0])
    before = _files(store)
    other = _suite(clean=_small(), bumps=_small(bumps=BumpConfig(count=7)))
    with pytest.raises(SuiteChanged, match="definition differs"):
        build(store, other, seeds=[0])
    assert _files(store) == before


def test_loading_checks_hashes_and_resolves_views(store):
    suite = _suite()
    build(store, suite)
    smoke = load_split(store, suite, "clean", "smoke")
    assert [s.seed for s in smoke] == [0, 1]
    assert all(s.split == "tune" for s in smoke)
    assert smoke[1].sha256 == sample_hash(suite.subsets["clean"].generate(1))
    np.testing.assert_array_equal(
        smoke[1].sample.mask, suite.subsets["clean"].generate(1).mask
    )
    # a damaged file
    path = smoke[0].path
    with np.load(path) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["mask"] = ~arrays["mask"]
    np.savez_compressed(path, **arrays)
    with pytest.raises(SuiteChanged, match="does not match"):
        load_split(store, suite, "clean", "tune", [0])
    with pytest.raises(SuiteChanged, match="damaged"):
        build(store, suite)
    with pytest.raises(ValueError, match="not in"):
        load_split(store, suite, "bumps", "smoke", [2])


def test_errors3d_v1_definition():
    suite = errors3d_v1()
    assert list(suite.subsets) == [
        "clean",
        "asym-clean",
        "bumps-h0.5",
        "bumps-h1.0",
        "bumps-h1.5",
        "noise-a0.5",
        "noise-a1.0",
        "all",
        "tortuous-clean",
        "contacts",
        "all-tortuous",
    ]
    assert suite.splits == {"tune": tuple(range(8)), "test": tuple(range(1000, 1020))}
    assert suite.resolve("smoke") == ("tune", (0, 1))
    configs = {name: subset.config for name, subset in suite.subsets.items()}
    for config in configs.values():
        assert config.shape == (128, 128, 128)
        assert config.corruptions.gaps is None
        if config.corruptions.contacts is not None:
            assert config.corruptions.contacts.min_gap == 2.0
    with_contacts = {n for n, c in configs.items() if c.corruptions.contacts}
    assert with_contacts == {"all", "contacts", "all-tortuous"}
    assert configs["asym-clean"].tree.side_branch_probability == 1.0
    for name in ("tortuous-clean", "contacts", "all-tortuous"):
        assert configs[name].tree == configs["tortuous-clean"].tree
        assert configs[name].tree.tortuosity == 0.1
    for name in ("clean", "bumps-h0.5", "noise-a1.0", "all"):
        assert configs[name].tree == configs["clean"].tree
    bumps = configs["bumps-h1.0"].corruptions.bumps
    assert (bumps.count, bumps.height, bumps.size) == (6, (1.0, 1.0), (1.0, 1.0))


def test_tortuous_subsets_share_their_tree():
    suite = errors3d_v1()
    clean = suite.subsets["tortuous-clean"].generate(0)
    contacts = suite.subsets["contacts"].generate(0)
    np.testing.assert_array_equal(clean.clean_mask, contacts.clean_mask)
    assert [r["type"] for r in contacts.corruptions] == ["contact"]
    assert contacts.corruptions[0]["outside"] >= 2.0


def test_quick_scope_builds(store):
    reports = build_quick(store)
    assert reports["errors3d-v1"].n_written == 4
    assert reports["proto-cases"].n_written == len(case_names())
    subsets, split, seeds = QUICK_SCOPE["errors3d-v1"]
    for subset in subsets:
        (loaded,) = load_split(store, "errors3d-v1", subset, split, seeds)
        assert loaded.seed == seeds[0]
    with pytest.raises(FileNotFoundError, match="have not been built"):
        load_split(store, "errors3d-v1", "clean", split)
    (contact,) = load_split(store, "proto-cases", "contact", "cases")
    np.testing.assert_array_equal(
        contact.sample.mask, build_case("contact").sample.mask
    )
    assert build_quick(store)["errors3d-v1"].n_written == 0
