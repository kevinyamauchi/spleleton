import pytest

from spleleton.bench import stages
from spleleton.bench.store import DirtyTree, Store


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "bench_quick", quick=True)


def _tree(monkeypatch, dirty: bool) -> None:
    import spleleton.bench.store as store_module

    monkeypatch.setattr(
        store_module,
        "git_state",
        lambda: {
            "commit": "abc",
            "dirty": dirty,
            "dirty_files": [" M src/x.py"] if dirty else [],
        },
    )


def test_a_full_stage_refuses_a_dirty_tree_and_a_quick_one_warns(store, monkeypatch):
    _tree(monkeypatch, dirty=True)
    with pytest.raises(DirtyTree):
        stages.run_stage(store, "data", quick=False, progress=None)
    assert stages.read_record(store, "data") is None
    monkeypatch.setitem(
        stages.STAGES,
        "data",
        stages.Stage("data", "S1", (), "", "", lambda q: [], recipe=lambda ctx: None),
    )
    with pytest.warns(UserWarning, match="uncommitted"):
        record = stages.run_stage(store, "data", quick=True, progress=None)
    assert record["status"] == "done" and record["dirty"]


def test_prerequisites_and_unbuilt_stages(store, monkeypatch):
    _tree(monkeypatch, dirty=False)
    with pytest.raises(stages.PrerequisiteMissing, match="needs data"):
        stages.run_stage(store, "baselines", quick=True, progress=None)
    # every stage is built
    assert all(stage.recipe is not None for stage in stages.STAGES.values())
    with pytest.raises(stages.PrerequisiteMissing, match="needs baselines"):
        stages.run_stage(store, "sensitivity", quick=True, progress=None)
    with pytest.raises(stages.PrerequisiteMissing, match="needs sensitivity"):
        stages.run_stage(store, "validate", quick=True, progress=None)
    with pytest.raises(stages.PrerequisiteMissing, match="needs baselines"):
        stages.run_stage(store, "diagnose", quick=True, progress=None)
    with pytest.raises(stages.PrerequisiteMissing, match="needs baselines"):
        stages.run_stage(store, "floor", quick=True, progress=None)
    assert stages.resolve(["all"]) == list(stages.ORDER)
    assert stages.resolve(["S2", "data"]) == ["data", "screen"]
    with pytest.raises(KeyError):
        stages.resolve(["nope"])
    lines = stages.dry_run("baselines", quick=False)
    assert any("bench run phase5 phase4" in line for line in lines)
    assert not any(
        "not available" in line
        for name in stages.ORDER
        for line in stages.dry_run(name, quick=True)
    )
    floor = stages.dry_run("floor", quick=False)
    assert any("--replicates 3" in line for line in floor)
    assert not any("not available" in line for line in floor)


def test_the_quick_data_stage_and_status(store, monkeypatch):
    _tree(monkeypatch, dirty=False)
    report = stages.status(store)
    assert report["next"] == "bench stage data"
    record = stages.run_stage(store, "data", quick=True, progress=None)
    assert record["status"] == "done"
    assert all(record["checks"].values()) and len(record["checks"]) == 2
    assert (store.stages_dir / "data.log").read_text().count("check PASS") == 2
    # a second run rebuilds nothing and still passes
    assert (
        stages.run_stage(store, "data", quick=True, progress=None)["status"] == "done"
    )
    report = stages.status(store)
    assert report["stages"][0]["status"] == "done"
    assert report["next"] == "bench stage baselines"
    assert report["warnings"] == []


def test_a_failed_check_fails_the_stage(store, monkeypatch):
    _tree(monkeypatch, dirty=False)

    def recipe(ctx):
        ctx.check("always fails", False, "detail")

    monkeypatch.setitem(
        stages.STAGES,
        "data",
        stages.Stage("data", "S1", (), "", "", lambda q: [], recipe=recipe),
    )
    record = stages.run_stage(store, "data", quick=True, progress=None)
    assert record["status"] == "failed"
    assert record["details"] == {"always fails": "detail"}

    def broken(ctx):
        raise RuntimeError("boom")

    monkeypatch.setitem(
        stages.STAGES,
        "data",
        stages.Stage("data", "S1", (), "", "", lambda q: [], recipe=broken),
    )
    with pytest.raises(RuntimeError, match="boom"):
        stages.run_stage(store, "data", quick=True, progress=None)
    record = stages.read_record(store, "data")
    assert record["status"] == "failed" and "boom" in record["error"]
