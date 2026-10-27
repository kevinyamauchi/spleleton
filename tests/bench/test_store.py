import warnings

import pytest

from spleleton import bench
from spleleton.bench import store as store_module
from spleleton.bench.store import (
    DirtyTree,
    Store,
    check_tree,
    fit_key,
    graph_key,
    numerics,
    provenance,
    score_key,
    src_hash,
)
from spleleton.pipeline import GraphConfig, ScratchFitConfig


def test_roots_and_layout(tmp_path, monkeypatch):
    monkeypatch.delenv("SPLELETON_BENCH_DIR", raising=False)
    monkeypatch.delenv("SPLELETON_BENCH_QUICK_DIR", raising=False)
    full, quick = Store.default(), Store.default(quick=True)
    assert full.root.name == "bench"
    assert quick.root.name == "bench_quick"
    assert quick.quick and not full.quick
    monkeypatch.setenv("SPLELETON_BENCH_DIR", str(tmp_path / "b"))
    monkeypatch.setenv("SPLELETON_BENCH_QUICK_DIR", str(tmp_path / "q"))
    assert Store.default().root == tmp_path / "b"
    assert Store.default(quick=True).root == tmp_path / "q"
    store = Store(tmp_path)
    assert store.sample_path("s", "clean", "tune", 3) == (
        tmp_path / "datasets/s/clean/tune/sample_0003.npz"
    )
    assert store.run_dir("s", "phase5", "abcdef0123").name == "phase5__abcdef01"
    assert store.fit_path("k").parent == tmp_path / "cache" / "fit"


def test_keys_depend_on_spec_versions_and_numerics(monkeypatch):
    spec = ScratchFitConfig()
    env = numerics()
    key = fit_key("a" * 64, spec, env)
    assert key == fit_key("a" * 64, ScratchFitConfig(), dict(env))
    assert key != fit_key("b" * 64, spec, env)
    assert key != fit_key("a" * 64, ScratchFitConfig(n_rounds=2), env)
    assert key != fit_key("a" * 64, spec, {**env, "dtype": "float64"})
    graph = graph_key(key, GraphConfig())
    assert graph != graph_key(key, GraphConfig(join=1.5))
    score = score_key(graph)
    monkeypatch.setattr(store_module, "FIT_VERSION", bench.FIT_VERSION + 1)
    monkeypatch.setattr(store_module, "GRAPH_VERSION", bench.GRAPH_VERSION + 1)
    monkeypatch.setattr(store_module, "METRICS_VERSION", bench.METRICS_VERSION + 1)
    assert fit_key("a" * 64, spec, env) != key
    assert graph_key(key, GraphConfig()) != graph
    assert score_key(graph) != score


def test_provenance():
    record = provenance()
    assert record["src_hash"] == src_hash()
    assert len(record["src_hash"]) == 64
    assert record["stage_versions"] == {
        "fit": bench.FIT_VERSION,
        "graph": bench.GRAPH_VERSION,
        "metrics": bench.METRICS_VERSION,
    }
    assert "numpy" in record["versions"]


def test_dirty_tree_blocks_full_stages_and_warns_quick_ones(monkeypatch):
    dirty = {"commit": "abc", "dirty": True, "dirty_files": [" M src/x.py"]}
    monkeypatch.setattr(store_module, "git_state", lambda: dirty)
    with pytest.raises(DirtyTree, match=r"src/x\.py"):
        check_tree(full=True)
    with pytest.warns(UserWarning, match="uncommitted"):
        check_tree(full=False)
    clean = {"commit": "abc", "dirty": False, "dirty_files": []}
    monkeypatch.setattr(store_module, "git_state", lambda: clean)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert check_tree(full=True) == clean
