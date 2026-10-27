"""The JAX pipelines' fit/graph split and the fit's final state."""

import numpy as np
import pytest

pytest.importorskip("jax")

from spleleton.assemble import assemble_graph
from spleleton.fit import FitConfig, fit_jax, loss_terms_jax
from spleleton.graph import SkeletonGraph
from spleleton.pipeline import (
    CleanConfig,
    GraphConfig,
    ScratchFitConfig,
    SkeletonFitConfig,
    graph_methods,
    graph_stage,
    register_graph_method,
    scratch_fit_jax,
    scratch_fit_stage_jax,
    skeleton_fit_jax,
    skeleton_fit_stage_jax,
)
from spleleton.primitives import JaxLineSegments, SegmentArrays
from spleleton.priors import JaxConnectionLoss
from spleleton.synthetic.dataset import airway_config, generate_sample


def _graphs_equal(a: SkeletonGraph, b: SkeletonGraph) -> bool:
    da, db = a.to_dict(), b.to_dict()
    return da.keys() == db.keys() and all(np.array_equal(da[k], db[k]) for k in da)


@pytest.fixture(scope="module")
def sample():
    return generate_sample(airway_config(2, shape=(64, 64)), 3)


def test_graph_methods_are_registered():
    assert {"layout", "assemble"} <= set(graph_methods())
    with pytest.raises(ValueError, match="already registered"):
        register_graph_method("assemble")(lambda *args: None)


def test_graph_stage_assembles_and_cleans():
    segments = SegmentArrays(
        np.array([[0.0, 20.0], [10.0, 20.0], [20.0, 20.0]]),
        np.array([[10.0, 20.0], [20.0, 20.0], [21.0, 26.0]]),
        radius=2.0,
        opacity=0.9,
    )
    mask = np.zeros((40, 40), dtype=bool)
    config = GraphConfig(params=(("contained", 0.25),))
    raw, clean = graph_stage(segments, None, mask, None, config)
    assert _graphs_equal(raw, assemble_graph(segments, 0.1, contained=0.25, join=1.0))
    assert _graphs_equal(clean, config.clean.apply(raw))
    off = GraphConfig(clean=CleanConfig(enabled=False))
    raw, clean = graph_stage(segments, None, mask, None, off)
    assert clean is raw


def test_graph_stage_errors():
    segments = SegmentArrays(np.zeros((1, 2)), np.ones((1, 2)))
    mask = np.zeros((4, 4), dtype=bool)
    with pytest.raises(ValueError, match="unknown graph method"):
        graph_stage(segments, None, mask, None, GraphConfig(method="nope"))
    with pytest.raises(ValueError, match="needs the fit's layout"):
        graph_stage(segments, None, mask, None, GraphConfig(method="layout"))


def test_scratch_fit_is_its_fit_stage_then_its_graph_stage(sample):
    config = ScratchFitConfig(fit=FitConfig(n_iterations=15), n_rounds=2)
    result = scratch_fit_jax(sample.mask, config=config)
    fit = scratch_fit_stage_jax(sample.mask, config=config)
    raw, clean = graph_stage(fit.segments, None, sample.mask, None, GraphConfig())
    assert _graphs_equal(result.graph, clean)
    assert _graphs_equal(result.raw_graph, raw)
    assert fit.history == result.history
    assert len(fit.history) == 30
    assert [r["n_iterations"] for r in fit.rounds] == [15, 15]
    first, second = fit.rounds
    assert first["added"] == first["n_segments"]
    assert second["n_segments"] == first["n_segments"] + second["added"]
    assert second["n_segments"] == len(fit.segments)


def test_skeleton_fit_is_its_fit_stage_then_its_graph_stage(sample):
    config = SkeletonFitConfig(fit=FitConfig(n_iterations=15))
    result = skeleton_fit_jax(sample.mask, config=config)
    fit = skeleton_fit_stage_jax(sample.mask, config=config)
    graph = GraphConfig(method="layout")
    raw, clean = graph_stage(fit.segments, fit.layout, sample.mask, None, graph)
    assert _graphs_equal(result.graph, clean)
    assert _graphs_equal(result.raw_graph, raw)
    assert len(fit.rounds) == 1
    assert fit.rounds[0]["n_segments"] == len(fit.segments)


def test_final_state_reproduces_the_last_losses_bit_for_bit(sample):
    seeds = scratch_fit_stage_jax(
        sample.mask, config=ScratchFitConfig(fit=FitConfig(n_iterations=0), n_rounds=1)
    ).segments
    config = FitConfig(n_iterations=12, temperature=(1.0, 0.5))
    target = sample.mask.astype(np.float32)
    connection = JaxConnectionLoss(sample.mask.size, 1.0, 1.5)
    segments, history, state = fit_jax(
        seeds, target, config=config, regularisers=[connection], return_state=True
    )
    # the same run without the state gives the same result
    again, history_again = fit_jax(
        seeds, target, config=config, regularisers=[connection]
    )
    assert history_again == history
    np.testing.assert_array_equal(np.asarray(again.start), np.asarray(segments.start))

    terms, overflow = loss_terms_jax(
        state.pre_last, target, state, config=config, regularisers=[connection]
    )
    assert not overflow
    assert terms == history[-1]
    final, _ = loss_terms_jax(
        segments, target, state, config=config, regularisers=[connection]
    )
    assert set(final) == set(history[-1])
    assert np.isfinite(final["loss"])
    assert final != history[-1]
    assert state.render_config.temperature == pytest.approx(0.5)
    assert len(state.prior_states) == 1


def test_final_state_of_an_empty_fit(sample):
    segments = JaxLineSegments.from_arrays(
        SegmentArrays(np.array([[10.0, 10.0]]), np.array([[20.0, 20.0]]), radius=2.0)
    )
    _, history, state = fit_jax(
        segments,
        sample.mask.astype(np.float32),
        config=FitConfig(n_iterations=0),
        return_state=True,
    )
    assert history == []
    assert state.pre_last is None
