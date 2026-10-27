import dataclasses
import itertools

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pd = pytest.importorskip("pandas")

import jax.numpy as jnp  # noqa: E402

from spleleton._serialise import from_dict, spec_hash, to_dict  # noqa: E402
from spleleton.bench.cases import build_case  # noqa: E402
from spleleton.bench.events import extract_events  # noqa: E402
from spleleton.bench.probe import (  # noqa: E402
    FAMILIES,
    ProbeSpec,
    load_probe,
    loss_correct_rates,
    probe_spec,
    probe_volume,
    run_probe,
    screen,
    weight_map,
)
from spleleton.bench.probe._analysis import cluster_bootstrap, rate_floor  # noqa: E402
from spleleton.bench.probe._crop import background, halo_region  # noqa: E402
from spleleton.bench.probe._probe import (  # noqa: E402
    check_full,
    classify,
    compare_padding,
    probe_setup,
)
from spleleton.bench.probe._setup import (  # noqa: E402
    config,
    make_setup,
    padded,
    select_events,
    subdivide,
    tangent,
)
from spleleton.bench.probe._terms import VOXEL_TERMS  # noqa: E402
from spleleton.bench.store import Store  # noqa: E402
from spleleton.bench.suites import build, proto_cases  # noqa: E402
from spleleton.bench.timing import compile_totals  # noqa: E402
from spleleton.bench.variants import get_variant  # noqa: E402
from spleleton.fit import LossTerm  # noqa: E402
from spleleton.fit._fit_jax import (  # noqa: E402
    LOSSES_JAX,
    LossContext,
    soft_skeleton_jax,
)

PHASE5 = get_variant("phase5").variant
CENTRELINE = ["target.name=edt_ridge", "fit.fit.render.mode=centreline"]


def _with_losses(*terms):
    fit = dataclasses.replace(
        PHASE5.fit, fit=dataclasses.replace(PHASE5.fit.fit, losses=terms)
    )
    return dataclasses.replace(PHASE5, fit=fit)


# -- the spec --------------------------------------------------------------------------


def test_the_spec_of_phase5():
    spec = probe_spec(PHASE5)
    assert spec.families == FAMILIES
    assert spec.terms == ("bce", "dice", "sparsity", "connection")
    assert spec.weights == (1.0, 1.0, 0.5, 1.0)
    assert spec.render.union == "max"
    assert from_dict(ProbeSpec, to_dict(spec)) == spec
    assert spec_hash(probe_spec(get_variant("phase5"))) == spec_hash(spec)
    slim = probe_spec(PHASE5, slim=True)
    assert spec_hash(slim) != spec_hash(spec)
    assert slim.family_eps("bump_spur") == (0.0, 0.25, 0.5, 1.0, 1.5)
    assert slim.family_eps("junction_detach") == spec.eps
    assert slim.new_slot_opacities == (1.0,)
    assert spec.new_slot_opacities == (1.0, 0.5)
    # the slim profile: 0, 0.5, 1 and the largest uncollapsed ε
    assert slim.profile_eps(spec.eps) == (0.0, 0.5, 1.0, 3.0)
    assert slim.profile_eps(spec.eps, collapsed=(3.0,)) == (0.0, 0.5, 1.0, 2.0)
    assert spec.profile_eps(spec.eps, collapsed=(3.0,)) == spec.eps[:-1]


def test_centreline_targets_skip_radius_scale():
    variant = get_variant(
        "phase5", ["target.name=edt_ridge", "fit.fit.render.mode=centreline"]
    )
    spec = probe_spec(variant)
    assert "radius_scale" not in spec.families
    assert len(spec.families) == len(FAMILIES) - 1


def test_the_probe_needs_a_scratch_fit_and_the_max_union():
    with pytest.raises(ValueError, match="scratch_fit"):
        probe_spec(get_variant("phase4"))
    soft = get_variant("phase5", ["fit.fit.render.union=softmax"])
    with pytest.raises(ValueError, match="max union"):
        probe_spec(soft)
    with pytest.raises(ValueError, match="unknown families"):
        probe_spec(PHASE5, families=("gap_retract",))


# -- the terms on crops ----------------------------------------------------------------


@pytest.mark.parametrize(
    "term",
    [
        LossTerm("bce"),
        LossTerm("dice"),
        LossTerm("weighted_l2", params=(("pos_weight", 10.0),)),
        LossTerm("focal", params=(("gamma", 2.0),)),
        LossTerm("tversky", params=(("alpha", 0.3), ("beta", 0.7))),
    ],
)
def test_per_voxel_terms_sum_to_the_library_terms(term):
    rng = np.random.default_rng(0)
    with jax.enable_x64(True):
        occupancy = jnp.asarray(rng.uniform(size=(12, 10, 8)))
        target = jnp.asarray((rng.uniform(size=(12, 10, 8)) > 0.6).astype(float))
        context = LossContext(occupancy, target, None, None)
        expected = float(LOSSES_JAX[term.name](context, **dict(term.params)))
        voxel = VOXEL_TERMS[term.name]
        params = dict(term.params)
        stats = voxel.stats(occupancy, target, **params)
        # split the volume in two parts, as a crop and the rest
        inside = np.zeros(occupancy.shape, bool)
        inside[3:9, 2:7, :5] = True
        sums = tuple(
            float(jnp.where(inside, s, 0).sum() + jnp.where(inside, 0, s).sum())
            for s in stats
        )
        got = float(voxel.finish(sums, occupancy.size, **params))
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-15)


def test_the_soft_skeleton_is_local_within_its_halo():
    # the halo of k iterations is k + 2 voxels: changing the occupancy in a
    # window changes the skeleton only within that distance of it
    k = 3
    h = k + 2
    rng = np.random.default_rng(1)
    with jax.enable_x64(True):
        image = rng.uniform(size=(40, 40, 40))
        changed = image.copy()
        changed[15:22, 17:21, 16:20] = rng.uniform(size=(7, 4, 4))
        a = np.asarray(soft_skeleton_jax(jnp.asarray(image), k))
        b = np.asarray(soft_skeleton_jax(jnp.asarray(changed), k))
    near = np.zeros(image.shape, bool)
    near[15 - h : 22 + h, 17 - h : 21 + h, 16 - h : 20 + h] = True
    assert np.array_equal(a[~near], b[~near])
    assert not np.array_equal(a[near], b[near])
    # and a region 2h around the window computes it exactly where it is kept
    origin, region, keep_lo, keep_hi = halo_region(
        np.array([15, 17, 16]), (7, 4, 4), image.shape, h
    )
    window = tuple(slice(o, o + s) for o, s in zip(origin, region, strict=True))
    with jax.enable_x64(True):
        part = np.asarray(soft_skeleton_jax(jnp.asarray(changed[window]), k))
    keep = tuple(slice(lo, hi) for lo, hi in zip(keep_lo, keep_hi, strict=True))
    assert np.array_equal(part[keep], b[window][keep])


# -- events, the reference and set-ups -------------------------------------------------


def test_events_are_chosen_deterministically():
    sample = build_case("y-thin").sample
    events = extract_events(sample)
    spec = probe_spec(PHASE5)
    chosen, skipped = select_events(sample, events, spec, seed=0)
    again, _ = select_events(sample, events, spec, seed=0)
    assert [(e.event_id, e.family) for e in chosen] == [
        (e.event_id, e.family) for e in again
    ]
    junctions = [e.event_id for e in chosen if e.family == "junction_detach"]
    assert junctions == ["junction#1-0", "junction#1-1", "junction#1-2"]
    slim, _ = select_events(sample, events, probe_spec(PHASE5, slim=True), seed=0)
    assert [e.event_id for e in slim if e.family == "junction_detach"] == [
        "junction#1-2"
    ]
    # the three control families share their mid-edge points
    controls = {
        f: [e.event_id for e in chosen if e.family == f]
        for f in ("edge_shift", "duplicate", "radius_scale")
    }
    assert controls["edge_shift"] == controls["duplicate"] == controls["radius_scale"]
    assert all(s["reason"] for s in skipped)


def test_bumps_and_protrusions_are_probed():
    sample = build_case("tube-bump").sample
    spec = probe_spec(PHASE5, families=("bump_spur", "bump_shift"))
    chosen, _ = select_events(sample, extract_events(sample), spec, seed=0)
    assert {(e.event_id, e.family) for e in chosen} == {
        ("bump#0", "bump_spur"),
        ("bump#0", "bump_shift"),
    }
    spur = next(e for e in chosen if e.family == "bump_spur").spur
    assert spur["length"] == pytest.approx(8.0, abs=0.1)  # r + height
    assert spur["radius"] == pytest.approx(2.0)
    sample = build_case("contact").sample
    chosen, _ = select_events(sample, extract_events(sample), spec, seed=0)
    assert {e.event_id for e in chosen} == {"protrusion#0"}


def test_whole_intervals_stay_single_pieces():
    graph = build_case("tube").sample.graph
    _, pieces = subdivide(graph, {0: [(10.0, 18.0), (18.0, 30.0)]})
    spans = [(p["lo"], p["hi"]) for p in pieces]
    assert (10.0, 18.0) in spans and (18.0, 30.0) in spans
    assert spans[0][0] == 0.0 and spans[-1][1] == pytest.approx(52.0)
    assert all(np.isclose(a[1], b[0]) for a, b in itertools.pairwise(spans))


class _FakeReference:
    def __init__(self, sample, arrays, pieces):
        from spleleton.primitives._primitives_jax import JaxLineSegments

        with jax.enable_x64(True):
            segments = JaxLineSegments.from_arrays(arrays, dtype=jnp.float64)
            self.raw = {
                k: np.asarray(getattr(segments, k))
                for k in ("start", "end", "log_radius", "opacity_logit")
            }
        self.pieces = pieces
        self.sample = sample


def _setups(name, family, spec=None):
    sample = build_case(name).sample
    spec = spec or probe_spec(PHASE5, families=(family,))
    chosen, _ = select_events(sample, extract_events(sample), spec, seed=0)
    whole = {}
    for event in chosen:
        whole.setdefault(event.edge, []).extend(event.intervals)
    arrays, pieces = subdivide(sample.graph, whole)
    reference = _FakeReference(sample, arrays, pieces)
    return reference, chosen, spec


def test_the_error_paths():
    reference, chosen, spec = _setups("y-thin", "junction_detach")
    thin = next(e for e in chosen if e.event_id == "junction#1-2")
    setup = make_setup(reference, thin, spec, None)
    arrays = padded(setup, 16)
    ((slot, field),) = setup.moved
    at_zero, at_one = config(arrays, setup, 0.0), config(arrays, setup, 1.0)
    moved = np.linalg.norm(at_one[field][slot] - at_zero[field][slot])
    assert moved == pytest.approx(1.8)  # ε r_child
    assert setup.frozen["start"][slot].all()
    assert not setup.frozen["log_radius"][slot].any()
    # the piece is 4 r_child long, so it collapses only beyond ε = 3.7
    assert setup.collapsed == []
    assert arrays.live.sum() == setup.n_exact and len(arrays.live) == 16

    # a shifted piece keeps its radius too
    reference, chosen, spec = _setups("y-thin", "edge_shift")
    setup = make_setup(reference, chosen[0], spec, None)
    for slot, _ in setup.moved:
        assert setup.frozen["log_radius"][slot].all()
        assert setup.frozen["start"][slot].all() and setup.frozen["end"][slot].all()

    reference, chosen, spec = _setups("y-thin", "radius_scale")
    setup = make_setup(reference, chosen[0], spec, None)
    arrays = padded(setup, setup.n_exact)
    j = setup.moved[0][0]
    scaled = np.exp(config(arrays, setup, 1.0)["log_radius"][j])
    assert scaled == pytest.approx(2 * np.exp(arrays.base["log_radius"][j]))
    assert tangent(arrays, 1.0)["log_radius"][j] == pytest.approx([0.5, 0.5])
    assert setup.frozen["log_radius"][j].all()

    reference, chosen, spec = _setups("tube-bump", "bump_spur")
    setup = make_setup(reference, chosen[0], spec, 1.0)
    arrays = padded(setup, setup.n_exact)
    j = setup.new_slot
    assert j == setup.n_exact - 1
    without = config(arrays, setup, 0.0, new=False)
    assert not without["live"][j]
    at_one = config(arrays, setup, 1.0)
    assert np.linalg.norm(at_one["end"][j] - at_one["start"][j]) == pytest.approx(
        8.0, abs=0.1
    )


def test_classes():
    tol = 1e-6
    snr = np.full(4, 10.0)
    assert classify(np.array([0, 1e-3, 2e-3, 3e-3]), snr, tol) == "restoring"
    assert classify(np.array([0, 1e-3, -1e-3, 3e-3]), snr, tol) == "attracting"
    assert classify(np.array([0, 2e-3, 1e-3, 3e-3]), snr, tol) == "barrier"
    weak = np.array([10.0, 0.1, 0.1, 10.0])
    assert classify(np.array([0, 1e-3, 2e-3, 3e-3]), weak, tol) == "flat"
    assert classify(np.array([0, 1e-3]), snr[:2], tol) == "undetermined"


# -- analysis and plots ----------------------------------------------------------------


def _events_frame(rng, correct):
    rows = []
    for family in ("junction_detach", "bump_spur", "edge_shift"):
        for seed in range(6):
            for k in range(4):
                rows.append(
                    {
                        "subset": "s",
                        "seed": seed,
                        "event_id": f"e{k}",
                        "family": family,
                        "opacity": None,
                        "loss_correct": bool(rng.uniform() < correct[family]),
                    }
                )
    return pd.DataFrame(rows)


def test_rates_and_the_screen():
    rng = np.random.default_rng(0)
    baseline = _events_frame(
        rng, {"junction_detach": 0.2, "bump_spur": 0.5, "edge_shift": 0.9}
    )
    rates = loss_correct_rates(baseline)
    assert set(rates["family"]) == {"junction_detach", "bump_spur", "edge_shift"}
    assert (rates["low"] <= rates["rate"]).all() and (
        rates["rate"] <= rates["high"]
    ).all()
    assert (rates["n_volumes"] == 6).all()
    # a candidate correct everywhere passes; one wrong everywhere does not
    better = baseline.assign(loss_correct=True)
    table, passes = screen(better, baseline)
    assert passes and table["gain"].any()
    worse = baseline.assign(loss_correct=False)
    assert not screen(worse, baseline)[1]
    # a candidate equal to the baseline is no gain
    assert not screen(baseline, baseline)[1]
    mean, low, high = cluster_bootstrap(np.ones(10), np.arange(10) % 3)
    assert (mean, low, high) == (1.0, 1.0, 1.0)


def test_the_weight_map_is_linear_in_the_weights():
    margins = pd.DataFrame(
        [
            {"subset": "s", "seed": 0, "event_id": "a", "family": "f", "opacity": None,
             "eps": 1.0, "term": term, "profile": value, "collapsed": False}
            for term, value in (("bce", 2e-3), ("dice", -1e-3), ("total", 0.0))
        ]
    )  # fmt: skip
    weights = {"bce": 1.0, "dice": 1.0}
    table = weight_map(margins, weights, "bce", "dice", [1.0], [1.0, 3.0], 1e-6)
    # 2e-3 - 1e-3 > 0 at (1, 1); 2e-3 - 3e-3 < 0 at (1, 3)
    assert table.set_index("weight_b")["rate"].to_dict() == {1.0: 1.0, 3.0: 0.0}
    # several events and volumes: each counted once
    rows = [
        {"subset": "s", "seed": seed, "event_id": event, "family": family,
         "opacity": None, "eps": 1.0, "term": "bce", "profile": 1e-3,
         "collapsed": False}
        for seed in (0, 1) for event, family in (("a", "f"), ("b", "g"))
    ]  # fmt: skip
    table = weight_map(pd.DataFrame(rows), {"bce": 1.0}, "bce", "bce", [1.0], [1.0], 0)
    assert table.set_index("family")["n_events"].to_dict() == {"f": 2, "g": 2}
    assert (table["rate"] == 1.0).all()
    events = pd.DataFrame(
        {
            "family": ["f", "f"],
            "loss_correct": [True, True],
            "loss_correct_floor": [True, False],
        }
    )
    assert rate_floor(events) == {"f": 0.5}


def test_the_plots_are_written(tmp_path):
    pytest.importorskip("matplotlib")
    from spleleton.bench.plots import (
        plot_eps_profiles,
        plot_screening,
        plot_weight_map,
    )

    rng = np.random.default_rng(0)
    rows = []
    for family in ("junction_detach", "bump_spur"):
        for event in range(3):
            for eps in (0.0, 0.5, 1.0):
                for term in ("bce", "dice", "total"):
                    rows.append(
                        {"subset": "s", "seed": 0, "event_id": f"e{event}",
                         "family": family, "opacity": None, "eps": eps, "term": term,
                         "profile": rng.normal() * eps, "force_rigid": rng.normal(),
                         "collapsed": False}
                    )  # fmt: skip
    margins = pd.DataFrame(rows)
    path = plot_eps_profiles(margins, {"bce": 1.0, "dice": 1.0}, tmp_path / "p.png")
    assert path.stat().st_size > 0
    table = weight_map(
        margins,
        {"bce": 1.0, "dice": 1.0},
        "bce",
        "dice",
        [0.5, 1, 2],
        [0.5, 1, 2],
        1e-6,
    )
    assert plot_weight_map(
        table, (1.0, 1.0), ("bce", "dice"), tmp_path / "w.png"
    ).exists()
    families = ("junction_detach", "bump_spur", "edge_shift")
    rates = pd.concat(
        [
            loss_correct_rates(_events_frame(rng, dict.fromkeys(families, p))).assign(
                candidate=name
            )
            for name, p in (("phase5", 0.5), ("cldice", 0.7))
        ]
    )
    assert plot_screening(rates, "phase5", tmp_path / "s.png").exists()


# -- end to end (slow) -----------------------------------------------------------------


def _quick_spec(variant=PHASE5, **settings):
    return probe_spec(
        variant, eps=(0.0, 0.5, 1.0, 2.0), profile_iterations=10, **settings
    )


@pytest.fixture(scope="module")
def probes():
    """Every family on two cases: the bumps on ``tube-bump``, the rest on ``y-thin``."""
    out = []
    for case, families in (
        ("tube-bump", ("bump_spur", "bump_shift")),
        ("y-thin", ("junction_detach", "break_retract", "edge_shift", "duplicate",
                    "radius_scale")),
    ):  # fmt: skip
        spec = _quick_spec(families=families)
        volume = probe_volume(
            build_case(case).sample,
            spec,
            seed=0,
            floor=True,
            keep_results=True,
            reference_iterations=10,
            max_events=1,
        )
        out.append((spec, volume))
    return out


@pytest.mark.slow
def test_every_family_runs_and_crops_equal_the_full_volume(probes):
    families = {r["family"] for _, volume in probes for r in volume.events}
    assert families == set(FAMILIES)
    with jax.enable_x64(True):
        for spec, volume in probes:
            for setup, result in volume.results:
                errors = check_full(volume.reference, setup, result, spec)
                assert max(errors.values()) < 1e-12, (setup.family, errors)
    for _, volume in probes:
        for row in volume.events:
            assert row["fd_agree"] >= 0.95, row
            assert row["floor"] is not None


@pytest.mark.slow
def test_padding_changes_nothing(probes):
    spec, volume = probes[1]
    with jax.enable_x64(True):
        for setup, result in volume.results[:2]:
            back = background(volume.reference, setup)
            exact = probe_setup(volume.reference, setup, back, spec, bucketed=False)
            assert max(compare_padding(result, exact, spec).values()) < 1e-12


@pytest.mark.slow
def test_a_second_batch_of_seen_shapes_compiles_nothing(probes):
    spec, volume = probes[0]
    setup, _ = volume.results[0]
    with jax.enable_x64(True):
        back = background(volume.reference, setup)
        probe_setup(volume.reference, setup, back, spec, floor=True)
        before = compile_totals()[1]
        result = probe_setup(volume.reference, setup, back, spec, floor=True)
    assert compile_totals()[1] == before
    assert result["timing"]["programs"] == 0


@pytest.mark.slow
@pytest.mark.parametrize(
    "variant",
    [
        _with_losses(
            LossTerm("bce"), LossTerm("dice", 0.5), LossTerm("sparsity", 0.5),
            LossTerm("cldice", 0.5, (("k", 3),)),
        ),
        _with_losses(
            LossTerm("focal", 1.0, (("gamma", 2.0),)),
            LossTerm("tversky", 1.0, (("alpha", 0.3), ("beta", 0.7))),
            LossTerm("weighted_l2", 0.2, (("pos_weight", 10.0),)),
            LossTerm("sparsity", 0.5),
        ),
        get_variant("phase5", CENTRELINE).variant,
    ],
    ids=["cldice", "focal-tversky-l2", "edt_ridge"],
)  # fmt: skip
def test_candidate_losses_are_exact_on_crops(variant):
    spec = _quick_spec(variant, families=("bump_shift",))
    spec = dataclasses.replace(spec, eps=(0.0, 1.0), profile_iterations=3)
    out = probe_volume(
        build_case("tube-bump").sample, spec, seed=0, keep_results=True,
        reference_iterations=3,
    )  # fmt: skip
    ((setup, result),) = out.results
    with jax.enable_x64(True):
        errors = check_full(out.reference, setup, result, spec)
    assert max(errors.values()) < 1e-12, errors


@pytest.mark.slow
def test_run_probe_writes_tables_and_resumes(tmp_path):
    pytest.importorskip("pyarrow")
    store = Store(tmp_path / "bench_quick", quick=True)
    build(store, proto_cases(), ["tube-bump"], ["cases"])
    spec = dataclasses.replace(
        _quick_spec(families=("bump_shift",)),
        reference_iterations=5,
        profile_iterations=3,
    )
    lines = []
    directory = run_probe(
        store,
        spec,
        "phase5",
        "proto-cases",
        ["tube-bump"],
        "cases",
        progress=lines.append,
    )
    results = load_probe(directory)
    assert list(results.events["event_id"]) == ["bump#0"]
    assert set(results.margins["term"]) == {*spec.terms, "total"}
    assert results.meta["volumes"]["tube-bump_0000"]["status"] == "done"
    assert (directory / "reference" / "tube-bump_0000.npz").exists()
    lines.clear()
    run_probe(
        store,
        spec,
        "phase5",
        "proto-cases",
        ["tube-bump"],
        "cases",
        progress=lines.append,
    )
    assert lines == ["tube-bump_0000: done, skipped"]
    # the napari probe mode's layers
    from spleleton.bench.napari import probe_event_view

    view = probe_event_view(directory, store, "tube-bump", 0, "bump#0", "bump_shift")
    layers = view.layers(view.eps[-1])
    names = [kwargs["name"] for _, kwargs, _ in layers]
    assert "force: total" in names and "attribution (dL/d occupancy)" in names
    event = next(
        data for data, kwargs, _ in layers if kwargs["name"].startswith("event")
    )
    assert event.shape[1:] == (2, 3)
