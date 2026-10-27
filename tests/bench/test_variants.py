import dataclasses
import json

import pytest

from spleleton.baseline import skeletonize_to_graph
from spleleton.bench.cases import build_case
from spleleton.bench.store import fit_key, numerics
from spleleton.bench.variants import (
    BASELINES,
    NamedVariant,
    SweepTag,
    Variant,
    apply_overrides,
    get_variant,
    override,
    parse_override,
    sweep,
    variant_names,
)
from spleleton.fit import LossTerm
from spleleton.pipeline import (
    CleanConfig,
    GraphConfig,
    ScratchFitConfig,
    SkeletonFitConfig,
    graph_stage,
)
from spleleton.postprocess import clean_graph
from spleleton.targets import TargetConfig


def test_the_baselines_are_registered_and_resolved():
    assert set(BASELINES) <= set(variant_names())
    phase5 = get_variant("phase5")
    assert phase5.name == "phase5"
    assert phase5.variant.fit == ScratchFitConfig()
    assert phase5.variant.graph == GraphConfig(method="assemble")
    phase4 = get_variant("phase4").variant
    assert phase4.fit == SkeletonFitConfig(free_end_weight=3.0)
    assert phase4.graph.method == "layout"
    plain = get_variant("skeletonize").variant
    cleaned = get_variant("skeletonize+clean").variant
    assert plain.fit is None and cleaned.fit is None
    assert not plain.graph.clean.enabled
    assert cleaned.graph.clean == CleanConfig()


def test_identity_is_the_resolved_spec_not_the_name():
    # phase5 is the scratch fit with every default
    assert Variant("scratch_fit").spec_hash() == get_variant("phase5").spec_hash
    assert Variant("scratch_fit").spec_hash() != get_variant("phase4").spec_hash
    # the order of params changes nothing
    a = Variant(
        "scratch_fit", graph=GraphConfig(params=(("contained", 0.5), ("parallel", 0.7)))
    )
    b = Variant(
        "scratch_fit", graph=GraphConfig(params=(("parallel", 0.7), ("contained", 0.5)))
    )
    assert a.spec_hash() == b.spec_hash()
    assert a.resolved().graph.params == (("contained", 0.5), ("parallel", 0.7))


@pytest.mark.parametrize("name", BASELINES)
def test_variants_round_trip_through_json(name):
    variant = get_variant(name).variant
    data = json.loads(json.dumps(variant.to_dict()))
    assert Variant.from_dict(data) == variant
    assert Variant.from_dict(data).spec_hash() == variant.spec_hash()


def test_a_custom_variant_round_trips():
    variant = get_variant(
        "phase5",
        [
            "target.name=edt_ridge",
            "target.params.sigma=1.5",
            "fit.fit.render.mode=centreline",
            "fit.fit.losses.dice.weight=0.5",
        ],
    ).variant
    data = json.loads(json.dumps(variant.to_dict()))
    assert Variant.from_dict(data) == variant
    phase4 = Variant.from_dict(get_variant("phase4").variant.to_dict())
    assert isinstance(phase4.fit, SkeletonFitConfig)


def test_the_fit_spec_leaves_the_graph_out():
    phase5 = get_variant("phase5").variant
    swept = override(phase5, "graph.join", 1.5)
    assert swept.spec_hash() != phase5.spec_hash()
    assert swept.fit_spec() == phase5.fit_spec()
    keys = numerics()

    def key(variant, replicate=0):
        return fit_key("sample", variant.fit_spec(replicate), keys)

    assert key(swept) == key(phase5)
    # the jitter, the replicate and the pipeline are in the fit's key
    assert key(override(phase5, "init_jitter", 1e-3)) != key(phase5)
    assert key(phase5, replicate=1) != key(phase5)
    assert key(get_variant("phase4").variant) != key(phase5)
    assert get_variant("skeletonize").variant.fit_spec() is None


def test_overrides_follow_dotted_paths():
    phase5 = get_variant("phase5").variant
    changed = override(phase5, "fit.connection_reach", 2)  # an int becomes a float
    assert changed.fit.connection_reach == 2.0
    assert (
        changed.spec_hash() == override(phase5, "fit.connection_reach", 2.0).spec_hash()
    )
    terms = override(phase5, "fit.fit.losses.dice.weight", 0.5).fit.fit.losses
    assert LossTerm("dice", 0.5) in terms
    assert LossTerm("bce", 1.0) in terms
    params = override(phase5, "graph.params.contained", 0.6).graph.params
    assert params == (("contained", 0.6),)
    params = apply_overrides(
        phase5, [("graph.params.parallel", 0.8), ("graph.params.contained", 0.6)]
    ).graph.params
    assert params == (("contained", 0.6), ("parallel", 0.8))
    # the base variant is unchanged
    assert phase5 == get_variant("phase5").variant


@pytest.mark.parametrize(
    ("path", "value", "error"),
    [
        ("fit.no_such_field", 1.0, KeyError),
        ("fit.fit.losses.focal.weight", 1.0, KeyError),
        ("graph.params.contained.x", 1.0, KeyError),
        ("fit..fit", 1.0, KeyError),
        ("fit.connection_reach", True, ValueError),
        ("init_jitter", -1.0, ValueError),
        ("graph.method", "layout", ValueError),  # Phase 5 has no layout
        ("target.name", "edt_ridge", ValueError),  # needs the centreline render
    ],
)
def test_bad_overrides_are_refused(path, value, error):
    with pytest.raises(error):
        override(get_variant("phase5").variant, path, value)


def test_overrides_that_go_together_are_applied_together():
    named = get_variant(
        "phase5", ["target.name=soft_skeleton", "fit.fit.render.mode=centreline"]
    )
    assert named.name == (
        "phase5[target.name=soft_skeleton,fit.fit.render.mode=centreline]"
    )
    assert named.variant.target == TargetConfig("soft_skeleton")


def test_skeletonize_fits_nothing():
    plain = get_variant("skeletonize").variant
    with pytest.raises(KeyError):
        override(plain, "fit.n_rounds", 2)
    with pytest.raises(ValueError, match="fits nothing"):
        Variant("skeletonize", fit=ScratchFitConfig()).resolved()
    with pytest.raises(ValueError, match="cannot use"):
        Variant("skeletonize", graph=GraphConfig(method="assemble")).resolved()
    with pytest.raises(ValueError, match="needs a ScratchFitConfig"):
        Variant("scratch_fit", fit=SkeletonFitConfig()).resolved()
    with pytest.raises(ValueError, match="unknown pipeline"):
        Variant("phase6").resolved()


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("fit.connection_reach=2.0", ("fit.connection_reach", 2.0)),
        ("fit.n_rounds = 2", ("fit.n_rounds", 2)),
        ("graph.clean.enabled=false", ("graph.clean.enabled", False)),
        ("fit.fit.temperature=None", ("fit.fit.temperature", None)),
        ("fit.fit.temperature=(1.0, 0.1)", ("fit.fit.temperature", (1.0, 0.1))),
        ("fit.fit.render.mode=centreline", ("fit.fit.render.mode", "centreline")),
    ],
)
def test_parse_override(text, expected):
    assert parse_override(text) == expected


def test_parse_override_needs_a_path():
    with pytest.raises(ValueError):
        parse_override("2.0")
    with pytest.raises(ValueError):
        parse_override("=2.0")


def test_sweeps_are_tagged():
    variants = sweep("phase5", "fit.connection_reach", [1.0, 1.5, 2.0])
    assert [v.name for v in variants] == [
        "phase5[fit.connection_reach=1.0]",
        "phase5[fit.connection_reach=1.5]",
        "phase5[fit.connection_reach=2.0]",
    ]
    assert [v.variant.fit.connection_reach for v in variants] == [1.0, 1.5, 2.0]
    assert variants[0].sweep == SweepTag("phase5", "fit.connection_reach", 1.0)
    assert len({v.spec_hash for v in variants}) == 3
    # the default value is the base variant itself
    assert variants[1].spec_hash == get_variant("phase5").spec_hash
    base = NamedVariant("mine", get_variant("phase4").variant)
    (one,) = sweep(base, "graph.prune_threshold", [0.2])
    assert one.sweep.base == "mine"


def test_the_skeletonize_graph_method_reproduces_the_baselines():
    sample = build_case("y-thin").sample
    expected = skeletonize_to_graph(sample.mask, sample.spacing)
    for name, cleaned in [
        ("skeletonize", expected),
        ("skeletonize+clean", clean_graph(expected)),
    ]:
        graph = get_variant(name).variant.graph
        raw, clean = graph_stage(None, None, sample.mask, sample.spacing, graph)
        for got, want in [(raw, expected), (clean, cleaned)]:
            assert got.n_edges == want.n_edges
            assert all(
                (a == b).all()
                for a, b in zip(got.edge_paths, want.edge_paths, strict=True)
            )


def test_variants_are_frozen():
    variant = get_variant("phase5").variant
    with pytest.raises(dataclasses.FrozenInstanceError):
        variant.pipeline = "skeleton_fit"
