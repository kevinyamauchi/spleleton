import dataclasses
import json
from typing import Literal

import pytest

from spleleton._serialise import canonical_json, from_dict, spec_hash, to_dict
from spleleton.fit import FitConfig
from spleleton.pipeline import (
    CleanConfig,
    GraphConfig,
    ScratchFitConfig,
    SkeletonFitConfig,
)
from spleleton.synthetic.dataset import SampleConfig, airway_config
from spleleton.synthetic.trees import TreeConfig


@dataclasses.dataclass(frozen=True)
class _Spec:
    pipeline: Literal["skeleton_fit", "scratch_fit"] = "scratch_fit"
    fit: SkeletonFitConfig | ScratchFitConfig | None = None
    graph: GraphConfig = dataclasses.field(default_factory=GraphConfig)


def _json_round_trip(config):
    return from_dict(type(config), json.loads(json.dumps(to_dict(config))))


@pytest.mark.parametrize(
    "config",
    [
        airway_config(3, spacing=(2.0, 1.0, 1.0)),
        SampleConfig(tree=TreeConfig(side_branch_probability=1.0)),
        ScratchFitConfig(fit=FitConfig(temperature=(1.0, 0.1)), n_rounds=2),
        GraphConfig(
            method="layout",
            params=(("contained", 0.7), ("min_edge", 0.25)),
            clean=CleanConfig(enabled=False),
        ),
        _Spec(fit=SkeletonFitConfig()),
        _Spec(fit=ScratchFitConfig()),
        _Spec(),
    ],
)
def test_configs_round_trip_through_json(config):
    restored = _json_round_trip(config)
    assert restored == config
    assert spec_hash(restored) == spec_hash(config)


def test_union_members_are_told_apart():
    assert isinstance(
        _json_round_trip(_Spec(fit=SkeletonFitConfig())).fit, SkeletonFitConfig
    )
    assert isinstance(
        _json_round_trip(_Spec(fit=ScratchFitConfig())).fit, ScratchFitConfig
    )


def test_nested_tuples_are_rebuilt():
    restored = _json_round_trip(GraphConfig(params=(("contained", 0.7),)))
    assert restored.params == (("contained", 0.7),)
    assert isinstance(restored.params[0], tuple)
    assert isinstance(
        _json_round_trip(FitConfig(temperature=(1.0, 0.1))).temperature, tuple
    )


def test_canonical_json_and_hash():
    config = GraphConfig(join=1.5)
    text = canonical_json(config)
    assert " " not in text
    assert text.index('"clean"') < text.index('"join"') < text.index('"method"')
    assert spec_hash(config) == spec_hash(GraphConfig(join=1.5))
    assert spec_hash(config) != spec_hash(GraphConfig(join=1.0))
    assert len(spec_hash(config)) == 64


def test_missing_fields_keep_defaults_and_unknown_fields_fail():
    assert from_dict(GraphConfig, {"join": 2.0}) == GraphConfig(join=2.0)
    with pytest.raises(ValueError, match="no fields"):
        from_dict(GraphConfig, {"joins": 2.0})
