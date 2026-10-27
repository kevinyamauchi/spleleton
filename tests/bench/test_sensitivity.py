import math

import numpy as np
import pytest

pd = pytest.importorskip("pandas")

from spleleton.bench import sensitivity as sens  # noqa: E402
from spleleton.bench.stats import POOLED  # noqa: E402
from spleleton.bench.variants import get_variant  # noqa: E402

CONFIG = get_variant("phase5").variant.graph


def test_the_grid_holds_each_threshold_and_the_pairs_once():
    settings = sens.grid_settings(CONFIG)
    assert settings[0] == {}
    keys = [tuple(sorted(s.items())) for s in settings]
    assert len(keys) == len(set(keys))
    for name in sens.THRESHOLDS:
        values = sens.grid_values(CONFIG, name)
        assert len(values) == 9
        assert any(np.isclose(v, sens.threshold_value(CONFIG, name)) for v in values)
    # the 2D grids: every pair of values of an interacting pair
    joins = [s for s in settings if set(s) == {"join", "prune_threshold"}]
    assert len(joins) == 8 * 8
    changed = sens.graph_with(
        CONFIG, {"join": 2.0, "contained": 0.25, "bridge_reach": 3.0}
    )
    assert sens.threshold_value(changed, "join") == 2.0
    assert sens.threshold_value(changed, "contained") == 0.25
    assert sens.threshold_value(changed, "bridge_reach") == 3.0
    assert sens.threshold_value(changed, "min_island") == sens.threshold_value(
        CONFIG, "min_island"
    )


def _events():
    base = {"subset": "s", "seed": 0, "stage": "raw"}
    return pd.DataFrame(
        [
            {
                **base,
                "event_type": "window",
                "event_id": "w0",
                "q_join": 0.4,
                "q_prune_connect": 0.99,
                "q_bridge": np.nan,
                "bridge_angle": np.nan,
                "q_prune_spur": np.nan,
                "q_protrusion": np.nan,
            },
            {
                **base,
                "event_type": "window",
                "event_id": "w1",
                "q_join": 1.3,
                "q_prune_connect": 0.99,
                "q_bridge": 1.5,
                "bridge_angle": 10.0,
                "q_prune_spur": np.nan,
                "q_protrusion": np.nan,
            },
            {
                **base,
                "event_type": "bump",
                "event_id": "b0",
                "q_join": np.nan,
                "q_prune_connect": np.nan,
                "q_bridge": np.nan,
                "bridge_angle": np.nan,
                "q_prune_spur": 0.98,
                "q_protrusion": 1.2,
            },
        ]
    )


def test_the_step_functions_follow_the_critical_values():
    events = _events()
    critical = sens.critical_rows(events)
    assert set(critical["threshold"]) == {
        "join",
        "prune_threshold",
        "bridge_reach",
        "bridge_angle",
        "min_protrusion",
    }
    p = sens.predict(events, {}, CONFIG).set_index(["event_id", "stage"])["predicted"]
    assert p["w0", "raw"] and not p["w1", "raw"]
    assert p["w1", "clean"]  # bridged: 1.5 <= reach 2 and 10 <= 30 degrees
    assert p["b0", "raw"] and not p["b0", "clean"]  # a protrusion under 1.5 r
    p = sens.predict(events, {"join": 1.5, "bridge_reach": 1.0}, CONFIG)
    p = p.set_index(["event_id", "stage"])["predicted"]
    assert p["w1", "raw"]
    p = sens.predict(events, {"min_protrusion": 1.0, "prune_threshold": 0.99}, CONFIG)
    p = p.set_index(["event_id", "stage"])["predicted"]
    assert not p["b0", "raw"]  # pruned


def _implicit():
    rows = []
    for seed in range(3):
        for k, (q, dq) in enumerate([(1.05, -1.0), (0.95, 1.0), (0.5, 1e-7)]):
            rows.append(
                {
                    "subset": "s",
                    "seed": seed,
                    "event_type": "window",
                    "event_id": f"w{k}",
                    "parameter": "fit.connection_weight",
                    "q": q,
                    "tau": 1.0,
                    "dq": dq,
                    "flip_distance": sens.flip_distance(q, 1.0, dq),
                }
            )
    return pd.DataFrame(rows)


def test_elasticities_count_flips_within_ten_percent():
    implicit = _implicit()
    assert np.isinf(implicit["flip_distance"].iloc[2])  # unsteerable
    table = sens.elasticities(implicit)
    row = table[table["subset"] == POOLED].iloc[0]
    # w0 connects (good) and w1 disconnects (bad) within log 1.1: net 0
    assert row["good"] == 3 and row["bad"] == 3 and row["elasticity"] == 0
    assert row["unsteerable"] == pytest.approx(1 / 3)
    frame = pd.DataFrame(
        [
            {
                "parameter": "fit.fit.losses.bce.weight",
                "subset": POOLED,
                "outcome": "thin_junction",
                "elasticity": 0.1,
            },
            {
                "parameter": "fit.fit.losses.dice.weight",
                "subset": "all",
                "outcome": "bump_spur",
                "elasticity": -0.3,
            },
            {
                "parameter": "fit.fit.losses.dice.weight",
                "subset": "all",
                "outcome": "junction",
                "elasticity": 0.9,
            },
        ]
    )
    chosen = sens.second_weight(frame)
    # the junction rate is not a target outcome, so dice wins on the bump spur
    assert chosen["parameter"] == "fit.fit.losses.dice.weight"
    assert chosen["elasticity"] == -0.3


def test_validation_against_the_prediction():
    implicit = _implicit().assign(outcome="window")
    base = pd.DataFrame(
        [
            {
                "subset": "s",
                "seed": s,
                "stage": "raw",
                "event_type": "window",
                "event_id": f"w{k}",
                "connected": k != 0,
            }
            for s in range(3)
            for k in range(3)
        ]
    )
    other = base.assign(connected=[k != 1 for s in range(3) for k in range(3)])
    rows = sens.validation_rows(base, other, implicit, "fit.connection_weight", 1.2)
    pooled = next(r for r in rows if r["subset"] == POOLED and r["outcome"] == "window")
    # one connects and one disconnects per volume: no change, as predicted
    assert pooled["observed"] == pytest.approx(0) and pooled["predicted"] == 0
    summary = sens.validation_agreement(pd.DataFrame(rows))
    assert summary["morris_needed"] is None  # one weight: no ranking


def test_morris_finds_the_coefficients_of_a_linear_model():
    points, orders = sens.morris_design(3, 6, seed=1)
    coefficients = np.array([3.0, -1.0, 0.0])
    outputs = points @ coefficients
    effects = sens.morris_effects(points, orders, outputs)
    assert np.allclose(effects["mu_star"], np.abs(coefficients))
    assert np.allclose(effects["sigma"], 0)
    values = sens.morris_values(np.array([0.0, 1.0]), {"a": (1.0, 4.0), "b": (2, 8)})
    assert values == {"a": pytest.approx(1.0), "b": pytest.approx(8.0)}


def test_robustness_margins_and_bump_levels():
    rng = np.random.default_rng(0)
    ratio = rng.uniform(0.1, 1.0, 400)
    events = pd.DataFrame(
        {
            "subset": "asym",
            "seed": rng.integers(0, 8, 400),
            "stage": "raw",
            "event_type": "junction",
            "radius_ratio": ratio,
            "connected": ratio > 0.3,
        }
    )
    margins = sens.robustness_margins(events, resamples=50)
    margin = margins[margins["level"] == 0.5]["margin"].iloc[0]
    assert 0.2 < margin < 0.45
    bumps = pd.DataFrame(
        [
            {
                "subset": f"bumps-h{h}",
                "seed": 0,
                "stage": "clean",
                "event_type": "bump",
                "event_id": "bump#0",
                "spur": h >= 1.0,
            }
            for h in (0.5, 1.0, 1.5)
        ]
    )
    levels = sens.bump_critical_levels(bumps)
    assert levels["critical_level"].iloc[0] == 1.0


def test_flip_distance():
    assert sens.flip_distance(0.8, 1.0, 0.1) == pytest.approx(2.0)
    assert math.isinf(sens.flip_distance(0.8, 1.0, 0.0))
