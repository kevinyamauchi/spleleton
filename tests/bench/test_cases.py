import numpy as np
import pytest

from spleleton.bench.cases import build_case, case_names, prediction_payload
from spleleton.metrics import compare_graphs


@pytest.mark.parametrize("name", case_names())
def test_cases_are_consistent(name):
    case = build_case(name)
    sample = case.sample
    assert sample.mask.shape == (64, 64, 64)
    assert set(case.predictions) == set(case.planted)
    assert case.planted["true"] == {}
    # the true graph lies in the clean mask
    points = np.round(sample.graph.sample_points(0.5)[0]).astype(int)
    assert sample.clean_mask[tuple(points.T)].all()
    payload = prediction_payload(case)
    assert all(key.split("__")[0] in case.predictions for key in payload)


def test_corrupted_cases_record_their_corruption():
    types = {
        name: [r["type"] for r in build_case(name).sample.corruptions]
        for name in case_names()
    }
    assert types == {
        "tube": [],
        "tube-gap": ["gap"],
        "y-thin": [],
        "y-thin-gap": ["junction_gap"],
        "tube-bump": ["bump"],
        "contact": ["contact"],
    }
    (gap,) = build_case("y-thin-gap").sample.corruptions
    assert gap["ratio"] == pytest.approx(0.3)
    assert gap["removed_voxels"] > 0
    assert gap["clearance"] > 0


def test_true_predictions_score_almost_perfectly():
    from spleleton.assemble import assemble_graph

    for name in case_names():
        case = build_case(name)
        graph = assemble_graph(case.predictions["true"])
        scores = compare_graphs(graph, case.sample.graph)
        assert scores.n_components == case.sample.graph.n_components()
        assert scores.centreline_f1 > 0.95, name
