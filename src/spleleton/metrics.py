"""Compare a predicted skeleton graph with the ground truth.

Tolerances scale with the local ground-truth radius, because a centreline is
only defined to within a fraction of the tube's width: a point counts as
correct if it lies within ``max(min_tolerance, relative * radius)``.
"""

import dataclasses
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from spleleton.graph import SkeletonGraph


@dataclass(frozen=True)
class SkeletonMetrics:
    """Scores of a predicted skeleton against the ground truth.

    Attributes
    ----------
    centreline_precision : float
        Fraction of the predicted centreline within tolerance of the true one.
    centreline_recall : float
        Fraction of the true centreline within tolerance of the predicted one.
    centreline_f1 : float
        Harmonic mean of centreline precision and recall.
    junction_precision : float
        Fraction of predicted junctions (degree >= 3) matched to a true one.
    junction_recall : float
        Fraction of true junctions matched to a predicted one.
    junction_f1 : float
        Harmonic mean of junction precision and recall.
    n_components : int
        Connected components of the prediction. The true tree has one.
    n_cycles : int
        Independent cycles of the prediction (edges - nodes + components).
        The true tree has none.
    n_spur_ends : int
        Predicted ends that match no true end and lie off the true
        centreline: the tips of spurs.
    n_break_ends : int
        Predicted ends that match no true end but lie on the true
        centreline: the two sides of a break each contribute one.
    radius_error : float
        Mean absolute radius error over the correct predicted centreline, or
        NaN if the prediction has no radii.
    """

    centreline_precision: float
    centreline_recall: float
    centreline_f1: float
    junction_precision: float
    junction_recall: float
    junction_f1: float
    n_components: int
    n_cycles: int
    n_spur_ends: int
    n_break_ends: int
    radius_error: float

    def to_dict(self) -> dict[str, float]:
        """The metrics as a plain dictionary."""
        return dataclasses.asdict(self)


def compare_graphs(
    predicted: SkeletonGraph,
    target: SkeletonGraph,
    relative_tolerance: float = 0.5,
    junction_tolerance: float = 1.0,
    end_tolerance: float = 1.5,
    min_tolerance: float = 1.0,
    sample_spacing: float = 0.25,
) -> SkeletonMetrics:
    """Score a predicted skeleton graph against the ground truth.

    Parameters
    ----------
    predicted : SkeletonGraph
        The skeleton to score.
    target : SkeletonGraph
        The ground truth. Must have radii.
    relative_tolerance : float
        Centreline tolerance, in units of the local true radius.
    junction_tolerance : float
        Junction matching tolerance, in units of the true radius at the
        junction.
    end_tolerance : float
        End matching tolerance, in units of the true radius at the end. It is
        larger than the others because a true end sits at the centre of the
        tube's rounded tip, while skeletons usually reach into the tip.
    min_tolerance : float
        Smallest tolerance, in physical units.
    sample_spacing : float
        Arc-length spacing used to sample both centrelines.

    Returns
    -------
    SkeletonMetrics
        The scores.
    """
    if target.edge_radii is None:
        raise ValueError("the target graph must have radii")
    target_points, target_radii, _ = target.sample_points(sample_spacing)
    predicted_points, predicted_radii, _ = predicted.sample_points(sample_spacing)

    def tolerance(radius: np.ndarray, relative: float) -> np.ndarray:
        return np.maximum(min_tolerance, relative * radius)

    # centreline precision: is each predicted point near the true centreline?
    target_tree = cKDTree(target_points)
    if len(predicted_points):
        distance, nearest = target_tree.query(predicted_points)
        correct = distance <= tolerance(target_radii[nearest], relative_tolerance)
        precision = float(correct.mean())
    else:
        correct = np.zeros(0, dtype=bool)
        precision = float("nan")

    # centreline recall: is each true point near the predicted centreline?
    if len(predicted_points):
        distance, _ = cKDTree(predicted_points).query(target_points)
        recall = float(np.mean(distance <= tolerance(target_radii, relative_tolerance)))
    else:
        recall = 0.0

    # junctions
    target_node_radii = target.node_radii()
    target_junctions = target.junction_nodes
    predicted_junctions = predicted.junction_nodes
    n_matched_junctions = len(
        _match(
            predicted.node_positions[predicted_junctions],
            target.node_positions[target_junctions],
            tolerance(target_node_radii[target_junctions], junction_tolerance),
        )
    )
    junction_precision = _ratio(n_matched_junctions, len(predicted_junctions))
    junction_recall = _ratio(n_matched_junctions, len(target_junctions))

    # ends that match no true end are spur tips or break ends
    target_ends = target.end_nodes
    predicted_ends = predicted.end_nodes
    matches = _match(
        predicted.node_positions[predicted_ends],
        target.node_positions[target_ends],
        tolerance(target_node_radii[target_ends], end_tolerance),
    )
    unmatched = np.setdiff1d(np.arange(len(predicted_ends)), [i for i, _ in matches])
    n_spur_ends = n_break_ends = 0
    if len(unmatched):
        positions = predicted.node_positions[predicted_ends[unmatched]]
        distance, nearest = target_tree.query(positions)
        on_centreline = distance <= tolerance(target_radii[nearest], relative_tolerance)
        n_break_ends = int(on_centreline.sum())
        n_spur_ends = int((~on_centreline).sum())

    if predicted_radii is not None and correct.any():
        _, nearest = target_tree.query(predicted_points[correct])
        radius_error = float(
            np.mean(np.abs(predicted_radii[correct] - target_radii[nearest]))
        )
    else:
        radius_error = float("nan")

    n_components = predicted.n_components()
    return SkeletonMetrics(
        centreline_precision=precision,
        centreline_recall=recall,
        centreline_f1=_f1(precision, recall),
        junction_precision=junction_precision,
        junction_recall=junction_recall,
        junction_f1=_f1(junction_precision, junction_recall),
        n_components=n_components,
        n_cycles=predicted.n_edges - predicted.n_nodes + n_components,
        n_spur_ends=n_spur_ends,
        n_break_ends=n_break_ends,
        radius_error=radius_error,
    )


def _match(
    predicted: np.ndarray, target: np.ndarray, target_tolerance: np.ndarray
) -> list[tuple[int, int]]:
    """One-to-one matching of points within each target point's tolerance."""
    if len(predicted) == 0 or len(target) == 0:
        return []
    distance = np.linalg.norm(predicted[:, None] - target[None], axis=-1)
    allowed = distance <= target_tolerance[None, :]
    cost = np.where(allowed, distance, 1e9)
    rows, columns = linear_sum_assignment(cost)
    return [(i, j) for i, j in zip(rows, columns, strict=True) if allowed[i, j]]


def _ratio(numerator: int, denominator: int) -> float:
    """A ratio that is 1 when there is nothing to find."""
    return 1.0 if denominator == 0 else numerator / denominator


def _f1(precision: float, recall: float) -> float:
    if np.isnan(precision) or precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
