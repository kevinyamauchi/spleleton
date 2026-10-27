"""Decision quantities of the graph stage, in numpy (plan, Section 11.2).

Every discrete outcome of an event is a continuous quantity ``q`` crossing a
threshold ``tau`` of the graph stage. They are computed from the fitted
segments at scoring time, for the ``assemble`` method (Phase 5); other
methods get their own when they are added.

- ``q_join`` (``tau = join``): the smallest ``join`` at which assembly links
  the event's two anchors. A **link** between two segments is the smaller of
  their closest endpoint distance over the larger of the two radii (endpoint
  clustering) and an end's distance to the other's interior over the larger
  radius (T-junctions). ``q_join`` is the bottleneck (minimax) of the links
  along the best chain of alive segments from an anchor's segments to the
  other's: the weakest link of the window's path when it is connected, the
  facing ends' link when it is broken, and for a junction branch the link of
  the child to its parent. It is 0 when one segment covers both anchors, and
  infinite when no alive segment covers an anchor.
- ``q_prune_connect`` (``tau = prune_threshold``): the largest
  ``prune_threshold`` at which the anchors stay linked at the run's
  ``join``: the widest chain, by opacity.
- ``q_bridge`` and ``bridge_angle`` (``tau = bridge_reach``,
  ``bridge_angle``), when ``q_join > 0``: the surface gap between the facing
  ends of the weakest direct link over their mean radius, and the angle
  between the end's tangent and the bridge, which clean-up's bridging
  compares with its reach and angle.
- ``q_prune_spur`` (``tau = prune_threshold``) and ``q_protrusion``
  (``tau = min_protrusion``), for bumps and protrusions: the largest opacity
  of a fitted spur segment, and the largest length of a spur segment's axis
  outside the other alive tubes, over its radius.

An event's segments and endpoints are selected once, at the fitted
segments; :mod:`spleleton.bench.events` makes the selection. Ported from
stage 0's ``scripts/proto/p03_score.py`` (P3), whose step functions agreed
with the actual graph stage on 98.3% of a ``join`` grid where the anchors
matched (plan, Appendix A).
"""

import inspect
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree

from spleleton.assemble import assemble_graph
from spleleton.graph import resample_polyline
from spleleton.primitives import SegmentArrays

#: Arc-length spacing of sampled segment axes.
SAMPLE_SPACING = 0.25
_EPS = 1e-9


def assemble_settings(config: Any) -> dict[str, float]:
    """``prune_threshold``, ``join`` and ``min_radius`` of an ``assemble`` config.

    Parameters
    ----------
    config : GraphConfig
        The graph stage's settings.

    Returns
    -------
    dict
        The three settings, ``min_radius`` from ``config.params`` or the
        default of :func:`spleleton.assemble.assemble_graph`.
    """
    default = inspect.signature(assemble_graph).parameters["min_radius"].default
    return {
        "prune_threshold": float(config.prune_threshold),
        "join": float(config.join),
        "min_radius": float(dict(config.params).get("min_radius", default)),
    }


@dataclass
class Links:
    """Pairwise link ratios between the candidate segments, and their bottlenecks.

    Attributes
    ----------
    segments : SegmentArrays
        All fitted segments.
    candidate : np.ndarray
        Indices of the segments that pass assembly's radius filter.
    opacity : np.ndarray
        The candidates' opacities.
    alive : np.ndarray
        Whether each candidate is above ``prune_threshold``.
    ratio : np.ndarray
        ``(C, C)`` link ratios between candidates, ``inf`` on the diagonal.
    minimax : np.ndarray
        ``(C, C)`` bottleneck link ratio over chains of alive candidates.
    widest : np.ndarray
        ``(C, C)`` widest chain (by opacity) over links below ``join``.
    """

    segments: SegmentArrays
    candidate: np.ndarray
    opacity: np.ndarray
    alive: np.ndarray
    ratio: np.ndarray
    minimax: np.ndarray
    widest: np.ndarray
    _ends: np.ndarray
    _end_radii: np.ndarray
    _endpoint_ratio: np.ndarray

    def covering(self, point: np.ndarray, tolerance: float, alive_only: bool):
        """Candidates (positions in ``candidate``) whose axis passes near a point."""
        start, end = self._ends[:, 0], self._ends[:, 1]
        vector = end - start
        t = np.clip(
            ((point - start) * vector).sum(1) / np.maximum((vector**2).sum(1), _EPS),
            0,
            1,
        )
        distance = np.linalg.norm(start + t[:, None] * vector - point, axis=1)
        keep = distance <= tolerance
        if alive_only:
            keep &= self.alive
        return np.flatnonzero(keep)

    def facing_ends(self, a: int, b: int):
        """The closest ends of candidates ``a`` and ``b``: points, radii, tangent."""
        ka, kb = np.unravel_index(np.argmin(self._endpoint_ratio[a, b]), (2, 2))
        p, q = self._ends[a, ka], self._ends[b, kb]
        tangent = p - self._ends[a, 1 - ka]
        tangent = tangent / max(float(np.linalg.norm(tangent)), _EPS)
        return (
            p,
            q,
            float(self._end_radii[a, ka]),
            float(self._end_radii[b, kb]),
            tangent,
        )


def segment_links(
    segments: SegmentArrays, prune_threshold: float, join: float, min_radius: float
) -> Links:
    """Link ratios between segments, as :func:`assemble_graph` joins them.

    Parameters
    ----------
    segments : SegmentArrays
        The fitted segments.
    prune_threshold, join, min_radius : float
        The assembly's settings.

    Returns
    -------
    Links
    """
    radius = segments.radius
    candidate = np.flatnonzero(radius.mean(axis=1) >= min_radius)
    start, end = segments.start[candidate], segments.end[candidate]
    r = radius[candidate]
    opacity = np.asarray(segments.opacity, dtype=float)[candidate]
    c = len(candidate)
    ends = np.stack([start, end], axis=1)  # (C, 2, D)
    ratio = np.full((c, c), np.inf)
    endpoint_ratio = np.full((c, c, 2, 2), np.inf)
    if c:
        # endpoint clustering: distance over the larger radius, 4 endpoint pairs
        d = np.linalg.norm(ends[:, None, :, None] - ends[None, :, None, :], axis=-1)
        limit = np.maximum(r[:, None, :, None], r[None, :, None, :])
        endpoint_ratio = d / limit
        ratio = endpoint_ratio.reshape(c, c, 4).min(axis=2)
        # T-junctions: an end of i near the interior of j
        vector = end - start
        length = np.linalg.norm(vector, axis=1)
        for k in range(2):
            p = ends[:, k]
            offset = p[:, None, :] - start[None]
            t = np.clip(
                (offset * vector[None]).sum(-1) / np.maximum(length**2, _EPS)[None],
                0,
                1,
            )
            closest = start[None] + t[..., None] * vector[None]
            distance = np.linalg.norm(p[:, None, :] - closest, axis=-1)
            rj = r[None, :, 0] + t * (r[None, :, 1] - r[None, :, 0])
            along = t * length[None]
            interior = (rj < along) & (along < length[None] - rj)
            value = np.where(interior, distance / np.maximum(rj, r[:, None, k]), np.inf)
            ratio = np.minimum(ratio, np.minimum(value, value.T))
        np.fill_diagonal(ratio, np.inf)
    alive = opacity >= prune_threshold
    minimax = _bottleneck(
        np.where(alive[:, None] & alive[None], ratio, np.inf), minimise=True
    )
    weight = np.where(ratio < join, np.minimum(opacity[:, None], opacity[None]), 0.0)
    widest = _bottleneck(weight, minimise=False)
    np.fill_diagonal(widest, opacity)
    return Links(
        segments, candidate, opacity, alive, ratio, minimax, widest,
        ends, r, endpoint_ratio,
    )  # fmt: skip


def _bottleneck(weight: np.ndarray, minimise: bool) -> np.ndarray:
    """All-pairs minimax (or maximin) chain values, from a spanning tree."""
    c = len(weight)
    empty = np.inf if minimise else 0.0
    out = np.full((c, c), empty)
    if c == 0:
        return out
    if minimise:
        # shifted by 1: csgraph treats zero weights as missing edges
        dense = np.where(np.isfinite(weight), weight + 1.0, 0.0)
    else:
        dense = np.where(weight > 0, 2.0 - weight, 0.0)  # a maximum spanning tree
    tree = minimum_spanning_tree(dense).toarray()
    tree = np.maximum(tree, tree.T)
    neighbours = [np.flatnonzero(tree[i]) for i in range(c)]
    for source in range(c):
        out[source, source] = 0.0 if minimise else np.inf
        stack = [source]
        while stack:
            u = stack.pop()
            for v in neighbours[u]:
                if v == source or out[source, v] != empty:
                    continue
                w = tree[u, v] - 1.0 if minimise else 2.0 - tree[u, v]
                out[source, v] = (
                    max(out[source, u], w) if minimise else min(out[source, u], w)
                )
                stack.append(v)
    return out


def _first_covering(links: Links, anchors, alive_only: bool) -> np.ndarray:
    """The candidates covering the first anchor (point, tolerance) that has any."""
    for point, tolerance in anchors:
        found = links.covering(point, tolerance, alive_only)
        if len(found):
            return found
    return np.zeros(0, dtype=int)


def connection_quantities(links: Links, a_anchors, b_anchors) -> dict[str, float]:
    """``q_join``, ``q_prune_connect``, ``q_bridge`` and ``bridge_angle``.

    Parameters
    ----------
    links : Links
        The fit's segment links.
    a_anchors, b_anchors : sequence of (np.ndarray, float)
        Candidate anchor points with their match tolerances, in the order
        to try: each side's anchor is the first that a segment covers (windows
        snap their anchors inward this way, as their outcome does).

    Returns
    -------
    dict
        ``q_bridge`` and ``bridge_angle`` are NaN unless ``0 < q_join < inf``.
    """
    a = _first_covering(links, a_anchors, True)
    b = _first_covering(links, b_anchors, True)
    out = {
        "q_join": np.inf,
        "q_prune_connect": 0.0,
        "q_bridge": np.nan,
        "bridge_angle": np.nan,
    }
    if len(a) and len(b):
        out["q_join"] = float(links.minimax[np.ix_(a, b)].min())
    a_all = _first_covering(links, a_anchors, False)
    b_all = _first_covering(links, b_anchors, False)
    if len(a_all) and len(b_all):
        out["q_prune_connect"] = float(links.widest[np.ix_(a_all, b_all)].max())
    if len(a) and len(b) and 0 < out["q_join"] < np.inf:
        # the facing ends: the weakest direct link between the two sides
        ratio = links.ratio[np.ix_(a, b)]
        i, j = np.unravel_index(np.argmin(ratio), ratio.shape)
        p, q, ra, rb, tangent = links.facing_ends(int(a[i]), int(b[j]))
        vector = q - p
        distance = float(np.linalg.norm(vector))
        out["q_bridge"] = (distance - ra - rb) / (0.5 * (ra + rb))
        cos = tangent @ vector / max(distance, _EPS)
        out["bridge_angle"] = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    return out


def spur_quantities(links: Links, spur: np.ndarray) -> dict[str, float]:
    """``q_prune_spur`` and ``q_protrusion`` of an event's spur segments.

    Parameters
    ----------
    links : Links
        The fit's segment links.
    spur : np.ndarray
        Positions in ``links.candidate`` of the event's spur segments.

    Returns
    -------
    dict
        NaN for both when there is no spur segment.
    """
    out = {"q_prune_spur": np.nan, "q_protrusion": np.nan}
    if not len(spur):
        return out
    out["q_prune_spur"] = float(links.opacity[spur].max())
    start, end = links._ends[:, 0], links._ends[:, 1]
    radius = links._end_radii
    values = []
    for s in spur:
        points, _ = resample_polyline(np.stack([start[s], end[s]]), SAMPLE_SPACING)
        others = np.flatnonzero(links.alive)
        others = others[others != s]
        inside = np.zeros(len(points), dtype=bool)
        if len(others):
            a, b, rr = start[others], end[others], radius[others]
            v = b - a
            t = np.clip(
                ((points[:, None] - a[None]) * v[None]).sum(-1)
                / np.maximum((v**2).sum(1), _EPS)[None],
                0,
                1,
            )
            distance = np.linalg.norm(
                a[None] + t[..., None] * v[None] - points[:, None], axis=-1
            )
            reach = rr[None, :, 0] + t * (rr[None, :, 1] - rr[None, :, 0])
            inside = (distance <= reach).any(axis=1)
        steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
        length = float(steps[~inside[:-1] & ~inside[1:]].sum())
        values.append(length / float(radius[s].mean()))
    out["q_protrusion"] = max(values)
    return out
