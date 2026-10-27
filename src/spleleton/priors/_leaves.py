"""Leaf chains of a segment layout (shared by all backends)."""

import numpy as np

from spleleton.initialise import SegmentLayout


def leaf_chains(layout: SegmentLayout) -> list[np.ndarray]:
    """The segment chains of the layout's leaf edges.

    A leaf edge has exactly one node of degree 1 (an isolated edge, with two,
    is not a leaf). Edges without segments are skipped.

    Parameters
    ----------
    layout : SegmentLayout
        Which segments make up which edge.

    Returns
    -------
    list of np.ndarray
        The segment indices of each leaf edge, in layout order.
    """
    degrees = np.bincount(np.asarray(layout.edges).ravel(), minlength=layout.n_nodes)
    return [
        np.asarray(chain)
        for (u, v), chain in zip(layout.edges, layout.edge_segments, strict=True)
        if (degrees[u] == 1) != (degrees[v] == 1) and len(chain)
    ]
