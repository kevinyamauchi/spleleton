"""Backend-neutral parts of the segment geometry."""

from typing import Any, NamedTuple

#: Floor on squared segment lengths, which keeps degenerate segments finite.
DEFAULT_EPS = 1e-12


class ClosestPoints(NamedTuple):
    """The closest points on a set of segments to a set of query points.

    The fields are arrays of the backend that computed them (torch tensors or
    JAX arrays).

    Attributes
    ----------
    t : array
        Segment parameter of the closest point, in ``[0, 1]``. ``0`` is the
        segment start and ``1`` its end.
    point : array
        Coordinates of the closest point, with a trailing dimension of size D.
    distance : array
        Euclidean distance from the query point to the closest point.
    """

    t: Any
    point: Any
    distance: Any
