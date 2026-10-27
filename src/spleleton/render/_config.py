"""Settings of the renderers (shared by all backends)."""

from dataclasses import dataclass
from typing import Literal

Mode = Literal["tube", "centreline"]
Union = Literal["max", "soft", "softmax"]


@dataclass(frozen=True)
class RenderConfig:
    """Parameters of the renderer.

    Parameters
    ----------
    mode : {"tube", "centreline"}
        Which kernel to render.
    union : {"max", "soft", "softmax"}
        How contributions of different primitives are combined.
    temperature : float
        Temperature of the ``"softmax"`` union. ``0`` gives the max.
    wall_width : float
        Width of the tube wall (the sigmoid's scale), in physical units.
    sigma : float
        Standard deviation of the centreline profile, in physical units.
    cutoff : float
        Padding of each primitive's bounding box, in wall widths (tube) or
        sigmas (centreline), beyond the radius.
    block_size : int
        Voxels per axis of a block.
    max_pairs : int
        Upper bound on voxel-primitive pairs evaluated at once.
    """

    mode: Mode = "tube"
    union: Union = "max"
    temperature: float = 0.1
    wall_width: float = 0.5
    sigma: float = 1.0
    cutoff: float = 5.0
    block_size: int = 16
    max_pairs: int = 2**22
