"""Differentiable rendering of skeleton primitives into a voxel grid.

Each primitive contributes a value ``g_i(x)`` in ``[0, 1)`` at every voxel
centre ``x``. There is no camera, so there is no projection and no depth
sorting; the contributions are combined at each voxel by one of three unions:

- ``"max"`` (default): ``O(x) = max_i g_i(x)``. For primitives of equal
  opacity this is exactly the union of their tubes, however finely a tube is
  cut into primitives. Each voxel's gradient goes to the primitive that
  dominates it, so a fully redundant primitive gets no rendering gradient.
- ``"soft"``: ``O(x) = 1 - prod_i (1 - g_i(x))``, the accumulated opacity of
  3D Gaussian splatting without a ray. It treats contributions as independent,
  so overlapping primitives reinforce each other: a tube cut into many short,
  overlapping segments renders fatter than the same tube as one segment.
- ``"softmax"``: ``O(x) = sum_i w_i g_i(x)`` with ``w = softmax(g / T)``, a
  temperature-controlled max. As ``T`` goes to 0 it becomes the max; for
  larger ``T``, primitives with similar contributions share the gradient.
  Like the max (and unlike the soft union), duplicated primitives render the
  same as one. Only primitives within the cutoff distance of a voxel take
  part, so that far-away primitives do not pull the average down and the
  result does not depend on the block size.

Two kernels are available:

- ``"tube"``: ``g = opacity * sigmoid((r(t) - d) / wall_width)``, a tube of
  radius ``r`` whose wall is about ``wall_width`` thick. ``O > 0.5`` is
  (almost exactly) the union of the tubes, matching a rasterised mask.
- ``"centreline"``: ``g = opacity * exp(-d^2 / (2 sigma^2))``, a thin line of
  fixed width that ignores the radius.

Here ``d`` is the distance to the primitive's centreline and ``r(t)`` the
radius at the closest point.

**Block binning.** The grid is split into cubic blocks of ``block_size``
voxels per axis (the 3D analogue of the 16 x 16 screen tiles of 3D Gaussian
splatting). Each primitive is assigned to the blocks its padded bounding box
overlaps, and each block evaluates only its own primitives. A primitive's box
is padded by its radius plus ``cutoff`` wall widths (tube) or by ``cutoff``
sigmas (centreline); beyond that its contribution is below about
``sigmoid(-cutoff)`` or ``exp(-cutoff^2 / 2)`` and is dropped. Blocks are
evaluated in batches, so memory stays bounded by ``max_pairs``.

Coordinates are physical: voxel ``i`` has its centre at ``i * spacing``.

Each renderer exists once per backend, with a ``_torch`` or ``_jax`` suffix.
The JAX renderer bins with fixed capacities (:class:`BinCapacity`), so that it
can run under ``jit``; see :mod:`spleleton.render._render_jax`.
"""

from typing import TYPE_CHECKING

from spleleton._backends import lazy_exports
from spleleton.render._config import Mode, RenderConfig, Union

if TYPE_CHECKING:
    from spleleton.render._render_jax import (
        BinCapacity,
        JaxBlockBins,
        bin_capacity_jax,
        bin_primitives_jax,
        render_dense_jax,
        render_jax,
    )
    from spleleton.render._render_torch import (
        TorchBlockBins,
        bin_primitives_torch,
        render_dense_torch,
        render_torch,
    )

__all__ = [
    "BinCapacity",
    "JaxBlockBins",
    "Mode",
    "RenderConfig",
    "TorchBlockBins",
    "Union",
    "bin_capacity_jax",
    "bin_primitives_jax",
    "bin_primitives_torch",
    "render_dense_jax",
    "render_dense_torch",
    "render_jax",
    "render_torch",
]

__getattr__, __dir__ = lazy_exports(
    __name__,
    torch={
        "._render_torch": [
            "TorchBlockBins",
            "bin_primitives_torch",
            "render_dense_torch",
            "render_torch",
        ]
    },
    jax={
        "._render_jax": [
            "BinCapacity",
            "JaxBlockBins",
            "bin_capacity_jax",
            "bin_primitives_jax",
            "render_dense_jax",
            "render_jax",
        ]
    },
)
