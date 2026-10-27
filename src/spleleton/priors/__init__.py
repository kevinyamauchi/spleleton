"""Priors (extra loss terms) for fitting skeletons.

Each prior exists once per backend (``TorchFreeEndCost``, ``JaxFreeEndCost``,
...). A torch prior is a callable ``prior(segments, iteration) -> {name: term}``
for the ``regularisers`` argument of :func:`spleleton.fit.fit_torch`. A JAX
prior, for :func:`spleleton.fit.fit_jax`, follows
:class:`spleleton.fit.JaxPrior`: a hashable object of settings, with
``init``, ``update`` and ``__call__`` methods around a state of arrays. Terms
are already weighted.

All costs are expressed in the units of the per-voxel mean BCE: a cost of
``c`` voxels is ``c / n_voxels``. A segment that explains ``A`` more voxels
than its neighbours lowers the BCE by roughly ``A / n_voxels``, so a prior
costing ``c`` voxels removes structure that explains fewer than about ``c``
voxels.
"""

from typing import TYPE_CHECKING

from spleleton._backends import lazy_exports

if TYPE_CHECKING:
    from spleleton.priors._priors_jax import (
        JaxConnectionLoss,
        JaxFreeEndCost,
        endpoint_pairs_jax,
    )
    from spleleton.priors._priors_torch import (
        TorchConnectionLoss,
        TorchFreeEndCost,
        endpoint_pairs_torch,
    )

__all__ = [
    "JaxConnectionLoss",
    "JaxFreeEndCost",
    "TorchConnectionLoss",
    "TorchFreeEndCost",
    "endpoint_pairs_jax",
    "endpoint_pairs_torch",
]

__getattr__, __dir__ = lazy_exports(
    __name__,
    torch={
        "._priors_torch": [
            "TorchConnectionLoss",
            "TorchFreeEndCost",
            "endpoint_pairs_torch",
        ]
    },
    jax={
        "._priors_jax": [
            "JaxConnectionLoss",
            "JaxFreeEndCost",
            "endpoint_pairs_jax",
        ]
    },
)
