"""Fit primitives to a target image by differentiable rendering.

This is per-volume optimisation, as in 3D Gaussian splatting: there is no
network, only the primitives' parameters and Adam. The topology is fixed: no
primitives are added or removed during the fit (that is density control, a
later phase), although the opacity penalty can drive redundant primitives
towards zero opacity.

The loss is a weighted sum of named terms (:class:`LossTerm`, see
:mod:`spleleton.fit._losses` for the terms registered), plus the
regularisers. The default (:func:`default_losses`) is

    BCE(O, target) + (1 - softDice(O, target))
    + 0.5 * sum(opacity_i * volume_i) / n_voxels

where ``O`` is the rendered occupancy and ``volume_i`` the volume of segment
``i``'s rendered kernel in voxels (not differentiated). The sparsity term is
in the same units as the per-voxel mean BCE, so it means the same for any
image size and dimension: a segment survives only if it explains (roughly)
more than a fraction (the term's weight) of its own volume that no other
segment covers. A segment hidden inside another tube explains none, so it is
pruned. The torch fit knows only the ``bce``, ``dice`` and ``sparsity``
terms; other terms are JAX only.

For :func:`fit_torch`, regularisers are callables
``regulariser(primitives, iteration) -> {name: weighted term}``, e.g. the
priors in :mod:`spleleton.priors`; their terms are added to the loss as they
are and recorded in the history under their names. For :func:`fit_jax`, they
follow the :class:`JaxPrior` protocol, which separates a prior's settings
from its data so that the fit step can be jitted.

The fit exists once per backend, with a ``_torch`` or ``_jax`` suffix. Torch
updates the primitives in place and returns the history; JAX returns
``(segments, history)``.
"""

from typing import TYPE_CHECKING

from spleleton._backends import lazy_exports
from spleleton.fit._config import FitConfig
from spleleton.fit._losses import LossTerm, default_losses

if TYPE_CHECKING:
    from spleleton.fit._fit_jax import (
        FitState,
        JaxPrior,
        LossContext,
        bce_loss_jax,
        dice_loss_jax,
        fit_jax,
        loss_gradient_jax,
        loss_terms_jax,
        register_loss_jax,
        soft_skeleton_jax,
    )
    from spleleton.fit._fit_torch import bce_loss_torch, dice_loss_torch, fit_torch

__all__ = [
    "FitConfig",
    "FitState",
    "JaxPrior",
    "LossContext",
    "LossTerm",
    "bce_loss_jax",
    "bce_loss_torch",
    "default_losses",
    "dice_loss_jax",
    "dice_loss_torch",
    "fit_jax",
    "fit_torch",
    "loss_gradient_jax",
    "loss_terms_jax",
    "register_loss_jax",
    "soft_skeleton_jax",
]

__getattr__, __dir__ = lazy_exports(
    __name__,
    torch={"._fit_torch": ["bce_loss_torch", "dice_loss_torch", "fit_torch"]},
    jax={
        "._fit_jax": [
            "FitState",
            "JaxPrior",
            "LossContext",
            "bce_loss_jax",
            "dice_loss_jax",
            "fit_jax",
            "loss_gradient_jax",
            "loss_terms_jax",
            "register_loss_jax",
            "soft_skeleton_jax",
        ]
    },
)
