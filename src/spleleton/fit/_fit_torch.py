"""Fit primitives to a target image by differentiable rendering, in PyTorch.

See :mod:`spleleton.fit` for the loss.
"""

from collections.abc import Callable, Sequence

import numpy as np
import torch

from spleleton.fit._config import FitConfig
from spleleton.primitives._primitives_torch import TorchLineSegments
from spleleton.render._render_torch import render_torch


def bce_loss_torch(
    occupancy: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Mean binary cross-entropy, with the occupancy clamped away from 0 and 1."""
    occupancy = occupancy.clamp(eps, 1 - eps)
    return -(
        target * torch.log(occupancy) + (1 - target) * torch.log1p(-occupancy)
    ).mean()


def dice_loss_torch(
    occupancy: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """One minus the soft Dice coefficient."""
    overlap = (occupancy * target).sum()
    return 1 - (2 * overlap + eps) / (occupancy.sum() + target.sum() + eps)


def fit_torch(
    primitives: TorchLineSegments,
    target: torch.Tensor,
    spacing: tuple[float, ...] | None = None,
    config: FitConfig | None = None,
    callback: Callable[[int, TorchLineSegments], None] | None = None,
    regularisers: Sequence[
        Callable[[TorchLineSegments, int], dict[str, torch.Tensor]]
    ] = (),
) -> list[dict[str, float]]:
    """Optimise primitives in place so that their rendering matches a target.

    Parameters
    ----------
    primitives : TorchLineSegments
        Initial primitives; updated in place.
    target : torch.Tensor
        Target image in ``[0, 1]`` (e.g. a binary mask), on the primitives'
        device.
    spacing : tuple of float, optional
        Physical voxel size per axis.
    config : FitConfig, optional
        Optimiser, loss and schedule settings.
    callback : callable, optional
        Called as ``callback(iteration, primitives)`` before every step and
        once after the last one (with ``iteration == n_iterations``).
    regularisers : sequence of callable
        Extra loss terms, each called as ``regulariser(primitives,
        iteration)`` and returning a dictionary of already weighted terms.

    Returns
    -------
    list of dict
        Per iteration: the total loss and each of its terms.
    """
    config = config or FitConfig()
    target = target.to(primitives.start.dtype)
    optimizer = torch.optim.Adam(
        [
            {"params": [primitives.start, primitives.end], "lr": config.position_lr},
            {"params": [primitives.log_radius], "lr": config.radius_lr},
            {"params": [primitives.opacity_logit], "lr": config.opacity_lr},
        ]
    )
    voxel_volume = 1.0 if spacing is None else float(np.prod(spacing))
    history = []
    for iteration in range(config.n_iterations):
        with torch.no_grad():
            volumes = primitives.tube_volumes() / voxel_volume
        if callback is not None:
            callback(iteration, primitives)
        optimizer.zero_grad()
        occupancy = render_torch(
            primitives, tuple(target.shape), spacing, config.render_config(iteration)
        )
        terms = {
            "bce": bce_loss_torch(occupancy, target),
            "dice": dice_loss_torch(occupancy, target),
            "sparsity": (primitives.opacity * volumes).sum() / target.numel(),
        }
        loss = (
            config.bce_weight * terms["bce"]
            + config.dice_weight * terms["dice"]
            + config.sparsity_weight * terms["sparsity"]
        )
        for regulariser in regularisers:
            for name, term in regulariser(primitives, iteration).items():
                terms[name] = term
                loss = loss + term
        loss.backward()
        optimizer.step()
        history.append({"loss": loss.item(), **{k: v.item() for k, v in terms.items()}})
    if callback is not None:
        callback(config.n_iterations, primitives)
    return history
