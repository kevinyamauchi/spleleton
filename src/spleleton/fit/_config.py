"""Settings of the fit (shared by all backends)."""

import dataclasses
from dataclasses import dataclass, field

from spleleton.fit._losses import LossTerm, default_losses, loss_weight
from spleleton.render._config import RenderConfig


@dataclass(frozen=True)
class FitConfig:
    """Optimiser and loss settings.

    Parameters
    ----------
    n_iterations : int
        Number of Adam steps.
    position_lr : float
        Learning rate of segment endpoints, in physical units per step.
    radius_lr : float
        Learning rate of the log-radius.
    opacity_lr : float
        Learning rate of the opacity logit.
    losses : tuple of LossTerm
        The weighted terms of the loss (see :mod:`spleleton.fit._losses`).
        The default is BCE + Dice + sparsity with weights 1, 1 and 0.5
        (:func:`default_losses`). The sparsity term costs each segment its
        opacity times its own rendered volume, in voxels, divided by the
        number of voxels, so redundant segments are pushed towards zero
        opacity.
    render : RenderConfig
        Renderer settings.
    temperature : tuple of float, optional
        For the ``"softmax"`` union: the temperature at the first and last
        iteration, interpolated geometrically. ``None`` keeps
        ``render.temperature`` fixed.
    switch_to_max : float, optional
        Fraction of the iterations after which the union switches to
        ``"max"``. ``None`` never switches.
    """

    n_iterations: int = 500
    position_lr: float = 0.1
    radius_lr: float = 0.02
    opacity_lr: float = 0.05
    losses: tuple[LossTerm, ...] = default_losses()
    render: RenderConfig = field(default_factory=RenderConfig)
    temperature: tuple[float, float] | None = None
    switch_to_max: float | None = None

    def __post_init__(self) -> None:
        """Check that every term is named once."""
        names = [term.name for term in self.losses]
        if len(set(names)) != len(names):
            raise ValueError(f"loss terms must have distinct names, got {names}")

    @property
    def bce_weight(self) -> float:
        """The weight of the ``bce`` term (0 if none); for the torch fit."""
        return loss_weight(self.losses, "bce")

    @property
    def dice_weight(self) -> float:
        """The weight of the ``dice`` term (0 if none); for the torch fit."""
        return loss_weight(self.losses, "dice")

    @property
    def sparsity_weight(self) -> float:
        """The weight of the ``sparsity`` term (0 if none); for the torch fit."""
        return loss_weight(self.losses, "sparsity")

    def render_config(self, iteration: int) -> RenderConfig:
        """Renderer settings at a given iteration of the schedule."""
        progress = iteration / max(self.n_iterations - 1, 1)
        if self.switch_to_max is not None and progress >= self.switch_to_max:
            return dataclasses.replace(self.render, union="max")
        if self.temperature is not None:
            start, end = self.temperature
            return dataclasses.replace(
                self.render, temperature=start * (end / start) ** progress
            )
        return self.render
