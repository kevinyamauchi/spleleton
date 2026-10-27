"""Loss terms of the fit, by name (shared by all backends).

A fit's loss is a weighted sum of named terms, :class:`LossTerm`, plus its
regularisers. The terms are computed by functions registered by name with
the backend (:func:`spleleton.fit.register_loss_jax`). The default,
:func:`default_losses`, is BCE + Dice + sparsity.

Terms registered for JAX:

- ``bce``: mean binary cross-entropy, ``eps`` clamps the occupancy;
- ``dice``: one minus the soft Dice coefficient;
- ``sparsity``: each segment's opacity times its rendered volume (the
  kernel's: a capsule for ``"tube"``, a Gaussian line for ``"centreline"``),
  in voxels, divided by the number of voxels. Volumes are not differentiated;
- ``weighted_l2``: mean squared error, with foreground voxels weighted by
  ``pos_weight``;
- ``focal``: the focal loss, BCE with each voxel down-weighted by ``(1 -
  p_t)^gamma``;
- ``tversky``: one minus the Tversky index, ``alpha`` weighting false
  positives and ``beta`` false negatives (``0.5, 0.5`` is Dice);
- ``cldice``: one minus the clDice score of the rendering against the target,
  with ``k`` iterations of the soft skeleton.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class LossTerm:
    """One weighted term of a fit's loss.

    Frozen and hashable, so that a jitted step can take it as a static
    argument.

    Parameters
    ----------
    name : str
        The term's registered name; it also names the term in the history.
    weight : float
        Its weight in the total loss.
    params : tuple of (str, value) pairs
        Keyword arguments of the term's function. Pairs, not a dict, so that
        the term stays hashable.
    """

    name: str
    weight: float = 1.0
    params: tuple[tuple[str, Any], ...] = ()


def default_losses(
    bce: float = 1.0, dice: float = 1.0, sparsity: float = 0.5
) -> tuple[LossTerm, ...]:
    """BCE + Dice + sparsity, with these weights (the default loss)."""
    return (
        LossTerm("bce", bce),
        LossTerm("dice", dice),
        LossTerm("sparsity", sparsity),
    )


def loss_weight(losses: tuple[LossTerm, ...], name: str) -> float:
    """The weight of the term of this name, or 0 if there is none."""
    return sum(term.weight for term in losses if term.name == name)
