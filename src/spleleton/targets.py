"""Target fields: what the rendering of a fit is compared with.

A fit compares the rendered segments with a field computed from the mask.
By default that field is the mask itself, but it can be any field in
``[0, 1]`` of the mask's shape. A :class:`TargetConfig` names a registered
target and its parameters; each target also declares the render mode it is
meant for: ``"tube"`` for fields that look like the mask (the segments'
tubes should fill them), ``"centreline"`` for fields peaked on the
centreline (a Gaussian line along each segment should match them).

Seeding and spawning keep using the mask; only the loss sees the target.

Registered targets (computed in NumPy, once per fit):

- ``mask`` (tube): the mask;
- ``smoothed_mask(sigma)`` (tube): the mask blurred by a Gaussian of
  ``sigma`` (physical units), which softens bumps and noise;
- ``opened_mask(radius)`` (tube): a morphological opening by a ball of
  ``radius`` (physical units), which removes protrusions thinner than the
  ball before the fit. It also shaves the tubes' stair-stepped surface: on
  the quick ``errors3d-v1`` volumes, radius 2 removes 39% of the bump volume
  of ``bumps-h1.5`` and 8% of the tubes' voxels, and breaks no tree (radius
  1.5 splits ``clean``; radius 1 removes 10% of the bumps);
- ``edt_ridge(sigma)`` (centreline): ``exp(-d^2 / 2 sigma^2)``, where ``d``
  is the distance to the ridge of the mask's distance transform (its local
  maxima);
- ``soft_skeleton(k)`` (centreline): the soft skeleton of Shit et al.
  (2021) after ``k`` iterations, the same as the ``cldice`` loss uses.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from scipy import ndimage

RenderMode = Literal["tube", "centreline"]


@dataclass(frozen=True)
class TargetConfig:
    """A target field, by name.

    Parameters
    ----------
    name : str
        The registered target.
    params : tuple of (str, value) pairs
        Its parameters (keyword arguments). Empty uses its defaults.
    """

    name: str = "mask"
    params: tuple[tuple[str, Any], ...] = ()


@dataclass(frozen=True)
class _Target:
    function: Callable[..., np.ndarray]
    render_mode: RenderMode


_TARGETS: dict[str, _Target] = {}


def register_target(
    name: str, render_mode: RenderMode
) -> Callable[[Callable], Callable]:
    """Register a target (a decorator).

    The function is called as ``function(mask, spacing, **params)`` with a
    boolean mask and the spacing as a tuple of floats, and returns a field of
    the mask's shape with values in ``[0, 1]``.
    """

    def register(function: Callable) -> Callable:
        if name in _TARGETS:
            raise ValueError(f"target {name!r} is already registered")
        _TARGETS[name] = _Target(function, render_mode)
        return function

    return register


def target_names() -> tuple[str, ...]:
    """The registered targets."""
    return tuple(_TARGETS)


def target_render_mode(config: TargetConfig) -> RenderMode:
    """The render mode a target is meant for."""
    return _lookup(config).render_mode


def target_field(
    config: TargetConfig, mask: np.ndarray, spacing: tuple[float, ...] | None = None
) -> np.ndarray:
    """Compute a target field from a mask.

    Parameters
    ----------
    config : TargetConfig
        The target.
    mask : np.ndarray
        Binary mask, 2D or 3D.
    spacing : tuple of float, optional
        Physical voxel size per axis.

    Returns
    -------
    np.ndarray
        The field, float32, of the mask's shape.
    """
    mask = np.asarray(mask, dtype=bool)
    spacing = (1.0,) * mask.ndim if spacing is None else tuple(map(float, spacing))
    field = _lookup(config).function(mask, spacing, **dict(config.params))
    return np.asarray(field, dtype=np.float32)


def _lookup(config: TargetConfig) -> _Target:
    try:
        return _TARGETS[config.name]
    except KeyError:
        raise ValueError(
            f"unknown target {config.name!r}; registered: {', '.join(_TARGETS)}"
        ) from None


@register_target("mask", "tube")
def _mask(mask: np.ndarray, spacing: tuple[float, ...]) -> np.ndarray:
    return mask.astype(np.float32)


@register_target("smoothed_mask", "tube")
def _smoothed_mask(
    mask: np.ndarray, spacing: tuple[float, ...], sigma: float = 1.0
) -> np.ndarray:
    sigmas = [sigma / step for step in spacing]
    return ndimage.gaussian_filter(mask.astype(np.float64), sigmas, mode="constant")


@register_target("opened_mask", "tube")
def _opened_mask(
    mask: np.ndarray, spacing: tuple[float, ...], radius: float = 2.0
) -> np.ndarray:
    return ndimage.binary_opening(mask, structure=_ball(radius, spacing))


@register_target("edt_ridge", "centreline")
def _edt_ridge(
    mask: np.ndarray, spacing: tuple[float, ...], sigma: float = 1.0
) -> np.ndarray:
    distance = ndimage.distance_transform_edt(mask, sampling=spacing)
    # the ridge: voxels of the mask where the distance is a local maximum
    ridge = mask & (distance >= ndimage.maximum_filter(distance, size=3))
    to_ridge = ndimage.distance_transform_edt(~ridge, sampling=spacing)
    return np.exp(-0.5 * (to_ridge / sigma) ** 2)


@register_target("soft_skeleton", "centreline")
def _soft_skeleton(
    mask: np.ndarray, spacing: tuple[float, ...], k: int = 10
) -> np.ndarray:
    return soft_skeleton(mask.astype(np.float32), k)


def soft_skeleton(image: np.ndarray, k: int = 10) -> np.ndarray:
    """The soft skeleton of Shit et al. (2021), in NumPy.

    As :func:`spleleton.fit.soft_skeleton_jax`: erosion is the minimum over
    3-voxel windows along each axis, dilation the maximum over a 3^D window,
    and each of ``k`` iterations erodes the image and adds what opening
    removes.
    """
    image = np.asarray(image, dtype=np.float32)
    ndim = image.ndim

    def erode(x: np.ndarray) -> np.ndarray:
        return np.min(
            [ndimage.minimum_filter1d(x, 3, axis=axis) for axis in range(ndim)],
            axis=0,
        )

    def open_(x: np.ndarray) -> np.ndarray:
        return ndimage.maximum_filter(erode(x), size=3)

    skeleton = np.maximum(image - open_(image), 0)
    for _ in range(k):
        image = erode(image)
        delta = np.maximum(image - open_(image), 0)
        skeleton = skeleton + np.maximum(delta - skeleton * delta, 0)
    return skeleton


def _ball(radius: float, spacing: tuple[float, ...]) -> np.ndarray:
    """A ball of a physical radius, as a boolean structuring element."""
    half = [int(np.floor(radius / step)) for step in spacing]
    grids = np.meshgrid(
        *[np.arange(-h, h + 1) * step for h, step in zip(half, spacing, strict=True)],
        indexing="ij",
    )
    return sum(g**2 for g in grids) <= radius**2 + 1e-9
