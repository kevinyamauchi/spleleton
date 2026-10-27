"""Loss terms by name (JAX) and target fields."""

import dataclasses

import numpy as np
import pytest

from spleleton._serialise import from_dict, to_dict
from spleleton.fit import FitConfig, LossTerm
from spleleton.targets import (
    TargetConfig,
    soft_skeleton,
    target_field,
    target_names,
    target_render_mode,
)

# ------------------------------------------------------------------ configs


def test_default_losses_and_weights():
    config = FitConfig()
    assert config.losses == (
        LossTerm("bce", 1.0),
        LossTerm("dice", 1.0),
        LossTerm("sparsity", 0.5),
    )
    assert (config.bce_weight, config.dice_weight, config.sparsity_weight) == (
        1.0,
        1.0,
        0.5,
    )
    other = FitConfig(losses=(LossTerm("focal", 2.0, (("gamma", 1.0),)),))
    assert (other.bce_weight, other.sparsity_weight) == (0.0, 0.0)
    assert hash(other) == hash(FitConfig(losses=other.losses))
    assert from_dict(FitConfig, to_dict(other)) == other
    with pytest.raises(ValueError, match="distinct names"):
        FitConfig(losses=(LossTerm("bce"), LossTerm("bce", 2.0)))


# ------------------------------------------------------------------- losses


@pytest.fixture
def jax_losses():
    jax = pytest.importorskip("jax")
    from spleleton.fit import _fit_jax

    with jax.enable_x64(True):
        yield _fit_jax


def _context(module, occupancy, target):
    import jax.numpy as jnp

    return module.LossContext(
        jnp.asarray(occupancy), jnp.asarray(target), None, jnp.zeros(1)
    )


def _fields(seed=0, shape=(12, 14)):
    rng = np.random.default_rng(seed)
    occupancy = rng.uniform(0.01, 0.99, shape)
    target = (rng.uniform(size=shape) > 0.6).astype(float)
    return occupancy, target


def test_candidate_losses_reduce_to_known_ones(jax_losses):
    m = jax_losses
    occupancy, target = _fields()
    context = _context(m, occupancy, target)
    losses = m.LOSSES_JAX
    tversky = losses["tversky"](context, alpha=0.5, beta=0.5, eps=0.0)
    assert float(tversky) == pytest.approx(float(losses["dice"](context, eps=0.0)))
    assert float(losses["focal"](context, gamma=0.0)) == pytest.approx(
        float(losses["bce"](context)), rel=1e-12
    )
    mse = np.mean((occupancy - target) ** 2)
    assert float(losses["weighted_l2"](context, pos_weight=1.0)) == pytest.approx(mse)
    weighted = np.mean((1 + 9 * target) * (occupancy - target) ** 2)
    assert float(losses["weighted_l2"](context)) == pytest.approx(weighted)
    # tversky with beta > alpha penalises misses more than false positives
    miss = _context(m, np.zeros_like(target), target)
    extra = _context(m, np.ones_like(target), target)
    assert float(losses["tversky"](miss)) > float(losses["tversky"](extra))


@pytest.mark.parametrize("ndim", [2, 3])
def test_soft_skeleton_matches_numpy(jax_losses, ndim):
    rng = np.random.default_rng(ndim)
    image = rng.uniform(size=(10,) * ndim)
    found = np.asarray(jax_losses.soft_skeleton_jax(np.asarray(image), k=4))
    np.testing.assert_allclose(found, soft_skeleton(image, k=4), atol=1e-6)


def test_soft_skeleton_of_a_bar_is_its_middle():
    image = np.zeros((20, 20), dtype=np.float32)
    image[8:13, 2:18] = 1
    skeleton = soft_skeleton(image, k=5)
    assert skeleton[10, 4:16].min() > 0.9
    assert skeleton[8].max() == 0
    assert skeleton[12].max() == 0


def test_cldice(jax_losses):
    import jax

    m = jax_losses
    target = np.zeros((20, 20))
    target[8:13, 2:18] = 1
    perfect = _context(m, target, target)
    assert float(m.LOSSES_JAX["cldice"](perfect)) == pytest.approx(0.0, abs=1e-6)
    occupancy, _ = _fields(shape=(20, 20))
    worse = float(m.LOSSES_JAX["cldice"](_context(m, occupancy, target)))
    assert 0 < worse <= 1
    gradient = jax.grad(lambda o: m.LOSSES_JAX["cldice"](_context(m, o, target)))(
        occupancy
    )
    assert np.isfinite(np.asarray(gradient)).all()
    assert np.abs(np.asarray(gradient)).max() > 0


@pytest.mark.parametrize("ndim", [2, 3])
def test_centreline_kernel_volume(jax_losses, ndim):
    from spleleton.primitives import JaxLineSegments, SegmentArrays
    from spleleton.render import RenderConfig, render_dense_jax

    shape = (40,) * ndim
    start, end = np.full(ndim, 20.0), np.full(ndim, 20.0)
    start[0], end[0] = 12.0, 28.0
    segments = JaxLineSegments.from_arrays(
        SegmentArrays(start[None], end[None], radius=3.0, opacity=1.0 - 1e-9)
    )
    config = RenderConfig(mode="centreline", sigma=1.5)
    rendered = float(render_dense_jax(segments, shape, config=config).sum())
    volume = float(jax_losses.kernel_volumes_jax(segments, config)[0])
    assert volume == pytest.approx(rendered, rel=1e-3)
    tube = jax_losses.kernel_volumes_jax(segments, RenderConfig())
    np.testing.assert_allclose(tube, segments.tube_volumes())


def test_unknown_losses_are_refused():
    pytest.importorskip("jax")
    from spleleton.fit import fit_jax
    from spleleton.primitives import JaxLineSegments, SegmentArrays

    segments = JaxLineSegments.from_arrays(
        SegmentArrays(np.array([[2.0, 2.0]]), np.array([[8.0, 8.0]]))
    )
    with pytest.raises(ValueError, match="unknown loss terms"):
        fit_jax(segments, np.zeros((10, 10)), config=FitConfig(losses=(LossTerm("x"),)))


# ------------------------------------------------------------------ targets


def _tube_mask(bump=False):
    mask = np.zeros((40, 40, 40), dtype=bool)
    z, y, x = np.ogrid[:40, :40, :40]
    mask |= ((y - 20) ** 2 + (x - 20) ** 2 <= 9) & (z >= 5) & (z < 35)
    if bump:
        mask[20, 24, 20] = True  # a one-voxel protrusion
    return mask


def test_targets_are_registered_with_their_modes():
    assert set(target_names()) >= {
        "mask",
        "smoothed_mask",
        "opened_mask",
        "edt_ridge",
        "soft_skeleton",
    }
    modes = {name: target_render_mode(TargetConfig(name)) for name in target_names()}
    assert modes["mask"] == modes["smoothed_mask"] == modes["opened_mask"] == "tube"
    assert modes["edt_ridge"] == modes["soft_skeleton"] == "centreline"
    with pytest.raises(ValueError, match="unknown target"):
        target_field(TargetConfig("nope"), _tube_mask())
    config = TargetConfig("smoothed_mask", (("sigma", 2.0),))
    assert from_dict(TargetConfig, to_dict(config)) == config


@pytest.mark.parametrize(
    "name", ["mask", "smoothed_mask", "opened_mask", "edt_ridge", "soft_skeleton"]
)
def test_target_fields_are_in_range(name):
    mask = _tube_mask(bump=True)
    field = target_field(TargetConfig(name), mask)
    assert field.dtype == np.float32
    assert field.shape == mask.shape
    assert field.min() >= 0
    assert field.max() <= 1 + 1e-6
    if name == "mask":
        np.testing.assert_array_equal(field, mask)


def test_opened_mask_removes_a_protrusion_and_keeps_the_tube():
    mask = _tube_mask(bump=True)
    opened = target_field(TargetConfig("opened_mask"), mask)
    assert opened[20, 24, 20] == 0
    # the tube stays, but for the corner voxels of its stair-stepped surface
    tube = _tube_mask()[10:30]
    kept = opened[10:30].astype(bool)
    assert not (kept & ~tube).any()
    assert kept.sum() >= 0.8 * tube.sum()
    assert kept[:, 18:23, 20].all()


def test_centreline_targets_peak_on_the_centreline():
    mask = _tube_mask()
    ridge = target_field(TargetConfig("edt_ridge"), mask)
    assert ridge[10:30, 20, 20].min() == pytest.approx(1.0)
    assert ridge[20, 20, 23] < 0.05
    skeleton = target_field(TargetConfig("soft_skeleton"), mask)
    assert skeleton[10:30, 20, 20].min() > 0.9
    assert skeleton[20, 20, 23] == 0


def test_fit_stage_checks_the_render_mode():
    pytest.importorskip("jax")
    from spleleton.pipeline import ScratchFitConfig, scratch_fit_stage_jax
    from spleleton.render import RenderConfig

    mask = _tube_mask()[::2, ::2, ::2]
    config = ScratchFitConfig(fit=FitConfig(n_iterations=3), n_rounds=1)
    with pytest.raises(ValueError, match="centreline"):
        scratch_fit_stage_jax(mask, config=config, target=TargetConfig("edt_ridge"))
    centreline = dataclasses.replace(
        config,
        fit=dataclasses.replace(config.fit, render=RenderConfig(mode="centreline")),
    )
    fit = scratch_fit_stage_jax(
        mask, config=centreline, target=TargetConfig("edt_ridge")
    )
    assert len(fit.history) == 3
    assert np.isfinite(fit.history[-1]["loss"])
