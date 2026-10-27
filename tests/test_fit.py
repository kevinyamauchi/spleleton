import numpy as np
import pytest

from spleleton.fit import FitConfig
from spleleton.render import RenderConfig


def _target(backend, shape=(40, 40)):
    """The mask of a vertical tube of radius 4, as NumPy."""
    truth = backend.segments([[20.0, 8.0]], [[20.0, 32.0]], radius=4.0, opacity=0.999)
    return (backend.asnumpy(backend.render(truth, shape)) > 0.5).astype(float)


def _hidden_duplicate(backend):
    # the second segment lies inside the first's tube and never wins a voxel
    return backend.segments(
        [[20.0, 8.0], [20.0, 15.0]],
        [[20.0, 32.0], [20.0, 25.0]],
        radius=np.array([4.0, 1.5]),
    )


def test_losses(backend):
    bce = backend.fn("fit", "bce_loss")
    dice = backend.fn("fit", "dice_loss")
    target = backend.array([[0.0, 1.0], [1.0, 0.0]])
    half = backend.array(np.full((2, 2), 0.5))
    assert float(dice(target, target)) == pytest.approx(0.0, abs=1e-6)
    assert float(dice(1 - target, target)) == pytest.approx(1.0, abs=1e-6)
    assert float(bce(target, target)) < 1e-4
    assert float(bce(half, target)) == pytest.approx(0.6931, abs=1e-4)


@pytest.mark.parametrize("union", ["max", "soft", "softmax"])
def test_fit_recovers_a_displaced_tube(backend, union):
    target = _target(backend)
    segments = backend.segments([[23.0, 12.0]], [[18.0, 28.0]], radius=2.5)
    segments, history = backend.fit(
        segments,
        target,
        config=FitConfig(n_iterations=300, render=RenderConfig(union=union)),
    )
    assert history[-1]["loss"] < 0.5 * history[0]["loss"]
    # the mask's boundary lies half a voxel inside the rendered radius of 4
    np.testing.assert_allclose(backend.asnumpy(segments.start)[0], [20, 8], atol=0.5)
    np.testing.assert_allclose(backend.asnumpy(segments.end)[0], [20, 32], atol=0.5)
    radius = backend.asnumpy(segments.radius)
    assert 3.2 < radius.min() and radius.max() < 4.0
    assert backend.asnumpy(segments.opacity)[0] > 0.9


def test_fit_prunes_a_hidden_duplicate_with_the_max_union(backend):
    segments, _ = backend.fit(
        _hidden_duplicate(backend), _target(backend), config=FitConfig(n_iterations=200)
    )
    opacity = backend.asnumpy(segments.opacity)
    assert opacity[0] > 0.9
    assert opacity[1] < 0.05


def test_callback_and_history(backend):
    segments = backend.segments([[20.0, 8.0]], [[20.0, 32.0]])
    calls = []
    _, history = backend.fit(
        segments,
        _target(backend),
        config=FitConfig(n_iterations=5),
        callback=lambda iteration, _: calls.append(iteration),
    )
    assert calls == [0, 1, 2, 3, 4, 5]
    assert len(history) == 5
    assert set(history[0]) == {"loss", "bce", "dice", "sparsity"}
    assert all(isinstance(value, float) for value in history[0].values())


def test_render_schedule():
    config = FitConfig(
        n_iterations=11,
        render=RenderConfig(union="softmax"),
        temperature=(1.0, 0.01),
        switch_to_max=0.8,
    )
    assert config.render_config(0).temperature == pytest.approx(1.0)
    assert config.render_config(5).temperature == pytest.approx(0.1)
    assert config.render_config(5).union == "softmax"
    assert config.render_config(8).union == "max"
    assert FitConfig().render_config(3) == RenderConfig()


@pytest.mark.parametrize("shape", [(40, 40), (120, 120)])
def test_sparsity_does_not_depend_on_image_size(backend, shape):
    # the same tube with a hidden duplicate, in a small and a large image
    segments, _ = backend.fit(
        _hidden_duplicate(backend),
        _target(backend, shape),
        config=FitConfig(n_iterations=200),
    )
    opacity = backend.asnumpy(segments.opacity)
    assert opacity[0] > 0.9
    assert opacity[1] < 0.05


def test_torch_fit_updates_in_place():
    torch = pytest.importorskip("torch")
    from spleleton.fit import fit_torch
    from spleleton.primitives import TorchLineSegments

    segments = TorchLineSegments(
        torch.tensor([[20.0, 8.0]]), torch.tensor([[20.0, 30.0]])
    )
    before = segments.start.detach().clone()
    fit_torch(segments, torch.zeros(40, 40), config=FitConfig(n_iterations=3))
    assert not torch.equal(segments.start.detach(), before)


def test_jax_fit_returns_new_segments():
    pytest.importorskip("jax")
    from spleleton.fit import fit_jax
    from spleleton.primitives import JaxLineSegments

    segments = JaxLineSegments.create([[20.0, 8.0]], [[20.0, 30.0]])
    before = np.asarray(segments.start).copy()
    fitted, history = fit_jax(
        segments, np.zeros((40, 40)), config=FitConfig(n_iterations=3)
    )
    np.testing.assert_array_equal(segments.start, before)
    assert not np.array_equal(fitted.start, before)
    assert len(history) == 3
