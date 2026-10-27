"""Shared test fixtures, including the ``backend`` adapter.

Tests that take the ``backend`` fixture run once per installed backend
("torch" and "jax"); a missing backend is skipped. The adapter hides the
differences between the backends (array types, autograd, in-place vs
functional updates), so a test can be written once with NumPy inputs. JAX
tests run with 64-bit mode on, so float64 inputs stay float64 as in torch.
"""

import dataclasses
import importlib
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pytest

_PARAMETERS = ("start", "end", "log_radius", "opacity_logit")

#: Reason for skipping torch tests that the JAX-only benchmark changes break.
TORCH_NOT_IN_SYNC = (
    "the torch backend is not kept in sync with the JAX-only benchmark changes "
    "(plans/benchmark_plan_v2.md, Section 2); it is ported back afterwards"
)


@dataclasses.dataclass(frozen=True)
class Backend:
    """Backend-independent access to one backend, for tests."""

    name: str

    @property
    def LineSegments(self) -> type:
        """The backend's line segment class."""
        from spleleton import primitives

        return getattr(primitives, f"{self.name.capitalize()}LineSegments")

    def fn(self, module: str, name: str) -> Callable:
        """The backend's version of ``spleleton.<module>.<name>``."""
        return getattr(
            importlib.import_module(f"spleleton.{module}"), f"{name}_{self.name}"
        )

    def array(self, values: Any, dtype: str = "float64") -> Any:
        """A backend array (float64 by default)."""
        if self.name == "torch":
            import torch

            return torch.as_tensor(np.asarray(values), dtype=self.dtype(dtype))
        import jax.numpy as jnp

        return jnp.asarray(np.asarray(values), dtype=self.dtype(dtype))

    def dtype(self, name: str) -> Any:
        """The backend's floating-point type of the given name."""
        module = importlib.import_module(
            "torch" if self.name == "torch" else "jax.numpy"
        )
        return getattr(module, name)

    def asnumpy(self, values: Any) -> np.ndarray:
        """A backend array (or a pytree leaf) as NumPy."""
        if hasattr(values, "detach"):
            return values.detach().cpu().numpy()
        return np.asarray(values)

    def segments(
        self,
        start: Any,
        end: Any,
        radius: Any = 1.0,
        opacity: Any = 0.5,
        dtype: str = "float64",
    ) -> Any:
        """Backend segments from NumPy-like values (validated as in the API)."""

        def convert(value: Any) -> Any:
            return value if np.isscalar(value) else self.array(value, dtype)

        start, end = self.array(start, dtype), self.array(end, dtype)
        if self.name == "torch":
            return self.LineSegments(
                start, end, radius=convert(radius), opacity=convert(opacity)
            )
        return self.LineSegments.create(
            start, end, radius=convert(radius), opacity=convert(opacity)
        )

    def render(
        self,
        segments: Any,
        shape: tuple[int, ...],
        spacing: Any = None,
        config: Any = None,
    ) -> Any:
        """The backend's binned render (same arguments as ``render_torch``).

        Under ``jax.grad`` the bin capacity cannot be measured, so JAX then
        gets a table that always fits: every block, with every primitive.
        """
        if self.name == "jax":
            import jax

            from spleleton.render import BinCapacity, RenderConfig

            if isinstance(segments.start, jax.core.Tracer):
                block = (config or RenderConfig()).block_size
                n_blocks = int(np.prod([-(-n // block) for n in shape]))
                capacity = BinCapacity(n_blocks, max(len(segments), 1))
                return self.fn("render", "render")(
                    segments, shape, spacing, config, capacity=capacity
                )
        return self.fn("render", "render")(segments, shape, spacing, config)

    def render_dense(
        self, segments: Any, shape: tuple[int, ...], *args, **kwargs
    ) -> Any:
        """The backend's dense reference render."""
        return self.fn("render", "render_dense")(segments, shape, *args, **kwargs)

    def fit(self, segments: Any, target: Any, *args, **kwargs) -> tuple[Any, list]:
        """The backend's fit, as ``(segments, history)`` for both backends.

        ``target`` is NumPy-like. Torch updates the segments in place, so the
        returned segments are the same object.
        """
        target = self.array(target, dtype=str(segments.start.dtype).split(".")[-1])
        if self.name == "torch":
            history = self.fn("fit", "fit")(segments, target, *args, **kwargs)
            return segments, history
        return self.fn("fit", "fit")(segments, target, *args, **kwargs)

    def prior_terms(self, prior: Any, segments: Any, iteration: int = 0) -> dict:
        """A prior's terms for some segments, as at an iteration of a fit."""
        if self.name == "torch":
            return prior(segments, iteration)
        state = prior.update(segments, prior.init(segments), iteration)
        return prior(segments, state)

    def parameters(self, segments: Any) -> dict[str, np.ndarray]:
        """The segments' stored (unconstrained) parameters, as NumPy."""
        return {name: self.asnumpy(getattr(segments, name)) for name in _PARAMETERS}

    def grad(self, function: Callable, *inputs: Any) -> tuple[np.ndarray, ...]:
        """Gradient of a scalar function of backend arrays at NumPy inputs.

        ``function`` takes backend arrays (float64) and returns a scalar.
        """
        if self.name == "torch":
            import torch

            tensors = [
                torch.tensor(np.asarray(x), dtype=torch.float64, requires_grad=True)
                for x in inputs
            ]
            function(*tensors).backward()
            return tuple(
                np.zeros(t.shape) if t.grad is None else t.grad.numpy() for t in tensors
            )
        import jax

        arrays = [self.array(x) for x in inputs]
        grads = jax.grad(function, argnums=tuple(range(len(arrays))))(*arrays)
        return tuple(np.asarray(g) for g in grads)

    def segment_grads(self, function: Callable, segments: Any) -> dict[str, np.ndarray]:
        """Gradient of a scalar function of the segments, per stored parameter."""
        if self.name == "torch":
            segments.zero_grad()
            function(segments).backward()
            # a parameter the function does not use has no gradient (None)
            return {
                name: np.zeros(tuple(parameter.shape))
                if (parameter := getattr(segments, name)).grad is None
                else parameter.grad.numpy().copy()
                for name in _PARAMETERS
            }
        import jax

        grads = jax.grad(function)(segments)
        return {name: np.asarray(getattr(grads, name)) for name in _PARAMETERS}

    def optimise(
        self,
        loss: Callable,
        segments: Any,
        parameters: Sequence[str],
        learning_rate: float,
        n_steps: int,
    ) -> Any:
        """Minimise ``loss(segments)`` with Adam over some stored parameters.

        Returns the optimised segments (the same object for torch, which is
        updated in place).
        """
        if self.name == "torch":
            import torch

            optimizer = torch.optim.Adam(
                [getattr(segments, name) for name in parameters], lr=learning_rate
            )
            for _ in range(n_steps):
                optimizer.zero_grad()
                loss(segments).backward()
                optimizer.step()
            return segments
        import jax
        import optax

        mask = type(segments)(**{name: name in parameters for name in _PARAMETERS})
        optimizer = optax.masked(optax.adam(learning_rate), mask)
        state = optimizer.init(segments)

        @jax.jit
        def step(segments, state):
            grads = jax.grad(loss)(segments)
            updates, state = optimizer.update(grads, state, segments)
            return optax.apply_updates(segments, updates), state

        for _ in range(n_steps):
            segments, state = step(segments, state)
        return segments

    def check_gradient(
        self,
        function: Callable,
        inputs: Sequence[np.ndarray],
        step: float = 1e-6,
        atol: float = 1e-6,
    ) -> None:
        """Compare the gradient with central finite differences (float64)."""
        analytic = self.grad(function, *inputs)

        def value(*arrays: np.ndarray) -> float:
            return float(self.asnumpy(function(*(self.array(a) for a in arrays))))

        for index, (x, gradient) in enumerate(zip(inputs, analytic, strict=True)):
            numeric = np.zeros_like(x, dtype=float)
            for flat in range(x.size):
                shifted = [np.array(a, dtype=float) for a in inputs]
                shifted[index].flat[flat] += step
                upper = value(*shifted)
                shifted[index].flat[flat] -= 2 * step
                lower = value(*shifted)
                numeric.flat[flat] = (upper - lower) / (2 * step)
            np.testing.assert_allclose(gradient, numeric, atol=atol, rtol=1e-5)


@pytest.fixture(params=["torch", "jax"])
def backend(request: pytest.FixtureRequest):
    """Each installed backend; JAX with 64-bit mode on for the test."""
    pytest.importorskip(request.param)
    if request.param == "jax":
        import jax

        with jax.enable_x64(True):
            yield Backend("jax")
    else:
        yield Backend("torch")


@pytest.fixture
def both_backends():
    """The torch and JAX adapters together (JAX in 64-bit mode)."""
    pytest.importorskip("torch")
    pytest.importorskip("jax")
    import jax

    with jax.enable_x64(True):
        yield Backend("torch"), Backend("jax")
