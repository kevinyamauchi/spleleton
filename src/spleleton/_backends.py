"""Optional array backends (PyTorch and JAX): availability and lazy exports.

Each backend is an optional extra (``pip install 'spleleton[torch]'`` or
``'spleleton[jax]'``). Packages with backend code (e.g. :mod:`spleleton.fit`)
import their backend-neutral names eagerly and their backend names lazily,
through a module ``__getattr__`` built by :func:`lazy_exports`:

- importing the package works with no backend installed;
- a backend is imported only when one of its names is first used, so a torch
  user never pays JAX's import time and vice versa;
- using a name whose backend is missing raises an :class:`ImportError` that
  says which extra to install.
"""

import importlib
import importlib.util
import sys
from collections.abc import Callable

#: Top-level packages each backend needs.
REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "torch": ("torch",),
    "jax": ("jax", "optax"),
}


def has_torch() -> bool:
    """Whether the PyTorch backend's dependencies are installed."""
    return _installed("torch")


def has_jax() -> bool:
    """Whether the JAX backend's dependencies are installed."""
    return _installed("jax")


def _installed(backend: str) -> bool:
    return all(
        importlib.util.find_spec(name) is not None for name in REQUIREMENTS[backend]
    )


def lazy_exports(
    package: str, **backends: dict[str, list[str]]
) -> tuple[Callable[[str], object], Callable[[], list[str]]]:
    """Module ``__getattr__`` and ``__dir__`` that import backend names lazily.

    Parameters
    ----------
    package : str
        The package's ``__name__``; submodules are resolved relative to it.
    **backends : dict of str to list of str
        For each backend (``torch=...``, ``jax=...``), a mapping from a
        relative submodule name (e.g. ``"._fit_torch"``) to the public names
        it provides.

    Returns
    -------
    __getattr__, __dir__ : callable
        To be assigned to the package's module-level names of the same name.

    Examples
    --------
    >>> __getattr__, __dir__ = lazy_exports(  # doctest: +SKIP
    ...     __name__, torch={"._fit_torch": ["fit_torch"]}
    ... )
    """
    owners = {
        name: (backend, submodule)
        for backend, modules in backends.items()
        for submodule, names in modules.items()
        for name in names
    }

    def __getattr__(name: str) -> object:
        if name not in owners:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        backend, submodule = owners[name]
        try:
            module = importlib.import_module(submodule, package)
        except ModuleNotFoundError as error:
            # only a missing backend dependency; a broken import still raises
            if error.name not in REQUIREMENTS[backend]:
                raise
            raise ImportError(
                f"{package}.{name} needs the {backend} backend: "
                f"pip install 'spleleton[{backend}]'"
            ) from error
        value = getattr(module, name)
        setattr(sys.modules[package], name, value)
        return value

    def __dir__() -> list[str]:
        return sorted(set(vars(sys.modules[package])) | set(owners))

    return __getattr__, __dir__
