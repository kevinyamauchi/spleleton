"""Where benchmark results live, how they are keyed, and what produced them.

**Roots.** Full runs write to ``bench/``, ``--quick`` runs to
``bench_quick/``, both next to the repository's ``pyproject.toml`` (or in
the working directory, outside a source checkout). ``SPLELETON_BENCH_DIR``
and ``SPLELETON_BENCH_QUICK_DIR`` override them. Nothing made while the code
is still changing can reach ``bench/``, because quick runs never write there.

**Cache keys** are SHA-256 hashes over canonical JSON (see
:mod:`spleleton._serialise`):

- fit: the sample's hash, the fit spec, ``FIT_VERSION`` and the numerics;
- graph: the fit key, the graph spec and ``GRAPH_VERSION``;
- score: the graph key and ``METRICS_VERSION``.

The numerics are the JAX and jaxlib versions and the dtype. Thread and
worker counts are left out: fits are bit-identical across them.

**Provenance** (git commit, dirty flag, the ``src/`` hash, versions, host)
is recorded with every result but is not part of any key. The ``src/`` hash
lets comparisons warn when results come from different code.
"""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import spleleton
from spleleton._serialise import canonical_json
from spleleton.bench import FIT_VERSION, GRAPH_VERSION, METRICS_VERSION

#: The package's source directory (``src/spleleton`` in a checkout).
PACKAGE_DIR = Path(spleleton.__file__).resolve().parent
#: Paths whose uncommitted changes make the tree dirty, relative to the repo.
DIRTY_PATHS = ("src", "scripts", "pyproject.toml")

_PACKAGES = (
    "numpy",
    "scipy",
    "scikit-image",
    "networkx",
    "jax",
    "jaxlib",
    "optax",
    "pandas",
    "pyarrow",
    "matplotlib",
)


def repo_root() -> Path | None:
    """The source checkout this package runs from, or ``None`` if installed."""
    candidate = PACKAGE_DIR.parents[1]
    return candidate if (candidate / "pyproject.toml").is_file() else None


@dataclass(frozen=True)
class Store:
    """A results root and the layout below it (plan, Section 6.2).

    Parameters
    ----------
    root : Path
        ``bench/`` or ``bench_quick/``.
    quick : bool
        Whether this is the quick root.
    """

    root: Path
    quick: bool = False

    @classmethod
    def default(cls, quick: bool = False) -> "Store":
        """The full root, or the quick one."""
        variable = "SPLELETON_BENCH_QUICK_DIR" if quick else "SPLELETON_BENCH_DIR"
        if variable in os.environ:
            return cls(Path(os.environ[variable]), quick)
        base = repo_root() or Path.cwd()
        return cls(base / ("bench_quick" if quick else "bench"), quick)

    # -- datasets -------------------------------------------------------------

    def split_dir(self, suite: str, subset: str, split: str) -> Path:
        """``datasets/<suite>/<subset>/<split>``."""
        return self.root / "datasets" / suite / subset / split

    def sample_path(self, suite: str, subset: str, split: str, seed: int) -> Path:
        """A sample's file."""
        return self.split_dir(suite, subset, split) / sample_name(seed)

    # -- caches and results -----------------------------------------------------

    def fit_path(self, key: str) -> Path:
        """``cache/fit/<key>.npz``, shared across runs."""
        return self.root / "cache" / "fit" / f"{key}.npz"

    def graph_path(self, key: str) -> Path:
        """``cache/graph/<key>.npz``, shared across runs."""
        return self.root / "cache" / "graph" / f"{key}.npz"

    @property
    def jax_cache(self) -> Path:
        """JAX's persistent compilation cache (holds no results)."""
        return self.root / "cache" / "jax"

    def run_dir(self, suite: str, variant: str, key: str) -> Path:
        """``runs/<suite>/<variant>__<key8>``."""
        return self.root / "runs" / suite / f"{variant}__{key[:8]}"

    def probe_dir(self, suite: str, probe: str, key: str) -> Path:
        """``probes/<suite>/<probe>__<key8>``."""
        return self.root / "probes" / suite / f"{probe}__{key[:8]}"

    def sensitivity_dir(self, name: str) -> Path:
        """``sensitivity/<name>``: validation runs and screening across runs."""
        return self.root / "sensitivity" / name

    @property
    def stages_dir(self) -> Path:
        """``stages/``: stage records and logs."""
        return self.root / "stages"

    def report_dir(self, name: str) -> Path:
        """``reports/<name>``."""
        return self.root / "reports" / name


def sample_name(seed: int) -> str:
    """File name of a sample."""
    return f"sample_{seed:04d}.npz"


# -- hashing and keys -------------------------------------------------------------


def array_hash(arrays: dict[str, np.ndarray]) -> str:
    """SHA-256 over decoded arrays: name, dtype, shape and bytes, sorted by name.

    Hashing the decoded arrays rather than the file bytes makes the hash
    independent of the zlib build that compressed the file.
    """
    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        digest.update(name.encode())
        digest.update(f"|{value.dtype.str}|{value.shape}|".encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def file_hash(path: Path) -> str:
    """The :func:`array_hash` of a saved ``.npz`` file's decoded arrays."""
    with np.load(path) as data:
        return array_hash({name: data[name] for name in data.files})


def key_hash(parts: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a dict of key parts."""
    return hashlib.sha256(canonical_json(parts).encode()).hexdigest()


def numerics(dtype: str = "float32") -> dict[str, str | None]:
    """The parts of the environment that fits' numbers depend on."""
    return {
        "jax": _version("jax"),
        "jaxlib": _version("jaxlib"),
        "dtype": dtype,
    }


def fit_key(sample_sha: str, fit_spec: Any, numerics: dict[str, Any]) -> str:
    """The cache key of a fit.

    Parameters
    ----------
    sample_sha : str
        The sample's hash (from its manifest).
    fit_spec : dataclass or dict
        Everything the fit depends on: pipeline, target, fit settings,
        initialisation, jitter and replicate index.
    numerics : dict
        See :func:`numerics`.
    """
    return key_hash(
        {
            "sample": sample_sha,
            "fit": fit_spec,
            "version": FIT_VERSION,
            "numerics": numerics,
        }
    )


def graph_key(fit_key: str, graph_spec: Any) -> str:
    """The cache key of a graph stage on a fit."""
    return key_hash({"fit": fit_key, "graph": graph_spec, "version": GRAPH_VERSION})


def score_key(graph_key: str) -> str:
    """The key of the scores of a graph stage's output."""
    return key_hash({"graph": graph_key, "version": METRICS_VERSION})


# -- provenance ---------------------------------------------------------------------


def src_hash() -> str:
    """SHA-256 over the package's ``.py`` files, sorted by path.

    Each file contributes its path relative to the package and its bytes, so
    a rename changes the hash too. ``__pycache__`` is excluded.
    """
    digest = hashlib.sha256()
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        digest.update(path.relative_to(PACKAGE_DIR).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_state() -> dict[str, Any]:
    """The commit, and whether the code has uncommitted changes.

    The tree is dirty when ``src/``, ``scripts/`` or ``pyproject.toml`` has
    uncommitted changes, untracked files included. Both are ``None`` outside
    a git checkout.
    """
    root = repo_root()

    def git(*args: str) -> str | None:
        if root is None:
            return None
        try:
            return subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain", "--untracked-files=all", "--", *DIRTY_PATHS)
    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
        "dirty_files": [] if not status else status.splitlines()[:20],
    }


class DirtyTree(RuntimeError):
    """A full stage was started with uncommitted changes to the code."""


def check_tree(full: bool) -> dict[str, Any]:
    """Check that the code is committed before a stage runs.

    A full stage refuses a dirty tree (:class:`DirtyTree`); a quick one warns.

    Returns
    -------
    dict
        The :func:`git_state`.
    """
    state = git_state()
    if state["dirty"]:
        message = (
            "uncommitted changes under src/, scripts/ or pyproject.toml: "
            + ", ".join(state["dirty_files"])
        )
        if full:
            raise DirtyTree(message + ". Commit them before running a full stage.")
        warnings.warn(message, stacklevel=2)
    return state


def versions() -> dict[str, str | None]:
    """Python and package versions, read without importing the packages."""
    return {
        "python": platform.python_version(),
        **{package: _version(package) for package in _PACKAGES},
    }


def provenance() -> dict[str, Any]:
    """What produced a result: code, versions, host and XLA settings."""
    return {
        **git_state(),
        "src_hash": src_hash(),
        "stage_versions": {
            "fit": FIT_VERSION,
            "graph": GRAPH_VERSION,
            "metrics": METRICS_VERSION,
        },
        "versions": versions(),
        "host": platform.node(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "xla_flags": os.environ.get("XLA_FLAGS"),
        "argv": sys.argv,
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def write_json(path: Path, data: Any) -> None:
    """Write JSON (NumPy values converted), atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=_to_jsonable) + "\n")
    temporary.replace(path)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
