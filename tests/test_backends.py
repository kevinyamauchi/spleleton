import subprocess
import sys

import numpy as np
import pytest

import spleleton.fit
from spleleton._backends import has_jax, has_torch
from spleleton.primitives import SegmentArrays, as_segment_arrays


def test_unknown_names_raise_attribute_error():
    with pytest.raises(AttributeError):
        _ = spleleton.fit.not_a_function
    assert "fit_torch" in dir(spleleton.fit)


def test_neutral_imports_load_no_backend():
    code = (
        "import sys\n"
        "import spleleton.assemble, spleleton.fit, spleleton.geometry, "
        "spleleton.initialise, spleleton.pipeline, spleleton.primitives, "
        "spleleton.priors, spleleton.recording, spleleton.render, spleleton.seeding\n"
        "assert 'torch' not in sys.modules and 'jax' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


_INSTALLED = {"torch": has_torch, "jax": has_jax}
_NAMES = {"torch": "fit_torch", "jax": "JaxLineSegments"}


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_missing_backend_says_which_extra(backend):
    if _INSTALLED[backend]():
        pytest.skip(f"{backend} is installed")
    package = "fit" if backend == "torch" else "primitives"
    module = __import__(f"spleleton.{package}", fromlist=["_"])
    with pytest.raises(ImportError, match=rf"spleleton\[{backend}\]"):
        getattr(module, _NAMES[backend])


@pytest.mark.skipif(not has_torch(), reason="needs torch")
def test_torch_names_load_lazily():
    from spleleton.fit import fit_torch
    from spleleton.fit._fit_torch import fit_torch as direct

    assert fit_torch is direct


@pytest.mark.skipif(not has_jax(), reason="needs jax")
def test_jax_names_load_lazily():
    from spleleton.primitives import JaxLineSegments
    from spleleton.primitives._primitives_jax import JaxLineSegments as direct

    assert JaxLineSegments is direct


def test_availability_checks_return_bools():
    assert isinstance(has_torch(), bool)
    assert isinstance(has_jax(), bool)


def test_segment_arrays_broadcast_and_validate():
    arrays = SegmentArrays(np.zeros((3, 2)), np.ones((3, 2)), radius=2.0, opacity=0.5)
    assert arrays.radius.shape == (3, 2)
    assert arrays.opacity.shape == (3,)
    assert arrays.ndim == 2
    assert as_segment_arrays(arrays) is arrays
    assert len(arrays.select(np.array([True, False, True]))) == 2
    with pytest.raises(ValueError):
        SegmentArrays(np.zeros((3, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError):
        SegmentArrays(np.zeros((3, 2)), np.ones((3, 2)), radius=np.ones(4))
    with pytest.raises(TypeError):
        as_segment_arrays(np.zeros((3, 2)))


def test_segment_arrays_from_graph():
    from spleleton.graph import SkeletonGraph

    nodes = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 5.0]])
    paths = [np.array([[0.0, 0.0], [5.0, 1.0], [10.0, 0.0]]), nodes[[1, 2]]]
    graph = SkeletonGraph(
        nodes, [[0, 1], [1, 2]], paths, [np.array([3.0, 2.0, 1.0]), np.array([1, 1.0])]
    )
    arrays = SegmentArrays.from_graph(graph, opacity=0.7)
    assert len(arrays) == 3
    np.testing.assert_allclose(arrays.radius, [[3, 2], [2, 1], [1, 1]])
    np.testing.assert_allclose(arrays.opacity, 0.7)
    empty = SegmentArrays.from_graph(SkeletonGraph(np.empty((0, 3)), [], [], []))
    assert empty.start.shape == (0, 3)
