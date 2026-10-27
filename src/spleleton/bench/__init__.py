"""Benchmarking skeletonisation variants (``plans/benchmark_plan_v2.md``).

The benchmark compares variants of the pipeline on immutable synthetic
suites. Its modules:

- :mod:`spleleton.bench.store`: where results live (``bench/`` and, for
  ``--quick`` runs, ``bench_quick/``), cache keys, provenance;
- :mod:`spleleton.bench.suites`: suite definitions, building them with hashed
  manifests, and loading samples through the manifests;
- :mod:`spleleton.bench.cases`: ``proto-cases``, hand-built volumes with
  known answers;
- :mod:`spleleton.bench.variants`: what is compared, the baselines,
  overrides and sweeps;
- :mod:`spleleton.bench.events`: the events of a sample (junction branches,
  bumps, edge windows, protrusions) with stable IDs and covariates, and the
  scores of a run's graphs: outcomes, errors with causes, sample rows;
- :mod:`spleleton.bench.decisions`: the decision quantities behind the
  outcomes (numpy);
- :mod:`spleleton.bench.stats`, :mod:`spleleton.bench.compare` and
  :mod:`spleleton.bench.collect`: paired statistics and the noise floor,
  comparisons of runs with their report, and Phase F's bundle and budget;
- :mod:`spleleton.bench.runner`: running variants (fit, graph, score; each
  cached), resumably, within per-sample limits;
- :mod:`spleleton.bench.stages`: the stages, their records and status;
- :mod:`spleleton.bench.probe`: the loss probe (JAX);
- :mod:`spleleton.bench.timing`, :mod:`spleleton.bench.plots` and
  :mod:`spleleton.bench.napari`: timing, figures and the viewer.

The **stage versions** below are part of every cache key. One is bumped by
hand when a change alters that stage's numbers on purpose (a new default, a
bug fix, a change to assembly); bumping it invalidates that stage and every
stage after it. A refactor, a docstring or a new option that is off by
default bumps nothing.
"""

#: Version of the fit stage's numbers.
FIT_VERSION = 1
#: Version of the graph stage's numbers (1 includes the ``_drop_contained`` fix).
GRAPH_VERSION = 1
#: Version of the scores (samples, events and errors tables).
METRICS_VERSION = 1

__all__ = ["FIT_VERSION", "GRAPH_VERSION", "METRICS_VERSION", "load"]


def load(runs):
    """The runs' merged ``samples``, ``events`` and ``errors`` tables.

    See :func:`spleleton.bench.compare.load`.
    """
    from spleleton.bench.compare import load as _load

    return _load(runs)
