"""The probe: does the loss prefer the correct configuration (plan, Section 10).

The benchmark says *that* a pipeline failed; the probe says *why*: whether
the loss itself prefers the error, or the loss was right and the fit never
reached the fix. Along a controlled error path ε (ε = 0 is the relaxed
ground truth) it measures, per loss term, the **rigid margin** (only the
error's own parameters move), the **profile margin** (the error's segments
held at ε, every other free parameter re-fitted) and the **force** along ε
(``jax.jvp``), and classifies each event as restoring, flat, barrier or
attracting. Everything runs in float64 on exact crops of the volume.

- :mod:`._spec`: :class:`ProbeSpec` and :func:`probe_spec` (from a variant);
- :mod:`._terms`: the loss terms on the full volume and on crops (JAX);
- :mod:`._setup`: probed events, the relaxed reference, the error paths;
- :mod:`._crop`: exact crops and their buckets;
- :mod:`._probe`: probing one set-up, classes, and the exactness checks;
- :mod:`._run`: probing volumes, and the results on disk;
- :mod:`._analysis`: loss-correct rates, the weight map and the screen.

The probe needs JAX; import this package only where JAX is installed.
"""

from spleleton.bench.probe._analysis import (
    loss_correct_rates,
    screen,
    weight_map,
)
from spleleton.bench.probe._run import (
    ProbeResults,
    VolumeProbe,
    combine,
    load_probe,
    probe_directory,
    probe_key,
    probe_volume,
    run_probe,
)
from spleleton.bench.probe._spec import (
    EPS_GRID,
    FAMILIES,
    NEW_SLOT_FAMILIES,
    TARGET_FAMILIES,
    ProbeSpec,
    probe_spec,
)

__all__ = [
    "EPS_GRID",
    "FAMILIES",
    "NEW_SLOT_FAMILIES",
    "TARGET_FAMILIES",
    "ProbeResults",
    "ProbeSpec",
    "VolumeProbe",
    "combine",
    "load_probe",
    "loss_correct_rates",
    "probe_directory",
    "probe_key",
    "probe_spec",
    "probe_volume",
    "run_probe",
    "screen",
    "weight_map",
]
