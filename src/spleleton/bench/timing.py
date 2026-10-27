"""Timing: wall and CPU time, JAX compilation, and the host's state.

Ported from stage 0's ``scripts/proto/_timing.py``. :class:`Stopwatch`
measures a block: wall and CPU seconds, the JAX compilation seconds and the
number of programs compiled in it (from a ``jax.monitoring`` listener,
installed on first use), and the host's power source and load, so that
throttled or contended timings can be told apart (plan, Section 6.5).
:func:`enable_compilation_cache` turns on JAX's persistent compilation cache,
which every benchmark process uses (it holds no results).
"""

import collections
import os
import re
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

#: The JAX compilation phases (seconds each), as ``jax.monitoring`` reports them.
_COMPILE_EVENTS = (
    "/jax/core/compile/jaxpr_trace_duration",
    "/jax/core/compile/jaxpr_to_mlir_module_duration",
    "/jax/core/compile/backend_compile_duration",
)

_compile: collections.Counter = collections.Counter()
_listening = False
_host_cache: tuple[float, dict] = (-1e9, {})


def compile_totals() -> tuple[float, int]:
    """JAX's compilation seconds, and programs compiled, since listening began.

    The listener is installed by the first call after JAX is imported.
    """
    _listen()
    return float(_compile["seconds"]), int(_compile["programs"])


@contextmanager
def stopwatch() -> Iterator[dict[str, Any]]:
    """Time a block; the dict it yields is filled in when the block ends.

    Keys: ``wall_s``, ``cpu_s`` (this process's user and system time),
    ``compile_s`` and ``programs`` (JAX compilation in the block), and
    ``host`` (:func:`host_state` at the end). The block may add its own keys.
    """
    compile_before = compile_totals()
    record: dict[str, Any] = {}
    wall = time.perf_counter()
    cpu = _cpu_seconds()
    try:
        yield record
    finally:
        compile_after = compile_totals()
        record.update(
            wall_s=time.perf_counter() - wall,
            cpu_s=_cpu_seconds() - cpu,
            compile_s=compile_after[0] - compile_before[0],
            programs=compile_after[1] - compile_before[1],
            host=host_state(),
        )


def enable_compilation_cache(directory: Path) -> None:
    """Use JAX's persistent compilation cache in ``directory``.

    Compiled programs are reused across processes. The cache holds no
    results and is not part of any key.
    """
    import jax

    directory.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(directory))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.5)


def host_state() -> dict[str, Any]:
    """Power source, CPU speed limit and 1-minute load (power cached for 30 s)."""
    global _host_cache
    stamp, state = _host_cache
    if time.monotonic() - stamp < 30:
        return {**state, "load_1m": os.getloadavg()[0]}
    battery = _pmset("batt")
    therm = _pmset("therm")
    power = None
    if battery is not None:
        match = re.search(r"drawing from '([^']+)'", battery)
        power = match.group(1) if match else None
    limit = None
    if therm is not None:
        match = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", therm)
        limit = int(match.group(1)) if match else None
    state = {
        "power": power,
        "on_mains": None if power is None else power == "AC Power",
        "cpu_speed_limit": limit,
        "load_1m": os.getloadavg()[0],
    }
    _host_cache = (time.monotonic(), state)
    return state


def _listen() -> None:
    global _listening
    if _listening or "jax" not in sys.modules:
        return
    import jax

    def listener(event: str, duration: float, **kwargs) -> None:
        if event in _COMPILE_EVENTS:
            _compile["seconds"] += duration
        if event == _COMPILE_EVENTS[-1]:
            _compile["programs"] += 1

    jax.monitoring.register_event_duration_secs_listener(listener)
    _listening = True


def _cpu_seconds() -> float:
    return time.process_time()


def _pmset(what: str) -> str | None:
    if sys.platform != "darwin":
        return None
    try:
        return subprocess.run(
            ["pmset", "-g", what],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
