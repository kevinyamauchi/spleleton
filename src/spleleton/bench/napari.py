"""The benchmark in napari (plan, Section 9.1).

**Run mode** (``bench view RUN --subset S --seed N``): :func:`run_view`
gathers one sample of a run (its masks, graphs, fitted segments, their
rendering made on demand with JAX, its errors and events), and
:func:`view_run` opens it:

- the mask, the clean mask (hidden), the target field when it is not the
  mask (hidden), and the rendering of the fitted segments (hidden);
- the fitted segments as lines coloured by opacity;
- the raw graph (orange, hidden), the clean graph (cyan) and the true graph
  (red);
- the corruption markers;
- the errors as points coloured by type (discs for the clean graph,
  squares for the raw one), with type, cause, stage and size as features;
- the **error browser**, a dock widget listing the sample's errors and
  events, filterable by kind, type, cause, stage and status. Selecting a row
  centres the 3D camera on it; ``n`` and ``p`` step to the next and previous
  row. The zoom is relative to the view's own, since napari's 3D zoom is in
  pixels per world unit.

With ``compare``, the second run's graphs are added in other colours, its
errors are matched to the first's (:func:`spleleton.bench.compare.match_errors`)
and shown by status (**fixed**: only in the first run; **new**: only in the
second; **shared**), and the flipped events are listed with their null flip
rate. :func:`record_sample` re-runs one sample's fit with a
:class:`~spleleton.recording.FitRecorder` (``--record``), checks it against
the stored fit, and saves the recording for ``view_fit``.
:func:`check_view` and :func:`anisotropic_check` are the automatic checks of
the coordinates (step 0.9), and :func:`screenshots` writes an overview and
the first error up close (the window must be shown: on macOS a hidden canvas
draws nothing).

With ``sensitivity`` (``--sensitivity PARAM``), the events with an implicit
derivative for that parameter are drawn as points coloured by their flip
distance (:func:`add_sensitivity_layer`; Section 11.7).

**Probe mode** (Section 10.8): :func:`view_probe` shows one probed
(event, family, opacity) on its volume, with a slider over ε. At each ε:

- the mask (and the target field, when it is not the mask);
- the relaxed reference's segments, and the event's segments at ε (the
  moved ones highlighted);
- the force of each loss term on the free endpoints: minus its gradient,
  one vectors layer per term, on a common scale;
- the per-voxel attribution: the total loss's gradient with respect to the
  crop's occupancy (where adding occupancy would raise the loss, and where
  it would lower it).

:func:`probe_event_view` builds the same layers as plain data, without
napari, so that they can be checked.
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np

from spleleton.bench.probe._crop import background
from spleleton.bench.probe._probe import PlacedCrop, place_crop
from spleleton.bench.probe._run import restore_volume
from spleleton.bench.probe._setup import (
    Padded,
    Reference,
    Setup,
    config,
    make_setup,
    padded,
)
from spleleton.bench.probe._spec import ProbeSpec
from spleleton.bench.probe._terms import attribution, segment_gradients, stack
from spleleton.bench.store import Store

#: Layer data: ``(data, kwargs, layer_type)``, as napari's layer tuples.
LayerTuple = tuple[Any, dict[str, Any], str]
#: The longest force vector drawn, in physical units.
FORCE_LENGTH = 4.0
MOVED_COLOUR = "#eb6834"
FREE_COLOUR = "#2a78d6"
REFERENCE_COLOUR = "#8a8984"


@dataclass
class ProbeEventView:
    """One probed (event, family, opacity), ready to evaluate at any ε."""

    reference: Reference
    setup: Setup
    spec: ProbeSpec
    arrays: Padded
    placed: PlacedCrop

    @property
    def eps(self) -> tuple[float, ...]:
        """The family's ε grid."""
        return self.setup.eps

    def layers(self, eps: float) -> list[LayerTuple]:
        """Every layer at ``eps``."""
        reference, setup = self.reference, self.setup
        spacing = np.asarray(reference.spacing)
        sample = reference.sample
        out: list[LayerTuple] = [
            (
                sample.mask.astype(np.uint8),
                {"name": "mask", "scale": spacing, "opacity": 0.2, "colormap": "gray"},
                "image",
            )
        ]
        if self.spec.target.name != "mask":
            out.append(
                (
                    reference.target,
                    {"name": f"target ({self.spec.target.name})", "scale": spacing,
                     "opacity": 0.4, "visible": False},
                    "image",
                )
            )  # fmt: skip
        raw = reference.raw
        out.append(
            (
                _vectors(raw["start"], raw["end"]),
                {
                    "name": "reference",
                    "edge_color": REFERENCE_COLOUR,
                    "edge_width": 0.4,
                },
                "vectors",
            )
        )
        c = config(self.arrays, setup, eps)
        live = c["live"]
        moved = np.zeros(len(live), bool)
        moved[[slot for slot, _ in setup.moved]] = True
        colours = np.where(moved[live], MOVED_COLOUR, FREE_COLOUR)
        out.append(
            (
                _vectors(c["start"][live], c["end"][live]),
                {"name": f"event at ε = {eps:g}", "edge_color": list(colours),
                 "edge_width": 0.8},
                "vectors",
            )
        )  # fmt: skip
        st = self.placed.static(reference)
        with jax.enable_x64(True):
            free, alive = stack([c])
            free = jax.tree.map(lambda x: x[0], free)
            alive = alive[0]
            gradients = segment_gradients(free, alive, self.placed.crop, st)
            gradients = jax.tree.map(np.asarray, gradients)
            occupancy_gradient = np.asarray(
                attribution(free, alive, self.placed.crop, st)
            )
        points = np.concatenate([c["start"][live], c["end"][live]])
        forces = {
            name: -np.concatenate([np.asarray(g.start)[live], np.asarray(g.end)[live]])
            for name, g in gradients.items()
        }
        largest = max(
            (
                float(np.linalg.norm(f, axis=1).max(initial=0.0))
                for f in forces.values()
            ),
            default=0.0,
        )
        scale = FORCE_LENGTH / largest if largest > 0 else 1.0
        for index, (name, force) in enumerate(forces.items()):
            out.append(
                (
                    np.stack([points, force * scale], axis=1),
                    {
                        "name": f"force: {name}",
                        "edge_color": _term_colour(index, name),
                        "edge_width": 0.5,
                        "visible": name == "total",
                        "features": {"magnitude": np.linalg.norm(force, axis=1)},
                    },
                    "vectors",
                )
            )
        limit = float(np.abs(occupancy_gradient).max()) or 1.0
        out.append(
            (
                occupancy_gradient,
                {
                    "name": "attribution (dL/d occupancy)",
                    "scale": spacing,
                    "translate": np.asarray(self.placed.origin) * spacing,
                    "colormap": "bwr",
                    "contrast_limits": (-limit, limit),
                    "opacity": 0.6,
                    "visible": False,
                },
                "image",
            )
        )
        return out


def _vectors(start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Napari vectors data, ``(N, 2, D)``: position and projection."""
    return np.stack([start, end - start], axis=1)


def _term_colour(index: int, name: str) -> str:
    from spleleton.bench.plots import CATEGORICAL, TEXT

    return TEXT if name == "total" else CATEGORICAL[index % len(CATEGORICAL)]


def probe_event_view(
    directory: Path,
    store: Store,
    subset: str,
    seed: int,
    event_id: str,
    family: str,
    opacity: float | None = None,
) -> ProbeEventView:
    """Rebuild one probed (event, family, opacity) from a probe's directory."""
    reference, chosen, spec = restore_volume(Path(directory), store, subset, seed)
    matches = [e for e in chosen if e.event_id == event_id and e.family == family]
    if not matches:
        probed = sorted({(e.event_id, e.family) for e in chosen})
        raise KeyError(f"{event_id} {family} was not probed; probed: {probed}")
    event = matches[0]
    if family in ("bump_spur", "duplicate") and opacity is None:
        opacity = 1.0
    with jax.enable_x64(True):
        setup = make_setup(reference, event, spec, opacity)
        back = background(reference, setup)
        arrays = padded(setup, setup.n_exact)
        configs = [config(arrays, setup, e) for e in setup.eps]
        placed = place_crop(reference, back, spec, configs, setup.n_exact, False)
    return ProbeEventView(reference, setup, spec, arrays, placed)


def view_probe(view: ProbeEventView, show: bool = True):
    """Open napari on a probed event, with a slider over ε.

    Returns
    -------
    napari.Viewer
    """
    import napari
    from magicgui import magicgui

    viewer = napari.Viewer(ndisplay=3, show=show)
    eps = list(view.eps)
    for data, kwargs, kind in view.layers(eps[0]):
        getattr(viewer, f"add_{kind}")(data, **kwargs)
    event = view.setup.event
    viewer.title = f"probe: {event.event_id} {event.family}" + (
        f" opacity {view.setup.opacity}" if view.setup.opacity is not None else ""
    )

    from magicgui.widgets import Label

    caption = Label(value=f"ε = {eps[0]:g}")

    @magicgui(
        auto_call=True,
        index={
            "widget_type": "Slider",
            "min": 0,
            "max": len(eps) - 1,
            "label": "ε step",
        },
    )
    def choose(index: int = 0) -> None:
        layers = view.layers(eps[index])
        dynamic = [lay for lay in viewer.layers if lay.name.startswith("event at")]
        for layer in dynamic:
            viewer.layers.remove(layer)
        for data, kwargs, kind in layers:
            if kwargs["name"].startswith("event at"):
                getattr(viewer, f"add_{kind}")(data, **kwargs)
            elif kwargs["name"] in viewer.layers and (
                kind != "image" or kwargs["name"].startswith("attribution")
            ):
                viewer.layers[kwargs["name"]].data = data
        caption.value = f"ε = {eps[index]:g}"

    choose.append(caption)
    viewer.window.add_dock_widget(choose, area="right", name="ε")
    return viewer


# == run mode ==========================================================================

#: Error colours by type, and symbols by stage.
ERROR_COLOURS = {
    "spur_end": "gold",
    "break_end": "orangered",
    "missed": "darkorange",
    "false_positive": "violet",
    "off_centre": "deepskyblue",
    "junction_missed": "white",
    "junction_extra": "springgreen",
}
STAGE_SYMBOLS = {"clean": "disc", "raw": "square"}
#: Compare mode: colours of the error statuses.
STATUS_COLOURS = {"fixed": "#1baf7a", "new": "#eb6834", "shared": "#8a8984"}
#: ``--record``: a snapshot every this many iterations (a recording is ~9 MB).
RECORD_EVERY = 30
#: The camera's zoom on a browsed row, relative to the first view.
ZOOM = 4.0
#: Checks: a point "sits on" what it names within this distance; the true
#: centreline is aligned when this share of it lies in the clean mask, and
#: misaligned (a layer without its scale) below the second share.
ON, ALIGNED, MISALIGNED = 1e-6, 0.99, 0.8
#: Opaque fitted segments at least this thick lie where their rendering >= 0.5.
OPAQUE, THICK = 0.5, 1.0


@dataclass
class RunView:
    """One sample of a run, with what the viewer shows (no napari needed).

    Attributes
    ----------
    name : str
        The run's name.
    subset, seed : str, int
        The sample.
    sample : Sample
    raw, clean : SkeletonGraph
        The run's graphs.
    segments : SegmentArrays, optional
        The fitted segments (``None`` for variants that fit nothing).
    rendering : np.ndarray, optional
        Their rendering with the fit's last render config.
    target : np.ndarray, optional
        The fitted field, when it is not the mask.
    errors, events : pandas.DataFrame
        The sample's rows of the run's tables.
    other : RunView, optional
        The compared run's view of the same sample.
    matched : pandas.DataFrame, optional
        With ``other``: every error of either run with its ``status``.
    flipped : pandas.DataFrame, optional
        With ``other``: the events whose outcome flipped, with the null flip
        rate of their type.
    sensitivity : pandas.DataFrame, optional
        With a parameter (``--sensitivity``): the sample's events with an
        implicit derivative for it, their positions, ``q``, ``dq`` and flip
        distance (Section 11.7).
    """

    name: str
    subset: str
    seed: int
    sample: Any
    raw: Any
    clean: Any
    segments: Any = None
    rendering: np.ndarray | None = None
    target: np.ndarray | None = None
    errors: Any = None
    events: Any = None
    other: "RunView | None" = None
    matched: Any = None
    flipped: Any = None
    sensitivity: Any = None

    @property
    def spacing(self) -> tuple[float, ...]:
        """The sample's voxel spacing."""
        return tuple(float(s) for s in self.sample.spacing)

    def graph(self, stage: str):
        """The run's graph of a stage."""
        return {"raw": self.raw, "clean": self.clean}[stage]


def run_view(
    store: Store,
    directory: Path | str,
    subset: str,
    seed: int,
    compare: Path | str | None = None,
    render: bool = True,
    sensitivity: str | None = None,
) -> RunView:
    """One sample of a run (and of a compared run), from the store.

    Parameters
    ----------
    store : Store
        Where the run's samples and caches are.
    directory : Path or str
        The run.
    subset, seed : str, int
        The sample.
    compare : Path or str, optional
        A second run, shown against the first.
    render : bool
        Render the fitted segments (with JAX).
    sensitivity : str, optional
        A parameter of the run's implicit derivatives
        (``sensitivity/implicit.parquet``), whose flip distances colour the
        events.
    """
    import pandas as pd

    from spleleton.bench.compare import (
        event_flips,
        match_errors,
        measure_floors,
        read_tables,
        replicate_runs,
    )
    from spleleton.bench.runner import load_fit, load_graphs, sample_label
    from spleleton.bench.suites import load_split
    from spleleton.bench.variants import Variant
    from spleleton.primitives._arrays import as_segment_arrays

    tables = read_tables(directory)
    label = sample_label(subset, seed)
    entry = tables.meta["samples"].get(label)
    if entry is None or entry.get("status") != "done":
        done = sorted(
            k for k, v in tables.meta["samples"].items() if v["status"] == "done"
        )
        raise KeyError(f"{label} is not done in {tables.name}; done: {done}")
    (loaded,) = load_split(
        store, tables.meta["suite"], subset, tables.meta["split"], [seed]
    )
    sample = loaded.sample
    raw, clean, _ = load_graphs(store.graph_path(entry["graph_key"]))
    variant = Variant.from_dict(tables.meta["spec"])
    segments = rendering = target = None
    if variant.pipeline != "skeletonize":
        record = load_fit(store.fit_path(entry["fit_key"]))
        segments = as_segment_arrays(record.segments)
        if render:
            from spleleton.render import render_jax

            rendering = np.asarray(
                render_jax(record.segments, sample.mask.shape, sample.spacing,
                           record.render_config)
            )  # fmt: skip
        if variant.target.name != "mask":
            from spleleton.targets import target_field

            target = np.asarray(
                target_field(variant.target, sample.mask, sample.spacing)
            )

    def mine(frame):
        if frame is None or not len(frame):
            return pd.DataFrame()
        return frame[(frame["subset"] == subset) & (frame["seed"] == seed)].reset_index(
            drop=True
        )

    view = RunView(
        tables.name, subset, seed, sample, raw, clean, segments, rendering, target,
        mine(tables.errors), mine(tables.events),
    )  # fmt: skip
    if compare is not None:
        view.other = run_view(store, compare, subset, seed, render=False)
        view.matched = match_errors(view.errors, view.other.errors)
        floors = measure_floors([read_tables(d) for d in replicate_runs(store, tables)])
        other_tables = read_tables(compare)
        if other_tables.name == tables.name:
            other_tables.name = f"{tables.name} (again)"
        one = dataclasses.replace(tables, events=view.events)
        two = dataclasses.replace(other_tables, events=view.other.events)
        table, flipped = event_flips(one, two, floors)
        if len(flipped):
            rates = {
                (r.event_type, r.stage): r.null_flip_rate
                for r in table.itertuples()
                if r.subset == subset
            }
            flipped = flipped.assign(
                null_flip_rate=[
                    rates.get((k, s), np.nan)
                    for k, s in zip(
                        flipped["event_type"], flipped["stage"], strict=True
                    )
                ]
            )
            positions = view.events.set_index(["event_id", "stage"])["position"]
            flipped["position"] = [
                positions[(e, s)]
                for e, s in zip(flipped["event_id"], flipped["stage"], strict=True)
            ]
        view.flipped = flipped
    if sensitivity is not None:
        view.sensitivity = sensitivity_rows(Path(directory), view, sensitivity)
    return view


def sensitivity_rows(directory: Path, view: RunView, parameter: str):
    """The sample's events with their flip distance for one parameter."""
    import pandas as pd

    path = directory / "sensitivity" / "implicit.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path}: run bench sensitivity implicit first")
    table = pd.read_parquet(path)
    table = table[
        (table["subset"] == view.subset)
        & (table["seed"] == view.seed)
        & (table["parameter"] == parameter)
    ]
    if table.empty:
        known = sorted(pd.read_parquet(path)["parameter"].unique())
        raise KeyError(f"no derivatives for {parameter!r} here; known: {known}")
    positions = view.events[view.events["stage"] == "raw"][
        ["event_type", "event_id", "position"]
    ]
    return table.merge(positions, on=["event_type", "event_id"]).reset_index(drop=True)


def add_sensitivity_layer(viewer, view: RunView, scale=None):
    """The events as points coloured by log10 |flip distance| (unsteerable grey)."""
    rows = view.sensitivity
    points = np.stack(rows["position"].map(np.asarray).to_list())
    distance = np.abs(rows["flip_distance"].astype(float).to_numpy())
    steerable = np.isfinite(distance)
    value = np.where(steerable, np.log10(np.clip(distance, 1e-3, 1e3)), np.nan)
    features = {
        "event_id": rows["event_id"].to_numpy(),
        "q": rows["q"].to_numpy(),
        "dq": rows["dq"].to_numpy(),
        "flip_distance": rows["flip_distance"].to_numpy(),
        "log10_flip": np.nan_to_num(value, nan=3.0),
    }
    return viewer.add_points(
        points, name=f"flip distance: {rows['parameter'].iloc[0]}", size=3,
        features=features, face_color="log10_flip", face_colormap="viridis_r",
        face_contrast_limits=(-3.0, 3.0), border_width=0, scale=scale,
    )  # fmt: skip


def browser_rows(view: RunView):
    """The error browser's rows: errors, events and (compared) flips.

    Returns
    -------
    pandas.DataFrame
        ``kind`` (``error``, ``event`` or ``flip``), ``id``, ``type``,
        ``stage``, ``cause`` (an error's cause, an event's outcome), ``status``
        (compare mode: ``fixed``, ``new`` or ``shared``), ``size``, ``note``
        and ``position``.
    """
    import pandas as pd

    from spleleton.bench.compare import FLIP_OUTCOMES

    rows = []
    status = {}
    if view.matched is not None and len(view.matched):
        for row in view.matched.itertuples():
            status[(row.run, row.error_id)] = row.status
    for e in view.errors.itertuples() if len(view.errors) else []:
        rows.append(
            {"kind": "error", "id": e.error_id, "type": e.type, "stage": e.stage,
             "cause": e.cause, "status": status.get(("baseline", e.error_id), ""),
             "size": float(e.size), "note": view.name, "position": list(e.position)}
        )  # fmt: skip
    if view.matched is not None and len(view.matched):
        for e in view.matched[view.matched["status"] == "new"].itertuples():
            rows.append(
                {"kind": "error", "id": e.error_id, "type": e.type, "stage": e.stage,
                 "cause": e.cause, "status": "new", "size": np.nan,
                 "note": view.other.name, "position": list(e.position)}
            )  # fmt: skip
    for e in view.events.itertuples() if len(view.events) else []:
        outcome = FLIP_OUTCOMES.get(e.event_type, (None,))[0]
        value = getattr(e, outcome, None) if outcome else None
        rows.append(
            {"kind": "event", "id": e.event_id, "type": e.event_type, "stage": e.stage,
             "cause": "" if outcome is None else f"{outcome}={value}", "status": "",
             "size": np.nan, "note": "", "position": list(e.position)}
        )  # fmt: skip
    if view.flipped is not None and len(view.flipped):
        for e in view.flipped.itertuples():
            rows.append(
                {"kind": "flip", "id": e.event_id, "type": e.event_type,
                 "stage": e.stage, "cause": f"{e.outcome}: {e.value_baseline} -> {e.value}",  # noqa: E501
                 "status": "fixed" if e.fixed else "new", "size": np.nan,
                 "note": f"null flip rate {e.null_flip_rate:.3f}",
                 "position": list(e.position)}
            )  # fmt: skip
    return pd.DataFrame(rows)


def add_run_layers(viewer, view: RunView, scaled: bool = True) -> dict[str, Any]:
    """Every layer of the run mode (see the module's docstring).

    ``scaled=False`` leaves the scale off the image layers: the mistake the
    spacing check must catch.

    Returns
    -------
    dict
        Name -> layer (or list of layers, for graphs).
    """
    from spleleton.viz import add_corruptions, add_graph

    scale = view.spacing if scaled else None
    sample = view.sample
    layers: dict[str, Any] = {
        "clean mask": viewer.add_labels(
            sample.clean_mask.astype(np.uint8), name="clean mask", scale=scale,
            visible=False,
        ),
        "mask": viewer.add_labels(
            sample.mask.astype(np.uint8), name="mask", scale=scale, opacity=0.3
        ),
    }  # fmt: skip
    if view.target is not None:
        layers["target"] = viewer.add_image(
            np.asarray(view.target, dtype=np.float32), name="target", scale=scale,
            colormap="gray", blending="additive", visible=False,
        )  # fmt: skip
    if view.rendering is not None:
        layers["rendering"] = viewer.add_image(
            np.asarray(view.rendering, dtype=np.float32), name="rendering",
            scale=scale, colormap="magma", contrast_limits=(0.0, 1.0),
            blending="additive", opacity=0.5, visible=False,
        )  # fmt: skip
    if view.segments is not None:
        arrays = view.segments
        layers["segments"] = viewer.add_shapes(
            list(np.stack([arrays.start, arrays.end], axis=1)),
            shape_type="line",
            features={
                "index": np.arange(len(arrays)),
                "opacity": arrays.opacity,
                "radius": arrays.radius.mean(axis=-1),
            },
            edge_color="opacity", edge_colormap="viridis",
            edge_contrast_limits=(0.0, 1.0), edge_width=0.4, name="segments",
        )  # fmt: skip
    graphs = [(view, "raw", "orange", False), (view, "clean", "cyan", True)]
    if view.other is not None:
        graphs += [
            (view.other, "raw", "khaki", False),
            (view.other, "clean", "#e87ba4", True),
        ]
    for owner, stage, colour, visible in graphs:
        name = f"{owner.name}: {stage} graph"
        added = add_graph(viewer, owner.graph(stage), name=name, color=colour)
        for layer in added:
            layer.visible = visible
        layers[name] = added
    layers["ground truth"] = add_graph(
        viewer, sample.graph, name="ground truth", color="red"
    )
    layers["corruptions"] = add_corruptions(viewer, sample.corruptions, ndim=3)
    errors = view.errors
    if len(errors):
        layers["errors"] = viewer.add_points(
            np.array(errors["position"].tolist(), dtype=float).reshape(-1, 3),
            features={
                "id": errors["error_id"].tolist(), "type": errors["type"].tolist(),
                "cause": errors["cause"].tolist(), "stage": errors["stage"].tolist(),
                "size": errors["size"].astype(float).tolist(),
            },
            face_color=[ERROR_COLOURS.get(t, "white") for t in errors["type"]],
            symbol=[STAGE_SYMBOLS[s] for s in errors["stage"]],
            size=2.5, name="errors",
        )  # fmt: skip
    if view.matched is not None and len(view.matched):
        matched = view.matched
        layers["error status"] = viewer.add_points(
            np.array(matched["position"].tolist(), dtype=float).reshape(-1, 3),
            features={
                "id": matched["error_id"].tolist(), "type": matched["type"].tolist(),
                "status": matched["status"].tolist(),
                "stage": matched["stage"].tolist(),
            },
            face_color="transparent",
            border_color=[STATUS_COLOURS[s] for s in matched["status"]],
            border_width=0.25, size=4.0, name="error status",
        )  # fmt: skip
    viewer.dims.axis_labels = ("z", "y", "x")
    return layers


class ErrorBrowser:
    """The error browser: a filterable table of rows, and the camera on them.

    Usable without a dock widget (``go_to``, ``next``, ``previous``,
    ``set_filter``), so that it can be driven and checked headless.
    """

    COLUMNS = ("kind", "id", "type", "stage", "cause", "status", "size", "note")
    FILTERS = ("kind", "type", "cause", "stage", "status")

    def __init__(self, viewer, rows) -> None:
        self.viewer = viewer
        self.rows = rows
        self.filters: dict[str, str] = {}
        self.index = -1
        self.base_zoom: float | None = None
        self.shown = rows
        self.table = None
        self._text = viewer.canvas.overlays.text
        self._text.visible = True
        self._text.position = "top_left"
        self._text.font_size = 10
        n = int((rows["kind"] == "error").sum()) if len(rows) else 0
        self._text.text = f"{n} errors: n / p step through the browser's rows"
        self.set_filter(kind="error")

    def set_filter(self, **filters: str) -> None:
        """Keep the rows whose columns equal the given values ("" for any)."""
        self.filters |= filters
        shown = self.rows
        for column, value in self.filters.items():
            if value and len(shown):
                key = shown[column].astype(str)
                if column == "cause":  # a cause kind matches its IDs too
                    key = key.str.split("#").str[0]
                    value = str(value).split("#")[0]
                shown = shown[key == str(value)]
        self.shown = shown.reset_index(drop=True)
        self.index = -1
        self._fill()

    def go_to(self, k: int) -> dict[str, Any] | None:
        """Centre the camera on shown row ``k`` (modulo their number)."""
        if not len(self.shown):
            return None
        self.index = k % len(self.shown)
        row = self.shown.iloc[self.index].to_dict()
        camera = self.viewer.scene.camera
        if self.base_zoom is None:
            self.base_zoom = camera.zoom
        camera.center = tuple(float(x) for x in row["position"])
        camera.zoom = ZOOM * self.base_zoom
        self._text.text = (
            f"{row['kind']} {self.index + 1}/{len(self.shown)}: {row['stage']} "
            f"{row['type']} {row['id']}, {row['cause']}"
            + (f", {row['status']}" if row["status"] else "")
            + (f" ({row['note']})" if row["note"] else "")
        )
        if self.table is not None:
            self.table.blockSignals(True)
            self.table.selectRow(self.index)
            self.table.blockSignals(False)
        return row

    def next(self) -> dict[str, Any] | None:
        """The next shown row."""
        return self.go_to(self.index + 1)

    def previous(self) -> dict[str, Any] | None:
        """The previous shown row."""
        return self.go_to(self.index - 1)

    def widget(self):
        """The dock widget: filters above the table (Qt)."""
        from qtpy.QtWidgets import (
            QAbstractItemView,
            QComboBox,
            QFormLayout,
            QTableWidget,
            QVBoxLayout,
            QWidget,
        )

        root = QWidget()
        layout = QVBoxLayout(root)
        form = QFormLayout()
        for column in self.FILTERS:
            box = QComboBox()
            values = [] if not len(self.rows) else self.rows[column].astype(str)
            if column == "cause" and len(self.rows):
                values = values.str.split("#").str[0]
            box.addItems(["", *sorted(set(values) - {""})])
            if column in self.filters:
                box.setCurrentText(self.filters[column])
            box.currentTextChanged.connect(
                lambda text, c=column: self.set_filter(**{c: text})
            )
            form.addRow(column, box)
        layout.addLayout(form)
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._selected)
        layout.addWidget(self.table)
        self._fill()
        return root

    def _selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if rows:
            self.go_to(rows[0].row())

    def _fill(self) -> None:
        if self.table is None:
            return
        from qtpy.QtWidgets import QTableWidgetItem

        self.table.blockSignals(True)
        self.table.clear()
        self.table.setColumnCount(len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(list(self.COLUMNS))
        self.table.setRowCount(len(self.shown))
        for i, row in enumerate(self.shown.itertuples(index=False)):
            for j, column in enumerate(self.COLUMNS):
                value = getattr(row, column)
                text = (
                    ""
                    if value is None or (isinstance(value, float) and np.isnan(value))
                    else (f"{value:.1f}" if isinstance(value, float) else str(value))
                )
                self.table.setItem(i, j, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()
        self.table.blockSignals(False)


def view_run(view: RunView, show: bool = True, dock: bool = True):
    """Open napari on one sample of a run (see the module's docstring).

    Returns
    -------
    viewer : napari.Viewer
    layers : dict
        From :func:`add_run_layers`.
    browser : ErrorBrowser
    """
    import napari

    title = f"{view.name}: {view.subset} seed {view.seed}"
    if view.other is not None:
        title += f" against {view.other.name}"
    viewer = napari.Viewer(ndisplay=3, show=show, title=title)
    layers = add_run_layers(viewer, view)
    if view.sensitivity is not None and len(view.sensitivity):
        layers["sensitivity"] = add_sensitivity_layer(viewer, view)
    browser = ErrorBrowser(viewer, browser_rows(view))
    viewer.bind_key("n", lambda _: browser.next(), overwrite=True)
    viewer.bind_key("p", lambda _: browser.previous(), overwrite=True)
    if dock:
        viewer.window.add_dock_widget(browser.widget(), area="right", name="errors")
    return viewer, layers, browser


# -- automatic checks ------------------------------------------------------------------


def _polyline_distance(points: np.ndarray, paths: list[np.ndarray]) -> np.ndarray:
    """Distance from each point to the nearest of a set of polylines."""
    points = np.atleast_2d(np.asarray(points, dtype=float))
    best = np.full(len(points), np.inf)
    for path in paths:
        a, b = path[:-1], path[1:]
        ab = b - a
        length2 = np.maximum((ab**2).sum(axis=1), 1e-30)
        t = np.clip(
            ((points[:, None, :] - a[None]) * ab[None]).sum(axis=-1) / length2, 0, 1
        )
        closest = a[None] + t[..., None] * ab[None]
        best = np.minimum(
            best, np.linalg.norm(points[:, None, :] - closest, axis=-1).min(axis=1)
        )
    return best


def _node_distance(points, graph, nodes) -> np.ndarray:
    positions = graph.node_positions[nodes]
    points = np.atleast_2d(np.asarray(points, dtype=float))
    if not len(positions):
        return np.full(len(points), np.inf)
    return np.linalg.norm(points[:, None] - positions[None], axis=-1).min(axis=1)


def _centreline(graph, step: float = 0.5) -> np.ndarray:
    from spleleton.graph import resample_polyline

    return np.concatenate(
        [resample_polyline(path, step)[0] for path in graph.edge_paths]
    )


def sample_layer(layer, points: np.ndarray) -> np.ndarray:
    """A layer's data at world points (nearest voxel; outside counts as 0)."""
    data = np.asarray(layer.data)
    index = np.round(
        np.array([layer.world_to_data(p) for p in points], dtype=float)
    ).astype(int)
    inside = np.all((index >= 0) & (index < np.array(data.shape)), axis=1)
    values = np.zeros(len(points), dtype=float)
    values[inside] = data[tuple(index[inside].T)]
    return values


def error_on_target(view: RunView, error: dict[str, Any]) -> float:
    """How far an error's point lies from what it names (0 when it sits on it).

    An end lies on an end node of its stage's graph, a missed run on the true
    centreline, a false-positive or off-centre run on its stage's graph, a
    missed junction on a true junction, an extra one on a junction node of
    its stage's graph.
    """
    graph = view.graph(error["stage"])
    truth = view.sample.graph
    position = np.asarray(error["position"], dtype=float)
    kind = error["type"]
    if kind in ("spur_end", "break_end"):
        distance = _node_distance(position, graph, graph.end_nodes)
    elif kind == "missed":
        distance = _polyline_distance(position, truth.edge_paths)
    elif kind in ("false_positive", "off_centre"):
        distance = _polyline_distance(position, graph.edge_paths)
    elif kind == "junction_missed":
        distance = _node_distance(position, truth, truth.junction_nodes)
    else:
        distance = _node_distance(position, graph, graph.junction_nodes)
    return float(distance[0])


def check_view(view: RunView, layers: dict[str, Any]) -> tuple[dict, dict]:
    """The automatic checks of one sample's layers (step 0.9).

    Returns
    -------
    checks : dict
        Name -> bool: every layer lies inside the mask's extent; the true
        centreline lies in the clean mask (through the clean mask layer's
        ``world_to_data``); opaque fitted segments at least a voxel thick lie
        where their rendering is at least 0.5; the error layer holds exactly
        the sample's errors; every error sits on what it names.
    info : dict
        The numbers behind them.
    """
    flat = []
    for value in layers.values():
        flat += value if isinstance(value, list) else [value]
    box = np.asarray(layers["mask"].extent.world)
    inside = [
        bool(
            np.all(np.asarray(layer.extent.world)[0] >= box[0] - 1.0)
            and np.all(np.asarray(layer.extent.world)[1] <= box[1] + 1.0)
        )
        for layer in flat
    ]
    in_clean = sample_layer(layers["clean mask"], _centreline(view.sample.graph)) > 0
    checks = {
        "every layer lies inside the mask's extent": all(inside),
        f"the true centreline lies in the clean mask (>= {ALIGNED:.0%})": bool(
            in_clean.mean() >= ALIGNED
        ),
    }
    info: dict[str, Any] = {
        "layers": [layer.name for layer in flat],
        "outside": [
            layer.name for layer, ok in zip(flat, inside, strict=True) if not ok
        ],
        "true_centreline_in_clean_mask": float(in_clean.mean()),
    }
    if view.segments is not None and "rendering" in layers:
        arrays = view.segments
        thick = (arrays.opacity >= OPAQUE) & (arrays.radius.mean(axis=1) >= THICK)
        middle = (arrays.start[thick] + arrays.end[thick]) / 2
        rendered = sample_layer(layers["rendering"], middle) >= 0.5
        checks["opaque fitted segments lie where their rendering is >= 0.5"] = bool(
            rendered.all()
        )
        info["opaque_segments_rendered"] = f"{int(rendered.sum())}/{len(rendered)}"
    errors = view.errors.to_dict("records") if len(view.errors) else []
    if errors:
        layer = layers["errors"]
        checks["the error layer holds the sample's errors"] = np.array_equal(
            layer.data, np.array([e["position"] for e in errors], dtype=float)
        ) and list(layer.features["id"]) == [e["error_id"] for e in errors]
        distances = {}
        for error in errors:
            distances.setdefault(error["type"], []).append(error_on_target(view, error))
        checks["every error sits on what it names"] = all(
            d <= ON for values in distances.values() for d in values
        )
        info["errors_on_target"] = {
            k: f"{sum(d <= ON for d in v)}/{len(v)}" for k, v in distances.items()
        }
    return checks, info


def anisotropic_check(
    case: str = "y-thin", spacing: tuple[float, ...] = (2.0, 1.0, 1.0)
) -> tuple[dict, dict]:
    """The image layers at an anisotropic spacing, with and without their scale.

    The case's true graph is rasterised at ``spacing``, and its true segments
    rendered with JAX; with the scale, the true centreline lies in the mask
    and in the rendering, and the two agree; without it (the mistake), the
    centreline misses the mask.
    """
    import jax.numpy as jnp
    import napari

    from spleleton.bench.cases import build_case
    from spleleton.primitives import SegmentArrays
    from spleleton.primitives._primitives_jax import JaxLineSegments
    from spleleton.render import render_jax
    from spleleton.synthetic.rasterize import rasterize_graph

    graph = build_case(case).sample.graph
    shape = tuple(
        int(np.ceil(64 / s)) for s in spacing
    )  # proto-cases are 64^3 at spacing 1
    mask = rasterize_graph(graph, shape, spacing)
    segments = JaxLineSegments.from_arrays(
        SegmentArrays.from_graph(graph), dtype=jnp.float32
    )
    rendering = np.asarray(render_jax(segments, shape, spacing))
    points = _centreline(graph)
    found = {}
    for scaled in (True, False):
        viewer = napari.Viewer(ndisplay=3, show=False)
        try:
            scale = spacing if scaled else None
            mask_layer = viewer.add_labels(mask.astype(np.uint8), scale=scale)
            render_layer = viewer.add_image(rendering, scale=scale)
            found[scaled] = {
                "mask": float((sample_layer(mask_layer, points) > 0).mean()),
                "rendering": float((sample_layer(render_layer, points) >= 0.5).mean()),
            }
        finally:
            viewer.close()
    solid = rendering >= 0.5
    iou = float((solid & mask).sum() / max((solid | mask).sum(), 1))
    checks = {
        f"spacing {spacing}: the true graph lies in the mask and the rendering": (
            found[True]["mask"] >= ALIGNED and found[True]["rendering"] >= ALIGNED
        ),
        f"spacing {spacing}: the JAX rendering agrees with the mask (IoU >= 0.9)": (
            iou >= 0.9
        ),
        f"spacing {spacing}: without the layer scale, the check fails": (
            found[False]["mask"] <= MISALIGNED
        ),
    }
    info = {"case": case, "spacing": spacing, "shape": shape, "scaled": found[True],
            "unscaled": found[False], "rendering_mask_iou": iou}  # fmt: skip
    return checks, info


def screenshots(viewer, browser: ErrorBrowser, directory: Path) -> dict[str, float]:
    """An overview, and the first clean-graph error up close.

    The viewer must be shown: on macOS a hidden canvas draws nothing. The 3D
    mask is an opaque surface over the graphs, so it is hidden meanwhile.

    Returns
    -------
    dict
        Each file, with the share of its pixels that are not black.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    shots = {}
    mask = viewer.layers["mask"]
    viewer.window.resize(1400, 1000)
    mask.visible = False
    viewer.reset_view()
    viewer.scene.camera.angles = (0.0, 30.0, 60.0)
    shots["overview.png"] = viewer.screenshot(str(directory / "overview.png"))
    browser.set_filter(kind="error", stage="clean")
    if browser.go_to(0) is not None:
        shots["first_error.png"] = viewer.screenshot(str(directory / "first_error.png"))
    browser.set_filter(stage="")
    mask.visible = True
    viewer.reset_view()
    return {
        str(directory / name): float((image[..., :3].sum(axis=-1) > 0).mean())
        for name, image in shots.items()
    }


# -- recording -------------------------------------------------------------------------


def record_sample(
    store: Store, directory: Path | str, subset: str, seed: int
) -> tuple[Path, dict[str, bool]]:
    """Re-run one sample's fit with a recorder; check it against the stored fit.

    The fit is deterministic, so the recording's fit must equal the stored
    one: segments (raw float32 parameters) and loss history bit for bit.
    The recording goes to ``recordings/<subset>_<seed>.npz`` in the run.

    Returns
    -------
    path : Path
        The recording.
    checks : dict
        Name -> bool.
    """
    from spleleton.bench.runner import (
        SEGMENT_FIELDS,
        Limits,
        load_fit,
        load_graphs,
        read_run,
        run_fit,
        sample_label,
    )
    from spleleton.bench.suites import load_split
    from spleleton.bench.variants import Variant
    from spleleton.recording import FitRecorder

    directory = Path(directory)
    meta = read_run(directory)
    variant = Variant.from_dict(meta["spec"])
    if variant.pipeline == "skeletonize":
        raise ValueError(f"{meta['name']} fits nothing: there is nothing to record")
    entry = meta["samples"][sample_label(subset, seed)]
    (loaded,) = load_split(store, meta["suite"], subset, meta["split"], [seed])
    sample = loaded.sample
    recorder = FitRecorder(
        sample.mask.shape, every=RECORD_EVERY, spacing=sample.spacing
    )
    output, _ = run_fit(
        variant, loaded, meta["replicate"], Limits(1e9, 10**9), extra_callback=recorder
    )
    stored = load_fit(store.fit_path(entry["fit_key"]))
    _, clean, _ = load_graphs(store.graph_path(entry["graph_key"]))
    recording = recorder.finish(
        output.history,
        target=sample.mask,
        layout=output.layout,
        ground_truth=sample.graph,
        final_graph=clean,
    )
    checks = {
        "segments bit-identical to the stored fit": all(
            np.array_equal(
                np.asarray(getattr(output.segments, f)),
                np.asarray(getattr(stored.segments, f)),
            )
            for f in SEGMENT_FIELDS
        ),
        "loss history bit-identical to the stored fit": output.history
        == stored.history,
    }
    (directory / "recordings").mkdir(parents=True, exist_ok=True)
    path = recording.save(directory / "recordings" / sample_label(subset, seed))
    return path, checks
