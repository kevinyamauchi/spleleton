"""Variants: what is compared (plan, Section 5.1).

A :class:`Variant` is a frozen dataclass that fully specifies a pipeline:
its input (the target field), its initialisation, its fit settings and its
graph stage (assembly and clean-up). It is identified by the hash of its
**resolved** spec (:meth:`Variant.resolved`, :meth:`Variant.spec_hash`), not
by its name: the name is only a label, and a name pointed at a different spec
gets a new hash, so nothing is overwritten.

The fit and the graph settings are disjoint, so the fit's cache key
(:meth:`Variant.fit_spec`) leaves the graph settings out, and a sweep over
graph settings reuses every fit.

**Registry.** Variants are registered by name (:func:`register_variant`).
The baselines reproduce ``scripts/evaluate_pipelines.py``:

- ``skeletonize``: ``skeletonize`` with no clean-up;
- ``skeletonize+clean``: ``skeletonize``, cleaned up;
- ``phase4``: ``skeleton_fit`` (Phase 4), assembled by its layout;
- ``phase5``: ``scratch_fit`` (Phase 5), assembled by its endpoints. Every
  experiment starts from it.

**Overrides** (:func:`override`, :func:`parse_override`) set one value along
a dotted path from the variant, as ``dataclasses.replace`` would:
``fit.connection_reach=2.0``. A loss term is addressed by its name
(``fit.fit.losses.dice.weight``), and a ``params`` entry by its key
(``graph.params.contained``, ``target.params.sigma``; a missing key is
added). **Sweeps** (:func:`sweep`) make one variant per value of one path,
each tagged so that plots can draw them as a curve.
"""

import ast
import dataclasses
import typing
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from spleleton._serialise import from_dict, spec_hash, to_dict
from spleleton.fit._losses import LossTerm
from spleleton.pipeline import (
    CleanConfig,
    GraphConfig,
    ScratchFitConfig,
    SkeletonFitConfig,
)
from spleleton.targets import TargetConfig, target_render_mode

Pipeline = Literal["skeletonize", "skeleton_fit", "scratch_fit"]
Init = Literal["default", "ground_truth"]

#: The fit settings' class of each pipeline that fits.
FIT_CONFIGS: dict[str, type] = {
    "skeleton_fit": SkeletonFitConfig,
    "scratch_fit": ScratchFitConfig,
}
#: The assembly method a pipeline's graph stage uses by default.
DEFAULT_METHODS = {
    "skeletonize": "skeletonize",
    "skeleton_fit": "layout",
    "scratch_fit": "assemble",
}


@dataclass(frozen=True)
class Variant:
    """A fully specified pipeline (plan, Section 5.1).

    Parameters
    ----------
    pipeline : {"skeletonize", "skeleton_fit", "scratch_fit"}
        ``skeletonize`` fits nothing; ``skeleton_fit`` is Phase 4 and
        ``scratch_fit`` Phase 5.
    target : TargetConfig
        The field the rendering is fitted to (see :mod:`spleleton.targets`).
    fit : SkeletonFitConfig or ScratchFitConfig, optional
        The fit settings, of the pipeline's class. ``None`` is the
        pipeline's default (and the only value for ``skeletonize``).
    graph : GraphConfig, optional
        The graph stage. ``None`` is ``GraphConfig`` with the pipeline's
        default method (:data:`DEFAULT_METHODS`).
    init : {"default", "ground_truth"}
        Where the fit starts: the pipeline's own initialisation, or the true
        graph cut into pieces (plan, Section 10.6).
    init_jitter : float
        Standard deviation of the noise added to the initial segments'
        endpoints, in physical units, for the noise floor's replicates (plan,
        Section 8.3). 0 adds none.

    Notes
    -----
    JAX only: there is no backend field. Two variants that resolve to the same
    spec are the same variant, whatever their names.
    """

    pipeline: Pipeline
    target: TargetConfig = field(default_factory=TargetConfig)
    fit: SkeletonFitConfig | ScratchFitConfig | None = None
    graph: GraphConfig | None = None
    init: Init = "default"
    init_jitter: float = 0.0

    def resolved(self) -> "Variant":
        """The variant with every default filled in, after checking it.

        Defaults are filled for ``fit`` and ``graph``, and every ``params``
        tuple (of the target, the graph and each loss term) is sorted by key,
        so that the order in which parameters were given changes no hash.

        Raises
        ------
        ValueError
            If the parts do not fit together: a fit config of the wrong
            class, a fit for ``skeletonize``, a graph method that the pipeline
            cannot use, a target for another render mode than the fit's, or a
            negative jitter.
        """
        if self.pipeline not in DEFAULT_METHODS:
            raise ValueError(
                f"unknown pipeline {self.pipeline!r}; "
                f"known: {', '.join(DEFAULT_METHODS)}"
            )
        if self.init not in typing.get_args(Init):
            raise ValueError(f"unknown init {self.init!r}")
        if self.init_jitter < 0:
            raise ValueError(f"init_jitter must be >= 0, got {self.init_jitter}")
        fit = self.fit
        if self.pipeline == "skeletonize":
            if fit is not None:
                raise ValueError("the skeletonize pipeline fits nothing: fit=None")
            if self.target != TargetConfig() or self.init != "default":
                raise ValueError("the skeletonize pipeline takes no target or init")
            if self.init_jitter:
                raise ValueError("the skeletonize pipeline takes no init_jitter")
        elif self.init == "ground_truth" and self.pipeline != "scratch_fit":
            raise ValueError("only scratch_fit can start from the ground truth")
        if self.pipeline != "skeletonize":
            cls = FIT_CONFIGS[self.pipeline]
            fit = cls() if fit is None else fit
            if not isinstance(fit, cls):
                raise ValueError(
                    f"{self.pipeline} needs a {cls.__name__}, "
                    f"not a {type(fit).__name__}"
                )
            mode = target_render_mode(self.target)
            if fit.fit.render.mode != mode:
                raise ValueError(
                    f"target {self.target.name!r} is for the {mode!r} render "
                    f"mode, but the fit renders in {fit.fit.render.mode!r} "
                    "(set fit.fit.render.mode)"
                )
        graph = self.graph or GraphConfig(method=DEFAULT_METHODS[self.pipeline])
        if (graph.method == "skeletonize") != (self.pipeline == "skeletonize"):
            raise ValueError(
                f"the {self.pipeline} pipeline cannot use the "
                f"{graph.method!r} graph method"
            )
        if graph.method == "layout" and self.pipeline != "skeleton_fit":
            raise ValueError("only skeleton_fit has a layout to assemble by")
        return _sort_params(
            dataclasses.replace(self, fit=fit, graph=graph), _hints(Variant)
        )

    def spec_hash(self) -> str:
        """SHA-256 of the resolved spec's canonical JSON: the variant's identity."""
        return spec_hash(self.resolved())

    def fit_spec(self, replicate: int = 0) -> dict[str, Any] | None:
        """What the fit depends on, for its cache key; ``None`` if nothing is fitted.

        The pipeline, the target, the fit settings, the initialisation, the
        jitter and the replicate index (plan, Section 6.1). The graph settings
        are left out, so that variants differing only in them share fits.
        """
        resolved = self.resolved()
        if resolved.pipeline == "skeletonize":
            return None
        return {
            "pipeline": resolved.pipeline,
            "target": resolved.target,
            "fit": resolved.fit,
            "init": resolved.init,
            "init_jitter": resolved.init_jitter,
            "replicate": replicate,
        }

    def to_dict(self) -> dict[str, Any]:
        """Plain Python types (for JSON)."""
        return to_dict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Variant":
        """Rebuild a variant from :meth:`to_dict`'s output.

        The fit settings' class is taken from the pipeline.
        """
        data = dict(data)
        fit = data.pop("fit", None)
        variant = from_dict(cls, data)
        if fit is not None:
            pipeline = FIT_CONFIGS.get(variant.pipeline)
            if pipeline is None:
                raise ValueError(f"the {variant.pipeline} pipeline fits nothing")
            variant = dataclasses.replace(variant, fit=from_dict(pipeline, fit))
        return variant


@dataclass(frozen=True)
class SweepTag:
    """Where a variant sits in a sweep: the base variant, the path and the value."""

    base: str
    path: str
    value: Any


@dataclass(frozen=True)
class NamedVariant:
    """A variant with its label (and its place in a sweep, if any).

    The name labels results; the identity is ``variant.spec_hash()``.
    """

    name: str
    variant: Variant
    sweep: SweepTag | None = None

    @property
    def spec_hash(self) -> str:
        """The variant's identity."""
        return self.variant.spec_hash()


# -- the registry ----------------------------------------------------------------------

_VARIANTS: dict[str, Variant] = {}


def register_variant(name: str, variant: Variant) -> Variant:
    """Register a variant under a name (resolved, so it is checked now)."""
    if name in _VARIANTS:
        raise ValueError(f"variant {name!r} is already registered")
    _VARIANTS[name] = variant.resolved()
    return _VARIANTS[name]


def variant_names() -> tuple[str, ...]:
    """The registered variants' names."""
    return tuple(_VARIANTS)


def get_variant(name: str, overrides: Iterable[str] = ()) -> NamedVariant:
    """A registered variant, with optional ``path=value`` overrides.

    Parameters
    ----------
    name : str
        The registered name.
    overrides : iterable of str
        ``path=value`` strings (see :func:`parse_override`), applied in
        order. With overrides, the name gets them appended in brackets.

    Returns
    -------
    NamedVariant
        The resolved variant and its label.
    """
    try:
        variant = _VARIANTS[name]
    except KeyError:
        raise KeyError(
            f"unknown variant {name!r}; registered: {', '.join(_VARIANTS)}"
        ) from None
    overrides = list(overrides)
    variant = apply_overrides(variant, [parse_override(text) for text in overrides])
    label = f"{name}[{','.join(overrides)}]" if overrides else name
    return NamedVariant(label, variant)


# -- overrides and sweeps --------------------------------------------------------------


def override(variant: Variant, path: str, value: Any) -> Variant:
    """Set one value along a dotted path, as ``dataclasses.replace`` would.

    Parameters
    ----------
    variant : Variant
        The variant; it is resolved first, so that default fit and graph
        settings can be addressed.
    path : str
        Dotted field names from the variant (``fit.connection_reach``). In a
        tuple of loss terms, a part names a term (``fit.fit.losses.dice``);
        in a ``params`` tuple, a part names a key, which is added if missing
        (``graph.params.contained``).
    value : object
        The new value. An int given for a float field becomes a float.

    Returns
    -------
    Variant
        The new variant, resolved (and so checked).

    Raises
    ------
    KeyError
        If the path does not exist.
    ValueError
        If the new variant does not resolve.
    """
    return apply_overrides(variant, [(path, value)])


def apply_overrides(variant: Variant, overrides: Iterable[tuple[str, Any]]) -> Variant:
    """Set several values (see :func:`override`), then resolve once.

    The variant is checked only after the last change, so that changes that
    must go together (a centreline target and ``fit.fit.render.mode``) can
    be made one at a time.
    """
    variant = variant.resolved()
    for path, value in overrides:
        parts = path.split(".")
        if not path or not all(parts):
            raise KeyError(f"empty part in path {path!r}")
        variant = _set(variant, Variant, parts, value, path)
    return variant.resolved()


def parse_override(text: str) -> tuple[str, Any]:
    """Split ``path=value`` and parse the value.

    The value is read as a Python literal (``2.0``, ``(1, 2)``, ``'max'``),
    with ``true``, ``false`` and ``none`` (any case) as ``True``, ``False``
    and ``None``. Anything else is kept as a string.
    """
    path, separator, raw = text.partition("=")
    if not separator or not path.strip():
        raise ValueError(f"an override is path=value, got {text!r}")
    raw = raw.strip()
    words = {"true": True, "false": False, "none": None, "null": None}
    if raw.lower() in words:
        return path.strip(), words[raw.lower()]
    try:
        return path.strip(), ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return path.strip(), raw


def sweep(
    base: str | NamedVariant, path: str, values: Sequence[Any]
) -> list[NamedVariant]:
    """One variant per value of one path, tagged as a sweep.

    Parameters
    ----------
    base : str or NamedVariant
        The registered name of the variant to sweep from, or a named variant.
    path : str
        The dotted path (see :func:`override`).
    values : sequence
        The values, in the order the curve is drawn.

    Returns
    -------
    list of NamedVariant
        Named ``<base>[<path>=<value>]``, each with a :class:`SweepTag`.
    """
    named = get_variant(base) if isinstance(base, str) else base
    return [
        NamedVariant(
            f"{named.name}[{path}={value!r}]",
            override(named.variant, path, value),
            SweepTag(named.name, path, value),
        )
        for value in values
    ]


def _hints(cls: type) -> dict[str, Any]:
    return typing.get_type_hints(cls)


def _set(obj: Any, hint: Any, parts: list[str], value: Any, path: str) -> Any:
    """``obj`` with ``value`` set at ``parts`` (``hint`` is ``obj``'s type)."""
    if not parts:
        return _coerce(obj, value, path)
    head, rest = parts[0], parts[1:]
    if dataclasses.is_dataclass(obj):
        hints = _hints(type(obj))
        if head not in {f.name for f in dataclasses.fields(obj)}:
            raise KeyError(f"{type(obj).__name__} has no field {head!r} ({path})")
        new = _set(getattr(obj, head), hints[head], rest, value, path)
        return dataclasses.replace(obj, **{head: new})
    kind = _tuple_kind(hint)
    if kind == "terms":
        names = [term.name for term in obj]
        if head not in names:
            raise KeyError(f"no loss term {head!r} in {names} ({path})")
        index = names.index(head)
        items = list(obj)
        items[index] = _set(obj[index], LossTerm, rest, value, path)
        return tuple(items)
    if kind == "pairs":
        if rest:
            raise KeyError(f"{head!r} is a parameter; nothing below it ({path})")
        items = list(obj)
        keys = [key for key, _ in items]
        if head in keys:
            index = keys.index(head)
            items[index] = (head, _coerce(items[index][1], value, path))
        else:
            items.append((head, value))
        return tuple(items)
    raise KeyError(f"cannot address {head!r} in a {type(obj).__name__} ({path})")


def _tuple_kind(hint: Any) -> str | None:
    """``"terms"`` for a tuple of loss terms, ``"pairs"`` for a params tuple."""
    if typing.get_origin(hint) is not tuple:
        return None
    args = typing.get_args(hint)
    if len(args) == 2 and args[1] is Ellipsis:
        if args[0] is LossTerm:
            return "terms"
        if typing.get_origin(args[0]) is tuple and typing.get_args(args[0])[:1] == (
            str,
        ):
            return "pairs"
    return None


def _coerce(old: Any, value: Any, path: str) -> Any:
    """``value``, as a float where it replaces a float (ints are accepted)."""
    if (
        isinstance(old, float)
        and isinstance(value, int)
        and not isinstance(value, bool)
    ):
        return float(value)
    if isinstance(old, tuple) and isinstance(value, list):
        return tuple(value)
    if (
        old is not None
        and value is not None
        and isinstance(old, bool) != isinstance(value, bool)
    ):
        raise ValueError(f"{path}: expected a {type(old).__name__}, got {value!r}")
    return value


def _sort_params(obj: Any, hints: dict[str, Any]) -> Any:
    """Every ``params`` tuple of pairs below ``obj`` sorted by key."""
    if not dataclasses.is_dataclass(obj):
        return obj
    changes = {}
    for item in dataclasses.fields(obj):
        value = getattr(obj, item.name)
        hint = hints.get(item.name)
        kind = _tuple_kind(hint)
        if kind == "pairs":
            new = tuple(sorted(value, key=lambda pair: pair[0]))
        elif kind == "terms":
            new = tuple(_sort_params(term, _hints(LossTerm)) for term in value)
        elif dataclasses.is_dataclass(value):
            new = _sort_params(value, _hints(type(value)))
        else:
            continue
        if new != value:
            changes[item.name] = new
    return dataclasses.replace(obj, **changes) if changes else obj


# -- the baselines ---------------------------------------------------------------------

register_variant(
    "skeletonize",
    Variant(
        "skeletonize",
        graph=GraphConfig(method="skeletonize", clean=CleanConfig(enabled=False)),
    ),
)
register_variant(
    "skeletonize+clean",
    Variant("skeletonize", graph=GraphConfig(method="skeletonize")),
)
register_variant(
    "phase4",
    Variant(
        "skeleton_fit",
        fit=SkeletonFitConfig(free_end_weight=3.0),
        graph=GraphConfig(method="layout"),
    ),
)
register_variant(
    "phase5",
    Variant("scratch_fit", fit=ScratchFitConfig(), graph=GraphConfig("assemble")),
)

#: The baselines (plan, Section 5.1), in the order they are reported.
BASELINES = ("phase5", "phase4", "skeletonize", "skeletonize+clean")

#: Phase 5 started from the subdivided true graph (plan, Section 10.6).
register_variant(
    "phase5-ground-truth",
    Variant(
        "scratch_fit",
        fit=ScratchFitConfig(),
        graph=GraphConfig("assemble"),
        init="ground_truth",
    ),
)


# -- the screening candidates ----------------------------------------------------------


def _with_losses(*terms: LossTerm) -> Variant:
    base = ScratchFitConfig()
    fit = dataclasses.replace(base.fit, losses=terms)
    return Variant("scratch_fit", fit=dataclasses.replace(base, fit=fit))


def _with_target(name: str, mode: str = "tube", **params: Any) -> Variant:
    base = ScratchFitConfig()
    render = dataclasses.replace(base.fit.render, mode=mode)
    fit = dataclasses.replace(base, fit=dataclasses.replace(base.fit, render=render))
    return Variant("scratch_fit", TargetConfig(name, tuple(params.items())), fit=fit)


#: The screening candidates of the plan's Section 5.2, as Phase 5 variants.
CANDIDATES = {
    "phase5-weighted_l2": _with_losses(
        LossTerm("weighted_l2", 1.0, (("pos_weight", 10.0),)),
        LossTerm("dice", 1.0),
        LossTerm("sparsity", 0.5),
    ),
    "phase5-focal": _with_losses(
        LossTerm("focal", 1.0, (("gamma", 2.0),)),
        LossTerm("dice", 1.0),
        LossTerm("sparsity", 0.5),
    ),
    "phase5-tversky": _with_losses(
        LossTerm("bce", 1.0),
        LossTerm("tversky", 1.0, (("alpha", 0.3), ("beta", 0.7))),
        LossTerm("sparsity", 0.5),
    ),
    "phase5-cldice": _with_losses(
        LossTerm("bce", 1.0),
        LossTerm("dice", 0.5),
        LossTerm("sparsity", 0.5),
        LossTerm("cldice", 0.5, (("k", 10),)),
    ),
    "phase5-smoothed_mask": _with_target("smoothed_mask", sigma=1.0),
    "phase5-opened_mask": _with_target("opened_mask", radius=2.0),
    "phase5-edt_ridge": _with_target("edt_ridge", "centreline", sigma=1.0),
    "phase5-soft_skeleton": _with_target("soft_skeleton", "centreline", k=10),
}
for _name, _variant in CANDIDATES.items():
    register_variant(_name, _variant)

#: The candidates stage S2 screens: all but ``cldice``, whose probe is too slow
#: (the plan's open question 1).
SCREENED = tuple(name for name in CANDIDATES if name != "phase5-cldice")
