"""The probe spec: the part of a variant that defines its loss (plan, Section 10.7)."""

import dataclasses
from dataclasses import dataclass

from spleleton.bench.variants import NamedVariant, Variant
from spleleton.fit._losses import LossTerm
from spleleton.render._config import RenderConfig
from spleleton.targets import TargetConfig

#: The error families (plan, Section 10.2), in the order they are reported.
FAMILIES = (
    "junction_detach",
    "break_retract",
    "bump_spur",
    "bump_shift",
    "edge_shift",
    "duplicate",
    "radius_scale",
)
#: The families of the three target error modes (the screen's rule).
TARGET_FAMILIES = ("junction_detach", "break_retract", "bump_spur", "bump_shift")
#: Families that add a primitive; their margins are taken against the
#: reference without it.
NEW_SLOT_FAMILIES = ("bump_spur", "duplicate")
#: Families whose knob does not act on the centreline render (Section 5.2).
CENTRELINE_SKIPS = ("radius_scale",)
#: The ε grid of rigid margins and forces.
EPS_GRID = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
#: The slim profile relaxes these ε and the largest one at which no moved
#: segment collapses; 0.5 catches the small attraction of a thin child (P6).
SLIM_EPS = (0.0, 0.5, 1.0)
#: The slim probe caps spurs' ε here: ε = 1 already reaches the bump's top.
SLIM_SPUR_EPS_MAX = 1.5
#: The prior term's name, after the loss terms.
CONNECTION = "connection"


@dataclass(frozen=True)
class ProbeSpec:
    """What a probe measures: a loss, its relaxation and the error paths.

    The loss part comes from a ``scratch_fit`` variant (:func:`probe_spec`);
    the rest are the probe's own settings. A probe is identified by the
    hash of this spec, like a variant.

    Parameters
    ----------
    target : TargetConfig
        The field the loss compares the rendering with.
    losses : tuple of LossTerm
        The loss terms and their weights.
    render : RenderConfig
        The renderer at the fit's last iteration (its union must be
        ``"max"``, so that a crop's render can be maxed with the rest).
    connection_weight, connection_reach : float
        The endpoint-connection prior. Its pairs are recomputed for every
        configuration (Section 10.4).
    position_lr, radius_lr, opacity_lr : float
        Adam's learning rates in every relaxation (the fit's).
    families : tuple of str
        The error families probed.
    eps : tuple of float
        The ε grid.
    reference_iterations : int
        Adam steps of the reference's relaxation on the full volume.
    profile_iterations : int
        Adam steps of every profile relaxation, on the crop.
    slim : bool
        The slim probe (Section 10.5): profiles only at ε in
        :data:`SLIM_EPS` and the largest uncollapsed ε, spurs' ε capped at
        1.5, new primitives at opacity 1 only, and only thin junction
        branches.
    max_events : int
        Events per family and volume, at most.
    mid_edge_points : int
        Random mid-edge points per volume (``break_retract`` takes as many
        windows of its own; ``edge_shift``, ``duplicate`` and
        ``radius_scale`` share theirs).
    free_radii : float
        Free segments lie within this many local radii of the event's path.
    jitter : float
        For the floor: the noise added to the free endpoints of the
        reference configuration before each of its two relaxations, in
        physical units.
    """

    target: TargetConfig
    losses: tuple[LossTerm, ...]
    render: RenderConfig
    connection_weight: float
    connection_reach: float
    position_lr: float
    radius_lr: float
    opacity_lr: float
    families: tuple[str, ...] = FAMILIES
    eps: tuple[float, ...] = EPS_GRID
    reference_iterations: int = 200
    profile_iterations: int = 100
    slim: bool = False
    max_events: int = 10
    mid_edge_points: int = 3
    free_radii: float = 3.0
    jitter: float = 1e-3

    def __post_init__(self) -> None:
        """Check the families and the ε grid."""
        unknown = sorted(set(self.families) - set(FAMILIES))
        if unknown:
            raise ValueError(f"unknown families {unknown}; known: {FAMILIES}")
        if self.render.union != "max":
            raise ValueError(
                "the probe needs the max union (a crop's render is maxed with the "
                f"rest of the volume), not {self.render.union!r}"
            )
        if 0.0 not in self.eps or list(self.eps) != sorted(set(self.eps)):
            raise ValueError(f"the ε grid must be sorted and contain 0, got {self.eps}")

    @property
    def terms(self) -> tuple[str, ...]:
        """The loss terms' names, then the connection prior's."""
        return (*(term.name for term in self.losses), CONNECTION)

    @property
    def weights(self) -> tuple[float, ...]:
        """The weight of each of :attr:`terms`."""
        return (*(term.weight for term in self.losses), self.connection_weight)

    def family_eps(self, family: str) -> tuple[float, ...]:
        """A family's ε grid: spurs' capped in the slim probe."""
        if self.slim and family == "bump_spur":
            return tuple(e for e in self.eps if e <= SLIM_SPUR_EPS_MAX)
        return self.eps

    def profile_eps(self, grid: tuple[float, ...], collapsed=()) -> tuple[float, ...]:
        """The ε whose profiles are relaxed (all but collapsed ones, or slim's)."""
        usable = [e for e in grid if e not in collapsed]
        if not self.slim:
            return tuple(usable)
        largest = max(usable)
        return tuple(sorted({*(e for e in SLIM_EPS if e <= largest), largest}))

    @property
    def new_slot_opacities(self) -> tuple[float, ...]:
        """Opacities of new primitives (spurs and duplicates)."""
        return (1.0,) if self.slim else (1.0, 0.5)

    @property
    def thin_only(self) -> bool:
        """Whether only thin junction branches are probed."""
        return self.slim


def probe_spec(
    variant: Variant | NamedVariant,
    slim: bool = False,
    families: tuple[str, ...] | None = None,
    eps: tuple[float, ...] | None = None,
    **settings,
) -> ProbeSpec:
    """The probe spec of a ``scratch_fit`` variant.

    Parameters
    ----------
    variant : Variant or NamedVariant
        The variant whose loss is probed.
    slim : bool
        The slim probe (the default for screening).
    families : tuple of str, optional
        The families; by default all, less those that do not apply to the
        variant's render mode (:data:`CENTRELINE_SKIPS`).
    eps : tuple of float, optional
        The ε grid; :data:`EPS_GRID` by default.
    **settings
        Other :class:`ProbeSpec` fields.

    Returns
    -------
    ProbeSpec
    """
    variant = variant.variant if isinstance(variant, NamedVariant) else variant
    variant = variant.resolved()
    if variant.pipeline != "scratch_fit":
        raise ValueError(
            f"the probe needs a scratch_fit variant (its connection prior), "
            f"not {variant.pipeline}"
        )
    fit = variant.fit.fit
    render = fit.render_config(max(fit.n_iterations - 1, 0))
    render = dataclasses.replace(render, temperature=0.0)
    if families is None:
        skips = CENTRELINE_SKIPS if render.mode == "centreline" else ()
        families = tuple(f for f in FAMILIES if f not in skips)
    return ProbeSpec(
        target=variant.target,
        losses=fit.losses,
        render=render,
        connection_weight=variant.fit.connection_weight,
        connection_reach=variant.fit.connection_reach,
        position_lr=fit.position_lr,
        radius_lr=fit.radius_lr,
        opacity_lr=fit.opacity_lr,
        families=tuple(families),
        eps=EPS_GRID if eps is None else tuple(eps),
        slim=slim,
        **settings,
    )
