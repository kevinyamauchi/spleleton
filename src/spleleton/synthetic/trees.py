"""Random airway-like trees with known skeleton graphs.

Trees grow recursively from a root branch (the "trachea"), which enters the
domain at the low end of axis 0 and points along +axis 0:

- each branch has length ``length_ratio * diameter`` (Weibel: about 3), with
  jitter, but never less than ``min_length_ratio`` radii, and tapers by
  ``taper`` along its length;
- at each junction the child radii follow Murray's law,
  ``r_parent^n = sum(r_child^n)``, with a sampled asymmetry, and the child with
  the smaller radius takes the larger branching angle;
- in 3D, the branching plane rotates by about 90 degrees per generation.

A set of children is placed all-or-nothing: if any child leaves the domain or
comes too close to an existing branch (after ``max_attempts`` resamples), the
node becomes a leaf. This guarantees that every node has degree 1 or at least
3, so straight trees have exactly straight edges.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np

from spleleton.graph import SkeletonGraph, resample_polyline

#: A child at most this thick, relative to its parent, is thin.
_THIN_RATIO = 0.5


@dataclass(frozen=True)
class TreeConfig:
    """Parameters of the airway-like tree generator.

    Lengths and radii are physical (voxel units when the spacing is 1).

    Parameters
    ----------
    root_radius : float
        Radius at the start of the root branch.
    min_radius : float
        Smallest radius anywhere in the tree. A junction whose thinnest child
        would taper below this becomes a leaf instead.
    length_ratio : float
        Branch length divided by branch diameter at its start.
    length_jitter : float
        Relative jitter of branch lengths: lengths are scaled by
        ``U(1 - length_jitter, 1 + length_jitter)``.
    min_length_ratio : float
        Minimum branch length, in units of the branch's start radius. It
        applies to every branch, including retries and short branches. A spur
        caused by a bump reaches at most about 2.5 radii from the centreline,
        so the default of 4 keeps every true leaf clearly longer than a spur.
    taper : float
        Ratio of end radius to start radius within a branch.
    murray_exponent : float
        Exponent ``n`` of Murray's law.
    min_asymmetry : float
        Lower bound of the radius ratio between the minor and major child,
        sampled uniformly in ``[min_asymmetry, 1]``.
    branch_angle : tuple of float
        Range of branching angles in degrees, measured from the parent's
        direction.
    plane_rotation : float
        Rotation of the branching plane per generation in degrees (3D only).
    plane_rotation_jitter : float
        Uniform jitter of ``plane_rotation`` in degrees (3D only).
    tortuosity : float
        Amplitude of the smooth lateral wiggle of a branch, relative to its
        length. ``0`` gives exactly straight branches.
    trifurcation_probability : float
        Probability that a junction has three children instead of two.
    short_branch_probability : float
        Probability that a branch is shortened by ``short_branch_factor``,
        creating closely spaced junctions.
    short_branch_factor : tuple of float
        Range of the shortening factor.
    side_branch_probability : float
        Probability that a bifurcation gets a thin third child (a side
        branch) next to its usual pair, making thin-thick junctions. ``0``
        (the default) draws nothing extra, so trees are unchanged.
    side_branch_ratio : tuple of float
        Range of a side branch's radius relative to the major child's,
        before Murray's law rescales all three children. A draw that would
        taper below ``min_radius`` is raised to the thinnest the parent
        allows, if that is still within the range; otherwise the junction
        stays a bifurcation.
    thin_min_length : float
        With side branches on, a thin child (radius at most half its
        parent's) is at least ``r_parent + thin_min_length * r_child`` long,
        so it reaches well past its parent's wall.
    clearance : float
        Minimum gap between the surfaces of branches that do not share a node.
    margin : float
        Minimum gap between a branch surface and the domain border (the root
        branch may touch the border at axis 0).
    max_generations : int
        Maximum number of generations below the root branch.
    max_attempts : int
        Number of resamples before a junction gives up and becomes a leaf.
        Each resample shortens the children by a further factor of 0.85.
    path_spacing : float
        Vertex spacing of curved branch polylines.
    """

    root_radius: float = 8.0
    min_radius: float = 1.5
    length_ratio: float = 3.0
    length_jitter: float = 0.3
    min_length_ratio: float = 4.0
    taper: float = 0.9
    murray_exponent: float = 3.0
    min_asymmetry: float = 0.6
    branch_angle: tuple[float, float] = (20.0, 45.0)
    plane_rotation: float = 90.0
    plane_rotation_jitter: float = 20.0
    tortuosity: float = 0.0
    trifurcation_probability: float = 0.0
    short_branch_probability: float = 0.0
    short_branch_factor: tuple[float, float] = (0.2, 0.5)
    side_branch_probability: float = 0.0
    side_branch_ratio: tuple[float, float] = (0.2, 0.5)
    thin_min_length: float = 4.0
    clearance: float = 2.0
    margin: float = 2.0
    max_generations: int = 10
    max_attempts: int = 10
    path_spacing: float = 1.0


def generate_tree(
    config: TreeConfig, extent: np.ndarray, rng: np.random.Generator
) -> SkeletonGraph:
    """Generate a random airway-like tree.

    Parameters
    ----------
    config : TreeConfig
        Generator parameters.
    extent : np.ndarray
        Physical size of the domain per axis, shape ``(D,)`` with D = 2 or 3.
        The domain is ``[0, extent]`` on each axis.
    rng : np.random.Generator
        Source of randomness.

    Returns
    -------
    SkeletonGraph
        The tree, with edges oriented from parent to child and radii along
        every edge. Node 0 is the root.
    """
    return _TreeBuilder(config, np.asarray(extent, dtype=float), rng).build()


@dataclass
class _Branch:
    path: np.ndarray
    radii: np.ndarray
    end_direction: np.ndarray
    plane: np.ndarray | None


class _TreeBuilder:
    def __init__(
        self, config: TreeConfig, extent: np.ndarray, rng: np.random.Generator
    ) -> None:
        if extent.shape not in ((2,), (3,)):
            raise ValueError(f"extent must have 2 or 3 entries, got {extent.shape}")
        self.config = config
        self.extent = extent
        self.ndim = len(extent)
        self.rng = rng
        self.nodes: list[np.ndarray] = []
        self.edges: list[tuple[int, int]] = []
        self.paths: list[np.ndarray] = []
        self.radii: list[np.ndarray] = []
        self.incident: list[list[int]] = []
        # dense samples of every edge, for clearance checks
        self.sample_spacing = 0.5 * min(config.path_spacing, config.min_radius)
        self.samples: list[np.ndarray] = []
        self.sample_radii: list[np.ndarray] = []

    def build(self) -> SkeletonGraph:
        config = self.config
        root = np.zeros(self.ndim)
        root[1:] = self.extent[1:] / 2
        self._add_node(root)
        direction = np.eye(self.ndim)[0]
        plane = self._random_perpendicular(direction) if self.ndim == 3 else None

        queue = deque()
        length = self._branch_length(config.root_radius)
        for _ in range(config.max_attempts):
            branch = self._make_branch(
                root, direction, config.root_radius, length, plane
            )
            if self._is_valid(branch, start_node=0, is_root=True):
                child = self._commit(0, branch)
                queue.append((child, branch, 0))
                break
            length *= 0.8

        while queue:
            node, parent_branch, generation = queue.popleft()
            if generation >= config.max_generations:
                continue
            for child, branch in self._spawn_children(node, parent_branch):
                queue.append((child, branch, generation + 1))

        return SkeletonGraph(
            node_positions=np.array(self.nodes).reshape(-1, self.ndim),
            edges=np.array(self.edges, dtype=np.int64).reshape(-1, 2),
            edge_paths=self.paths,
            edge_radii=self.radii,
        )

    def _spawn_children(self, node: int, parent: _Branch) -> list[tuple[int, _Branch]]:
        config = self.config
        start = self.nodes[node]
        for attempt in range(config.max_attempts):
            trifurcation = self.rng.random() < config.trifurcation_probability
            radii = self._child_radii(parent.radii[-1], 3 if trifurcation else 2)
            if radii.min() * config.taper < config.min_radius:
                return []
            # later attempts try shorter branches, to fit into crowded regions
            length_scale = 0.85**attempt
            # thinner children branch off at larger angles
            angles = np.sort(
                np.deg2rad(self.rng.uniform(*config.branch_angle, len(radii)))
            )
            directions = self._child_directions(
                parent.end_direction, parent.plane, angles
            )
            branches = []
            for radius, direction in zip(radii, directions, strict=True):
                plane = self._child_plane(parent.plane, direction)
                length = length_scale * self._branch_length(radius)
                if config.side_branch_probability > 0:
                    length = self._thin_length(length, radius, parent.radii[-1])
                branch = self._make_branch(start, direction, radius, length, plane)
                if not self._is_valid(branch, start_node=node, is_root=False):
                    break
                branches.append(branch)
            else:
                return [(self._commit(node, branch), branch) for branch in branches]
        return []

    def _child_radii(self, parent_radius: float, n_children: int) -> np.ndarray:
        """Child radii from Murray's law, sorted from thickest to thinnest."""
        exponent = self.config.murray_exponent
        weights = np.concatenate(
            [[1.0], self.rng.uniform(self.config.min_asymmetry, 1.0, n_children - 1)]
        )
        radii = parent_radius * weights / np.sum(weights**exponent) ** (1 / exponent)
        radii = np.sort(radii)[::-1]
        config = self.config
        if config.side_branch_probability == 0 or n_children != 2:
            return radii
        if self.rng.random() >= config.side_branch_probability:
            return radii
        # a thin side branch next to the pair; weights relative to the major child
        weights = np.append(
            radii / radii[0], self.rng.uniform(*config.side_branch_ratio)
        )
        # Murray's law gives the thin child parent * w / norm(weights): raise w
        # to the least that tapers to min_radius (norm depends on w: iterate)
        lowest = config.min_radius / (config.taper * parent_radius)
        for _ in range(10):
            norm = np.sum(weights**exponent) ** (1 / exponent)
            weights[-1] = max(weights[-1], 1.0001 * lowest * norm)
        if weights[-1] > config.side_branch_ratio[1]:
            return radii  # the parent is too thin for a side branch
        radii = parent_radius * weights / np.sum(weights**exponent) ** (1 / exponent)
        return np.sort(radii)[::-1]

    def _thin_length(self, length: float, radius: float, parent_radius: float) -> float:
        """The length of a child, extended if it is thin (see ``TreeConfig``)."""
        if radius / parent_radius > _THIN_RATIO:
            return length
        return max(length, parent_radius + self.config.thin_min_length * radius)

    def _child_directions(
        self, direction: np.ndarray, plane: np.ndarray | None, angles: np.ndarray
    ) -> list[np.ndarray]:
        """Unit child directions; ``angles[i]`` is the angle of child ``i``."""
        n_children = len(angles)
        if self.ndim == 2:
            normal = np.array([-direction[1], direction[0]])
            if n_children == 2:
                side = self.rng.choice([-1.0, 1.0])
                sides = [side, -side]
            else:
                # the thickest child continues roughly straight on
                angles = angles.copy()
                angles[0] = np.deg2rad(self.rng.uniform(-5.0, 5.0))
                side = self.rng.choice([-1.0, 1.0])
                sides = [1.0, side, -side]
            return [
                np.cos(angle) * direction + s * np.sin(angle) * normal
                for angle, s in zip(angles, sides, strict=True)
            ]

        u = plane
        w = np.cross(direction, u)
        if n_children == 2:
            side = self.rng.choice([-1.0, 1.0])
            azimuths = [0.0 if side > 0 else np.pi, np.pi if side > 0 else 0.0]
        else:
            offset = self.rng.uniform(0, 2 * np.pi)
            azimuths = [offset + 2 * np.pi * k / 3 for k in range(3)]
        return [
            np.cos(angle) * direction
            + np.sin(angle) * (np.cos(azimuth) * u + np.sin(azimuth) * w)
            for angle, azimuth in zip(angles, azimuths, strict=True)
        ]

    def _child_plane(
        self, plane: np.ndarray | None, direction: np.ndarray
    ) -> np.ndarray | None:
        """Rotate the parent's branching plane about the child's direction."""
        if plane is None:
            return None
        projected = plane - np.dot(plane, direction) * direction
        norm = np.linalg.norm(projected)
        projected = (
            projected / norm if norm > 1e-8 else self._random_perpendicular(direction)
        )
        rotation = np.deg2rad(
            self.config.plane_rotation
            + self.rng.uniform(-1, 1) * self.config.plane_rotation_jitter
        )
        return np.cos(rotation) * projected + np.sin(rotation) * np.cross(
            direction, projected
        )

    def _branch_length(self, radius: float) -> float:
        config = self.config
        jitter = self.rng.uniform(1 - config.length_jitter, 1 + config.length_jitter)
        length = config.length_ratio * 2 * radius * jitter
        if self.rng.random() < config.short_branch_probability:
            length *= self.rng.uniform(*config.short_branch_factor)
        return length

    def _make_branch(
        self,
        start: np.ndarray,
        direction: np.ndarray,
        radius: float,
        length: float,
        plane: np.ndarray | None,
    ) -> _Branch:
        config = self.config
        length = max(length, config.min_length_ratio * radius)
        end = start + length * direction
        end_radius = radius * config.taper
        if config.tortuosity == 0:
            return _Branch(
                path=np.stack([start, end]),
                radii=np.array([radius, end_radius]),
                end_direction=direction,
                plane=plane,
            )

        n_points = max(3, int(np.ceil(length / config.path_spacing)) + 1)
        s = np.linspace(0.0, 1.0, n_points)
        path = start + s[:, None] * (end - start)
        for normal in self._perpendicular_basis(direction):
            # smooth wiggle with fixed endpoints; higher modes are weaker
            modes = np.arange(1, 4)
            amplitudes = self.rng.normal(0.0, config.tortuosity * length / modes)
            offset = np.sin(np.pi * s[:, None] * modes) @ amplitudes
            path += offset[:, None] * normal
        arc = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        )
        radii = radius + (end_radius - radius) * arc / arc[-1]
        end_direction = path[-1] - path[-2]
        end_direction /= np.linalg.norm(end_direction)
        return _Branch(path=path, radii=radii, end_direction=end_direction, plane=plane)

    def _is_valid(self, branch: _Branch, start_node: int, is_root: bool) -> bool:
        config = self.config
        points, radii = resample_polyline(
            branch.path, self.sample_spacing, branch.radii
        )

        lower = np.full_like(points, config.margin) + radii[:, None]
        upper = self.extent - config.margin - radii[:, None]
        if is_root:
            lower[:, 0] = 0.0
        if np.any(points < lower) or np.any(points > upper):
            return False

        excluded = set(self.incident[start_node])
        others = [i for i in range(len(self.edges)) if i not in excluded]
        if not others:
            return True
        other_points = np.concatenate([self.samples[i] for i in others])
        other_radii = np.concatenate([self.sample_radii[i] for i in others])
        for chunk in range(0, len(points), 256):
            chunk_points = points[chunk : chunk + 256]
            chunk_radii = radii[chunk : chunk + 256]
            distance = np.linalg.norm(
                chunk_points[:, None, :] - other_points[None, :, :], axis=-1
            )
            required = chunk_radii[:, None] + other_radii[None, :] + config.clearance
            if np.any(distance < required):
                return False
        return True

    def _commit(self, start_node: int, branch: _Branch) -> int:
        end_node = self._add_node(branch.path[-1])
        edge = len(self.edges)
        self.edges.append((start_node, end_node))
        self.paths.append(branch.path)
        self.radii.append(branch.radii)
        self.incident[start_node].append(edge)
        self.incident[end_node].append(edge)
        points, radii = resample_polyline(
            branch.path, self.sample_spacing, branch.radii
        )
        self.samples.append(points)
        self.sample_radii.append(radii)
        return end_node

    def _add_node(self, position: np.ndarray) -> int:
        self.nodes.append(np.asarray(position, dtype=float))
        self.incident.append([])
        return len(self.nodes) - 1

    def _random_perpendicular(self, direction: np.ndarray) -> np.ndarray:
        vector = self.rng.normal(size=self.ndim)
        vector -= np.dot(vector, direction) * direction
        return vector / np.linalg.norm(vector)

    def _perpendicular_basis(self, direction: np.ndarray) -> list[np.ndarray]:
        if self.ndim == 2:
            return [np.array([-direction[1], direction[0]])]
        axis = np.eye(3)[np.argmin(np.abs(direction))]
        u = axis - np.dot(axis, direction) * direction
        u /= np.linalg.norm(u)
        return [u, np.cross(direction, u)]
