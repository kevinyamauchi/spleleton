import dataclasses
import json

import numpy as np
import pytest
from scipy import ndimage

from spleleton.graph import SkeletonGraph
from spleleton.synthetic.corruptions import (
    BumpConfig,
    ContactConfig,
    CorruptionConfig,
    GapConfig,
    HoleConfig,
    SurfaceNoiseConfig,
    corrupt,
)
from spleleton.synthetic.dataset import (
    Sample,
    SampleConfig,
    airway_config,
    generate_dataset,
    generate_sample,
    load_dataset,
)
from spleleton.synthetic.rasterize import rasterize_graph
from spleleton.synthetic.trees import TreeConfig, generate_tree

# ---------------------------------------------------------------------- trees


@pytest.mark.parametrize("extent", [(127.0, 127.0), (95.0, 95.0, 95.0)])
@pytest.mark.parametrize("seed", range(3))
def test_tree_structure(extent, seed):
    config = TreeConfig()
    graph = generate_tree(config, np.array(extent), np.random.default_rng(seed))
    assert graph.ndim == len(extent)
    # a tree: connected and acyclic
    assert graph.n_edges == graph.n_nodes - 1
    assert graph.n_components() == 1
    # the root is an end, every other node an end or a bifurcation
    assert graph.degrees[0] == 1
    assert set(graph.degrees) <= {1, 3}
    # straight branches by default
    assert all(len(path) == 2 for path in graph.edge_paths)
    # radii respect the floor, and tubes stay inside the domain (the root
    # branch may touch the border at axis 0)
    for edge, (path, radii) in enumerate(
        zip(graph.edge_paths, graph.edge_radii, strict=True)
    ):
        assert radii.min() >= config.min_radius - 1e-9
        lower = path - radii[:, None]
        upper = path + radii[:, None]
        if edge > 0:
            assert np.all(lower >= config.margin - 1e-9)
        assert np.all(upper <= np.array(extent) - config.margin + 1e-9)


@pytest.mark.parametrize("extent", [(127.0, 127.0), (95.0, 95.0, 95.0)])
def test_branches_respect_the_minimum_length(extent):
    config = TreeConfig(short_branch_probability=0.5)
    for seed in range(5):
        graph = generate_tree(config, np.array(extent), np.random.default_rng(seed))
        start_radii = np.array([radii[0] for radii in graph.edge_radii])
        assert np.all(
            graph.edge_lengths >= config.min_length_ratio * start_radii - 1e-9
        )


def test_tree_follows_murrays_law():
    config = TreeConfig()
    graph = generate_tree(config, np.array([127.0, 127.0]), np.random.default_rng(0))
    for junction in graph.junction_nodes:
        incoming = np.flatnonzero(graph.edges[:, 1] == junction)
        outgoing = np.flatnonzero(graph.edges[:, 0] == junction)
        assert len(incoming) == 1
        parent = graph.edge_radii[incoming[0]][-1]
        children = np.array([graph.edge_radii[e][0] for e in outgoing])
        np.testing.assert_allclose(np.sum(children**3), parent**3)


def test_tree_is_deterministic():
    extent = np.array([127.0, 127.0])
    a = generate_tree(TreeConfig(), extent, np.random.default_rng(5))
    b = generate_tree(TreeConfig(), extent, np.random.default_rng(5))
    np.testing.assert_array_equal(a.node_positions, b.node_positions)


def test_curved_tree():
    graph = generate_tree(
        TreeConfig(tortuosity=0.05), np.array([127.0, 127.0]), np.random.default_rng(0)
    )
    assert all(len(path) > 2 for path in graph.edge_paths)
    assert np.any(
        graph.edge_lengths
        > np.linalg.norm(
            graph.node_positions[graph.edges[:, 1]]
            - graph.node_positions[graph.edges[:, 0]],
            axis=1,
        )
        + 0.1
    )


@pytest.mark.parametrize("extent", [(127.0, 127.0), (95.0, 95.0, 95.0)])
def test_trifurcations(extent):
    graph = generate_tree(
        TreeConfig(trifurcation_probability=1.0),
        np.array(extent),
        np.random.default_rng(0),
    )
    assert set(graph.degrees) <= {1, 4}
    assert 4 in set(graph.degrees)


# ---------------------------------------------------------------- rasterising


def _straight_tube(ndim, radius=4.0, length=30.0, offset=0.0):
    start = np.full(ndim, 20.0) + offset
    end = start.copy()
    end[0] += length
    return SkeletonGraph(
        np.stack([start, end]),
        [[0, 1]],
        [np.stack([start, end])],
        [np.array([radius, radius])],
    )


# A tube centred exactly on voxel centres with an integer radius is a worst
# case for voxelisation (whole rings of voxels sit at exactly 50% coverage), so
# sizes are averaged over sub-voxel placements.
_OFFSETS = np.linspace(0.0, 1.0, 5, endpoint=False)


def test_rasterized_area_2d():
    radius, length = 4.0, 30.0
    sizes = [
        rasterize_graph(_straight_tube(2, radius, length, offset), (80, 50)).sum()
        for offset in _OFFSETS
    ]
    expected = 2 * radius * length + np.pi * radius**2
    assert abs(np.mean(sizes) - expected) / expected < 0.03


def test_rasterized_volume_3d():
    radius, length = 4.0, 30.0
    sizes = [
        rasterize_graph(_straight_tube(3, radius, length, offset), (80, 50, 50)).sum()
        for offset in _OFFSETS
    ]
    expected = np.pi * radius**2 * length + 4 / 3 * np.pi * radius**3
    assert abs(np.mean(sizes) - expected) / expected < 0.03


def test_rasterized_with_anisotropic_spacing():
    radius, length = 4.0, 30.0
    graph = _straight_tube(2, radius, length)
    mask = rasterize_graph(graph, (80, 25), spacing=(1.0, 2.0))
    expected = (2 * radius * length + np.pi * radius**2) / 2.0
    assert abs(mask.sum() - expected) / expected < 0.05


def test_centrelines_lie_inside_the_mask():
    graph = generate_tree(
        TreeConfig(), np.array([127.0, 127.0]), np.random.default_rng(1)
    )
    mask = rasterize_graph(graph, (128, 128))
    points, _, _ = graph.sample_points(0.5)
    indices = np.clip(np.round(points).astype(int), 0, 127)
    assert mask[tuple(indices.T)].all()


# ---------------------------------------------------------------- corruptions


def _tree_and_mask(seed=0):
    graph = generate_tree(
        TreeConfig(), np.array([127.0, 127.0]), np.random.default_rng(seed)
    )
    return graph, rasterize_graph(graph, (128, 128))


def test_no_corruptions_is_identity():
    graph, mask = _tree_and_mask()
    corrupted, records = corrupt(
        mask, graph, CorruptionConfig(), np.random.default_rng(0)
    )
    np.testing.assert_array_equal(corrupted, mask)
    assert records == []


def test_gaps_disconnect_the_mask():
    graph, mask = _tree_and_mask()
    corrupted, records = corrupt(
        mask, graph, CorruptionConfig(gaps=GapConfig(count=2)), np.random.default_rng(0)
    )
    assert [r["type"] for r in records] == ["gap", "gap"]
    assert not np.any(corrupted & ~mask)
    assert ndimage.label(corrupted)[1] == 3


def test_bumps_only_add_and_stay_attached():
    graph, mask = _tree_and_mask()
    corrupted, records = corrupt(
        mask,
        graph,
        CorruptionConfig(bumps=BumpConfig(count=6)),
        np.random.default_rng(0),
    )
    assert len(records) == 6
    assert not np.any(mask & ~corrupted)
    assert corrupted.sum() > mask.sum()
    assert ndimage.label(corrupted)[1] == 1


def test_holes_only_remove_interior_voxels():
    graph, mask = _tree_and_mask()
    corrupted, records = corrupt(
        mask,
        graph,
        CorruptionConfig(holes=HoleConfig(count=3)),
        np.random.default_rng(0),
    )
    assert records
    assert not np.any(corrupted & ~mask)
    # holes are cavities: the background stays one component (in 2D, the
    # holes are extra background components)
    assert ndimage.label(corrupted)[1] == 1
    assert ndimage.label(~corrupted)[1] == 1 + len(records)


def test_surface_noise_changes_only_the_boundary():
    graph, mask = _tree_and_mask()
    corrupted, _ = corrupt(
        mask,
        graph,
        CorruptionConfig(surface_noise=SurfaceNoiseConfig(amplitude=0.75)),
        np.random.default_rng(0),
    )
    changed = corrupted != mask
    assert changed.any()
    boundary_distance = np.where(
        mask,
        ndimage.distance_transform_edt(mask),
        ndimage.distance_transform_edt(~mask),
    )
    # 0.75 is one standard deviation: changes stay within a few voxels
    assert boundary_distance[changed].max() <= 4


def test_contacts_bridge_nearby_branches():
    # two branches that diverge at a right angle, then bend back so that
    # their tips nearly touch
    nodes = np.array([[0.0, 30.0], [20.0, 30.0], [50.0, 27.0], [50.0, 33.0]])
    edges = np.array([[0, 1], [1, 2], [1, 3]])
    paths = [
        nodes[[0, 1]],
        np.array([[20.0, 30.0], [35.0, 15.0], [50.0, 27.0]]),
        np.array([[20.0, 30.0], [35.0, 45.0], [50.0, 33.0]]),
    ]
    radii = [np.full(2, 3.0), np.full(3, 2.5), np.full(3, 2.5)]
    graph = SkeletonGraph(nodes, edges, paths, radii)
    mask = rasterize_graph(graph, (60, 60))
    # the two tips are separate within the region x >= 40
    tips = mask.copy()
    tips[:40] = False
    assert ndimage.label(tips)[1] == 2
    corrupted, records = corrupt(
        mask,
        graph,
        CorruptionConfig(contacts=ContactConfig()),
        np.random.default_rng(0),
    )
    assert [r["type"] for r in records] == ["contact"]
    assert records[0]["edges"] == [1, 2]
    assert records[0]["start"][0] > 40
    tips = corrupted.copy()
    tips[:40] = False
    assert ndimage.label(tips)[1] == 1


def test_contacts_skip_points_joined_through_a_junction():
    nodes = np.array([[0.0, 30.0], [20.0, 30.0], [50.0, 22.0], [50.0, 38.0]])
    edges = np.array([[0, 1], [1, 2], [1, 3]])
    graph = SkeletonGraph(
        nodes,
        edges,
        [nodes[[u, v]] for u, v in edges],
        [np.array([3.0, 3.0]), np.array([2.5, 2.5]), np.array([2.5, 2.5])],
    )
    mask = rasterize_graph(graph, (60, 60))
    _, records = corrupt(
        mask,
        graph,
        CorruptionConfig(contacts=ContactConfig(max_gap=12.0, min_path_ratio=5.0)),
        np.random.default_rng(0),
    )
    assert records == []


# -------------------------------------------------------------------- dataset


def test_config_round_trip():
    config = airway_config(3, spacing=(2.0, 1.0, 1.0))
    restored = SampleConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored == config
    assert isinstance(restored.corruptions.gaps.width, tuple)


def test_generate_sample_is_deterministic():
    config = airway_config(2)
    a, b = generate_sample(config, 3), generate_sample(config, 3)
    np.testing.assert_array_equal(a.mask, b.mask)
    assert a.corruptions == b.corruptions


def test_corruptions_do_not_change_the_tree():
    clean = generate_sample(airway_config(2, corrupted=False), 4)
    corrupted = generate_sample(airway_config(2), 4)
    np.testing.assert_array_equal(
        clean.graph.node_positions, corrupted.graph.node_positions
    )
    np.testing.assert_array_equal(clean.clean_mask, corrupted.clean_mask)
    np.testing.assert_array_equal(clean.clean_mask, clean.mask)


def test_anisotropic_sample():
    config = airway_config(3, shape=(48, 96, 96), spacing=(2.0, 1.0, 1.0))
    sample = generate_sample(config, 0)
    assert sample.mask.shape == (48, 96, 96)
    assert sample.graph.node_positions[:, 0].max() <= 94.0


def test_dataset_round_trip(tmp_path):
    config = airway_config(2)
    paths = generate_dataset(config, tmp_path, n_samples=3, seed=10)
    assert [p.name for p in paths] == [f"sample_{i:04d}.npz" for i in range(3)]
    samples = load_dataset(tmp_path)
    for index, sample in enumerate(samples):
        expected = generate_sample(config, 10 + index)
        assert sample.seed == 10 + index
        assert sample.config == config
        assert sample.corruptions == expected.corruptions
        np.testing.assert_array_equal(sample.mask, expected.mask)
        np.testing.assert_array_equal(sample.clean_mask, expected.clean_mask)
        np.testing.assert_allclose(
            sample.graph.node_positions, expected.graph.node_positions
        )


def test_sample_save_adds_suffix(tmp_path):
    sample = generate_sample(dataclasses.replace(airway_config(2), shape=(64, 64)), 0)
    path = sample.save(tmp_path / "one")
    assert path.suffix == ".npz"
    assert Sample.load(path).seed == 0


# ------------------------------------------------- generator additions (bench)


def _side_branch_junctions(graph):
    """(r_parent, child radii, child lengths) of every degree-4 junction."""
    lengths = graph.edge_lengths
    for node in np.flatnonzero(graph.degrees == 4):
        (parent,) = np.flatnonzero(graph.edges[:, 1] == node)
        children = np.flatnonzero(graph.edges[:, 0] == node)
        radii = np.array([graph.edge_radii[e][0] for e in children])
        yield float(graph.edge_radii[parent][-1]), radii, lengths[children]


def test_side_branches_make_thin_thick_junctions():
    config = TreeConfig(side_branch_probability=1.0)
    found = 0
    for seed in range(4):
        graph = generate_tree(config, np.full(3, 127.0), np.random.default_rng(seed))
        assert graph.n_edges == graph.n_nodes - 1
        assert graph.n_components() == 1
        for r_parent, radii, lengths in _side_branch_junctions(graph):
            thin = np.argmin(radii)
            # relative to the major child, within the configured range
            ratio = radii[thin] / radii.max()
            assert 0.2 - 1e-9 <= ratio <= 0.5 + 1e-9
            assert radii[thin] / r_parent <= 0.5
            assert radii[thin] * config.taper >= config.min_radius
            # long enough to reach well past the parent's wall
            assert (
                lengths[thin] >= r_parent + config.thin_min_length * radii[thin] - 1e-9
            )
            found += 1
    assert found >= 8


def test_side_branch_settings_are_inert_when_off():
    extent = np.full(3, 127.0)
    off = TreeConfig(side_branch_ratio=(0.3, 0.4), thin_min_length=9.0)
    for seed in range(3):
        a = generate_tree(TreeConfig(), extent, np.random.default_rng(seed))
        b = generate_tree(off, extent, np.random.default_rng(seed))
        np.testing.assert_array_equal(a.node_positions, b.node_positions)
        np.testing.assert_array_equal(a.edges, b.edges)
        assert 4 not in a.degrees


def test_bump_sweep_levels_are_paired():
    # the same tree, and bumps in the same places, at every height
    base = airway_config(3, corrupted=False)

    def bumps(height):
        config = dataclasses.replace(
            base,
            corruptions=CorruptionConfig(
                bumps=BumpConfig(count=6, height=(height, height), size=(1.0, 1.0))
            ),
        )
        return generate_sample(config, 3)

    low, high = bumps(0.5), bumps(1.5)
    clean = generate_sample(base, 3)
    for sample in (low, high):
        np.testing.assert_array_equal(sample.clean_mask, clean.clean_mask)
        np.testing.assert_array_equal(
            sample.graph.node_positions, clean.graph.node_positions
        )
    assert len(low.corruptions) == len(high.corruptions) == 6
    for a, b in zip(low.corruptions, high.corruptions, strict=True):
        # same place and size; the centre moves out by the extra height
        assert a["edge"] == b["edge"]
        assert a["radius"] == b["radius"]
        assert a["height"] == pytest.approx(0.5 * a["radius"])
        assert b["height"] == pytest.approx(1.5 * b["radius"])
        shift = np.linalg.norm(np.subtract(b["position"], a["position"]))
        assert shift == pytest.approx(b["height"] - a["height"])


def test_contacts_keep_a_minimum_gap():
    # the tips of two branches nearly touch (a gap of 1); with min_gap = 2
    # the bridge moves back to where the branches are 2 apart
    nodes = np.array([[0.0, 30.0], [20.0, 30.0], [50.0, 27.0], [50.0, 33.0]])
    edges = np.array([[0, 1], [1, 2], [1, 3]])
    paths = [
        nodes[[0, 1]],
        np.array([[20.0, 30.0], [35.0, 15.0], [50.0, 27.0]]),
        np.array([[20.0, 30.0], [35.0, 45.0], [50.0, 33.0]]),
    ]
    radii = [np.full(2, 3.0), np.full(3, 2.5), np.full(3, 2.5)]
    graph = SkeletonGraph(nodes, edges, paths, radii)
    mask = rasterize_graph(graph, (60, 60))

    def contact(**settings):
        corrupted, records = corrupt(
            mask,
            graph,
            CorruptionConfig(contacts=ContactConfig(**settings)),
            np.random.default_rng(0),
        )
        (record,) = records
        return corrupted, record

    near, near_record = contact()
    far, far_record = contact(min_gap=2.0)
    assert near_record["gap"] < 1.5
    assert "outside" not in near_record
    assert far_record["gap"] >= 2.0
    assert far_record["outside"] >= 2.0
    # the wider bridge adds more voxels
    assert (far & ~mask).sum() > (near & ~mask).sum()
    # with the default, nothing changes
    same, same_record = contact(min_gap=0.0)
    np.testing.assert_array_equal(same, near)
    assert same_record == near_record


def test_contacts_need_a_bridge_outside_the_tubes():
    # two separate parallel tubes, 1 apart everywhere: pairs 2 apart exist,
    # but only slanted, and their bridges run almost entirely inside the
    # tubes, so with min_gap = 2 there is no contact
    nodes = np.array([[5.0, 20.0], [55.0, 20.0], [5.0, 27.0], [55.0, 27.0]])
    edges = np.array([[0, 1], [2, 3]])
    graph = SkeletonGraph(
        nodes, edges, [nodes[[u, v]] for u, v in edges], [np.full(2, 3.0)] * 2
    )
    mask = rasterize_graph(graph, (60, 50))

    def records(**settings):
        return corrupt(
            mask,
            graph,
            CorruptionConfig(contacts=ContactConfig(**settings)),
            np.random.default_rng(0),
        )[1]

    (near,) = records()
    assert near["gap"] == pytest.approx(1.0)
    assert records(min_gap=2.0) == []
    (wide,) = records(min_gap=0.5)
    assert wide["outside"] >= 0.5
