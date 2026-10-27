"""Generate synthetic samples and save them to disk.

A sample is a ground-truth tree, its clean mask, the corrupted mask and a
record of the corruptions. Each sample is saved as one compressed ``.npz``
file, which also stores its config and seed as JSON, so it can be
regenerated exactly.
"""

import dataclasses
import json
import typing
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spleleton._serialise import from_dict
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
from spleleton.synthetic.rasterize import rasterize_graph
from spleleton.synthetic.trees import TreeConfig, generate_tree


@dataclass(frozen=True)
class SampleConfig:
    """Everything needed to generate one sample.

    Parameters
    ----------
    shape : tuple of int
        Shape of the image. Its length (2 or 3) sets the dimension.
    spacing : tuple of float, optional
        Physical voxel size per axis. Defaults to 1 on every axis.
    supersample : int
        Sub-samples per voxel and axis when rasterising.
    tree : TreeConfig
        Tree generator parameters.
    corruptions : CorruptionConfig
        Corruptions applied to the clean mask.
    """

    shape: tuple[int, ...] = (128, 128)
    spacing: tuple[float, ...] | None = None
    supersample: int = 3
    tree: TreeConfig = field(default_factory=TreeConfig)
    corruptions: CorruptionConfig = field(default_factory=CorruptionConfig)

    @property
    def extent(self) -> np.ndarray:
        """Physical size of the domain: from the first to the last voxel centre."""
        spacing = np.ones(len(self.shape)) if self.spacing is None else self.spacing
        return (np.asarray(self.shape) - 1) * np.asarray(spacing, dtype=float)

    def to_dict(self) -> dict:
        """Convert to plain Python types (e.g. for JSON)."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SampleConfig":
        """Rebuild from the output of :meth:`to_dict`."""
        return from_dict(cls, data)


def airway_config(
    ndim: int = 2, corrupted: bool = True, **overrides: typing.Any
) -> SampleConfig:
    """The airway-like preset.

    Parameters
    ----------
    ndim : int
        2 for a 128 x 128 image, 3 for a 128 x 128 x 128 volume (trees of
        about 21 nodes).
    corrupted : bool
        Apply every corruption with its default strength. Otherwise the
        corrupted mask equals the clean one.
    **overrides
        Replace any field of :class:`SampleConfig`.

    Returns
    -------
    SampleConfig
        The preset config.
    """
    if ndim not in (2, 3):
        raise ValueError(f"ndim must be 2 or 3, got {ndim}")
    corruptions = (
        CorruptionConfig(
            surface_noise=SurfaceNoiseConfig(),
            bumps=BumpConfig(),
            contacts=ContactConfig(),
            holes=HoleConfig(),
            gaps=GapConfig(),
        )
        if corrupted
        else CorruptionConfig()
    )
    config = SampleConfig(
        shape=(128, 128) if ndim == 2 else (128, 128, 128), corruptions=corruptions
    )
    return dataclasses.replace(config, **overrides)


@dataclass
class Sample:
    """One generated sample.

    Attributes
    ----------
    graph : SkeletonGraph
        The true tree, in physical coordinates.
    clean_mask : np.ndarray
        The rasterised tree.
    mask : np.ndarray
        The corrupted mask, the input for skeletonisation.
    corruptions : list of dict
        What each corruption did and where.
    config : SampleConfig
        The config that generated the sample.
    seed : int
        The seed that generated the sample.
    """

    graph: SkeletonGraph
    clean_mask: np.ndarray
    mask: np.ndarray
    corruptions: list[dict]
    config: SampleConfig
    seed: int

    @property
    def spacing(self) -> tuple[float, ...]:
        """Physical voxel size per axis."""
        if self.config.spacing is None:
            return (1.0,) * len(self.config.shape)
        return tuple(self.config.spacing)

    def save(self, path: str | Path) -> Path:
        """Save as a compressed ``.npz`` file.

        Parameters
        ----------
        path : str or Path
            Output file. The ``.npz`` suffix is added if missing.

        Returns
        -------
        Path
            The file that was written.
        """
        path = Path(path)
        # append rather than replace, so a name like "fit_0.1" keeps its "0.1"
        if path.suffix != ".npz":
            path = path.with_name(path.name + ".npz")
        graph = {f"graph_{key}": value for key, value in self.graph.to_dict().items()}
        metadata = {
            "config": self.config.to_dict(),
            "seed": self.seed,
            "corruptions": self.corruptions,
        }
        np.savez_compressed(
            path,
            clean_mask=self.clean_mask,
            mask=self.mask,
            metadata=np.array(json.dumps(metadata)),
            **graph,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Sample":
        """Load a sample saved with :meth:`save`."""
        with np.load(path) as data:
            metadata = json.loads(str(data["metadata"]))
            graph = SkeletonGraph.from_dict(
                {
                    key.removeprefix("graph_"): data[key]
                    for key in data.files
                    if key.startswith("graph_")
                }
            )
            return cls(
                graph=graph,
                clean_mask=data["clean_mask"],
                mask=data["mask"],
                corruptions=metadata["corruptions"],
                config=SampleConfig.from_dict(metadata["config"]),
                seed=metadata["seed"],
            )


def generate_sample(config: SampleConfig, seed: int) -> Sample:
    """Generate one sample.

    The tree and the corruptions use independent random streams derived from
    ``seed``, so changing the corruptions keeps the tree the same.

    Parameters
    ----------
    config : SampleConfig
        What to generate.
    seed : int
        Seed for the random streams.

    Returns
    -------
    Sample
        The generated sample.
    """
    tree_seed, corruption_seed = np.random.SeedSequence(seed).spawn(2)
    graph = generate_tree(config.tree, config.extent, np.random.default_rng(tree_seed))
    clean_mask = rasterize_graph(
        graph, config.shape, spacing=config.spacing, supersample=config.supersample
    )
    mask, records = corrupt(
        clean_mask,
        graph,
        config.corruptions,
        np.random.default_rng(corruption_seed),
        spacing=config.spacing,
    )
    return Sample(
        graph=graph,
        clean_mask=clean_mask,
        mask=mask,
        corruptions=records,
        config=config,
        seed=seed,
    )


def generate_dataset(
    config: SampleConfig, output_dir: str | Path, n_samples: int, seed: int = 0
) -> list[Path]:
    """Generate samples and save them to a directory.

    Sample ``i`` uses seed ``seed + i`` and is written to
    ``sample_{i:04d}.npz``. A ``dataset.json`` file records the config and
    seeds.

    Parameters
    ----------
    config : SampleConfig
        What to generate.
    output_dir : str or Path
        Directory to write to. It is created if needed.
    n_samples : int
        Number of samples.
    seed : int
        Seed of the first sample.

    Returns
    -------
    list of Path
        The sample files, in order.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [
        generate_sample(config, seed + index).save(output_dir / f"sample_{index:04d}")
        for index in range(n_samples)
    ]
    summary = {
        "config": config.to_dict(),
        "seeds": [seed + index for index in range(n_samples)],
        "files": [path.name for path in paths],
    }
    (output_dir / "dataset.json").write_text(json.dumps(summary, indent=2))
    return paths


def load_dataset(directory: str | Path) -> list[Sample]:
    """Load every sample listed in a directory's ``dataset.json``."""
    directory = Path(directory)
    summary = json.loads((directory / "dataset.json").read_text())
    return [Sample.load(directory / name) for name in summary["files"]]
