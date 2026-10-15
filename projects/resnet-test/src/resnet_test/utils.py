from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import jax
import jax.numpy as jnp
import numpy as np


class DataDict(TypedDict):
    data: np.ndarray
    labels: list[int]


@dataclass
class Batch:
    images: jax.Array
    labels: jax.Array


def unpickle(file: str | Path) -> DataDict:
    import pickle

    with open(file, "rb") as f:
        return pickle.load(f, encoding="latin1")


def normalize(data: DataDict) -> Batch:
    images = jnp.asarray(data["data"], dtype=jnp.float32)
    labels = jnp.asarray(data["labels"], dtype=jnp.int32)

    images = images.reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)
    images = images / 255.0

    return Batch(
        images=images,
        labels=labels,
    )


def cifar_batches(
    data_dir: str | Path, batch_size: int, key: jax.Array
) -> Iterator[Batch]:
    """Yield shuffled CIFAR-10 training batches from the extracted Python dataset."""
    paths = sorted(Path(data_dir).glob("data_batch_*"))
    if not paths:
        raise FileNotFoundError(f"No CIFAR-10 data_batch_* files in {data_dir}")

    for path in paths:
        batch = normalize(unpickle(path))
        order = jax.random.permutation(key, len(batch.labels))
        key, _ = jax.random.split(key)
        for start in range(0, len(order) - batch_size + 1, batch_size):
            indices = order[start : start + batch_size]
            yield Batch(batch.images[indices], batch.labels[indices])
