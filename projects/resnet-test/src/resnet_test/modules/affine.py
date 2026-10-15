from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass
class Affine:
    w: jax.Array
    b: jax.Array

    def __init__(self, w: jax.Array, b: jax.Array):
        self.w = w
        self.b = b

    def __call__(self, x: jax.Array) -> jax.Array:
        return x @ self.w + self.b

    @classmethod
    def init_linear(cls, key: jax.Array, in_dim: int, out_dim: int) -> Self:
        w = jax.random.normal(key, (in_dim, out_dim)) / jnp.sqrt(in_dim)
        b = jnp.zeros((out_dim,))
        return cls(w, b)
