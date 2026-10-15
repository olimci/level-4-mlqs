from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass
class BatchNorm2d:
    gamma: jax.Array
    beta: jax.Array
    running_mean: jax.Array
    running_var: jax.Array

    def __call__(self, x: jax.Array) -> jax.Array:
        x_norm = (x - self.running_mean) / jnp.sqrt(self.running_var + 1e-5)
        return self.gamma * x_norm + self.beta

    def train(
        self, x: jax.Array, state: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        mean = jnp.mean(x, axis=(0, 1, 2))
        var = jnp.var(x, axis=(0, 1, 2))
        x_norm = (x - mean) / jnp.sqrt(var + 1e-5)
        running_mean, running_var = state
        new_state = (
            0.9 * running_mean + 0.1 * mean,
            0.9 * running_var + 0.1 * var,
        )
        return self.gamma * x_norm + self.beta, new_state

    @classmethod
    def init_batch_norm(cls, channels: int) -> Self:
        return cls(
            jnp.ones((channels,)),
            jnp.zeros((channels,)),
            jnp.zeros((channels,)),
            jnp.ones((channels,)),
        )
