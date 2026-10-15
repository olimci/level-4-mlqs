from dataclasses import dataclass

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass
class MaxPool2d:
    kernel_size: tuple[int, int]
    stride: int
    padding: str

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.lax.reduce_window(
            x,
            -jnp.inf,
            jax.lax.max,
            window_dimensions=(1, *self.kernel_size, 1),
            window_strides=(1, self.stride, self.stride, 1),
            padding=self.padding,
        )


@jax.tree_util.register_dataclass
@dataclass
class GlobalAvgPool2d:
    def __call__(self, x: jax.Array) -> jax.Array:
        return jnp.mean(x, axis=(1, 2))
