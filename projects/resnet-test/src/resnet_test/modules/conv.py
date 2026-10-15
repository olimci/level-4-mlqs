from dataclasses import dataclass
from typing import Self

import jax
import jax.numpy as jnp


@jax.tree_util.register_dataclass
@dataclass
class Conv2d:
    kernel: jax.Array
    stride: int
    padding: str

    def __init__(self, kernel: jax.Array, stride: int, padding: str):
        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def __call__(self, x: jax.Array) -> jax.Array:
        return jax.lax.conv_general_dilated(
            x,
            self.kernel,
            window_strides=(self.stride, self.stride),
            padding=self.padding,
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )

    @classmethod
    def init_conv(
        cls,
        key: jax.Array,
        kernel_size: tuple[int, int],
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: str = "SAME",
    ) -> Self:
        fan_in = kernel_size[0] * kernel_size[1] * in_channels
        kernel = jax.random.normal(
            key, (*kernel_size, in_channels, out_channels)
        ) / jnp.sqrt(fan_in)
        return cls(kernel, stride, padding)
