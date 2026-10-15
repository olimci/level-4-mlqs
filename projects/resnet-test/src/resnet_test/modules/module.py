from typing import Protocol

import jax


class Module(Protocol):
    def __call__(self, x: jax.Array, /) -> jax.Array: ...
