from dataclasses import dataclass

import jax

from .module import Module


@jax.tree_util.register_pytree_node_class
@dataclass(init=False)
class Stack:
    modules: tuple[Module, ...]

    def __init__(self, *modules: Module):
        self.modules = modules

    def __call__(self, x: jax.Array) -> jax.Array:
        for module in self.modules:
            x = module(x)
        return x

    def tree_flatten(self):
        return (self.modules,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        (modules,) = children
        return cls(*modules)
