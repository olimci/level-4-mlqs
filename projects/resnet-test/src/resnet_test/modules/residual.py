from dataclasses import dataclass

import jax

from .module import Module


@jax.tree_util.register_pytree_node_class
@dataclass(init=False)
class Residual:
    modules: tuple[Module, ...]
    skip: Module

    def __init__(self, *modules: Module, skip: Module = lambda x: x):
        self.modules = modules
        self.skip = skip

    def __call__(self, x: jax.Array) -> jax.Array:
        xt = x
        for module in self.modules:
            xt = module(xt)
        return xt + self.skip(x)

    def tree_flatten(self):
        return (self.modules, self.skip), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        modules, skip = children
        return cls(*modules, skip=skip)
