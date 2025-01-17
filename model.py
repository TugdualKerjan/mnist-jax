import jax
import jax.random as jr
import equinox as eqx
import equinox.nn as nn


class MLP(eqx.Module):
    layers: list

    def __init__(self, hidden=(784, 512, 256, 64, 16, 10), key=None):
        self.layers = [
            nn.Linear(hidden[i], hidden[i + 1], key=k)
            for i, k in enumerate(jr.split(key, len(hidden) - 1))
        ]

    @jax.jit
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            x = jax.nn.sigmoid(x)
        return x
