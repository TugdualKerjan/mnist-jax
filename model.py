import jax
import jax.random as jr
import equinox as eqx
import equinox.nn as nn


class MLP(eqx.Module):
    layers: list

# @discussion{
#   "comments": [
#     {
#       "author": "anonymous",
#       "timestamp": "2025-01-17T23:35:44.774088",
#       "content": "Regarding:\n> hidden=(784, 512, 256, 64, 16, 10)\n\nIs there any particular reason for these layers being chosen ?",
#       "paper_ref": null
#     },
#     {
#       "author": "anonymous",
#       "timestamp": "2025-01-17T23:37:49.364130",
#       "content": "It's thanks to this : ",
#       "paper_ref": {
#         "arxiv_id": "2205.15439",
#         "section": "2"
#       }
#     }
#   ]
# }
# @discussion{
#   "comments": [
#     {
#       "author": "anonymous",
#       "timestamp": "2025-01-17T23:35:44.774088",
#       "content": "Regarding:\n> hidden=(784, 512, 256, 64, 16, 10)\n\nIs there any particular reason for these layers being chosen ?",
#       "paper_ref": null
#     }
#   ]
# }
    def __init__(self, hidden=(784, 512, 256, 64, 16, 10), key=None):
        self.layers = [
# @discussion{
#   "comments": [
#     {
#       "author": "anonymous",
#       "timestamp": "2025-01-17T23:34:30.979157",
#       "content": "Regarding:\n>  hidden[i + 1], key=k)\n\n",
#       "paper_ref": null
#     }
#   ]
# }
            nn.Linear(hidden[i], hidden[i + 1], key=k)
            for i, k in enumerate(jr.split(key, len(hidden) - 1))
        ]

    @jax.jit
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            x = jax.nn.sigmoid(x)
        return x
