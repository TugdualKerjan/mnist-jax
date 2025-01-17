import datasets
import jax
import jax.random as jr
import jax.numpy as jnp
import equinox as eqx
import optax
from model import MLP


@eqx.filter_value_and_grad
def calc_loss(model, x, y):
    pred = jax.vmap(model)(x)
    return jnp.mean((pred - y) ** 2)


def make_step(model, optimizer, opt_state, x, y):
    loss, grad = calc_loss(model, x, y)
    updates, opt_state = optimizer.update(grad, opt_state, model)
    model = eqx.apply_updates(model, updates)

    return model, opt_state, loss


seed = 1
epochs = 100
batch_size = 32

key = jr.PRNGKey(seed)

data = datasets.load_dataset("ylecun/mnist").with_format("jax")

image_train_set, image_test_set = (
    jnp.reshape(data["train"]["image"] / 255, (-1, 784)),
    jnp.reshape(data["test"]["image"] / 255, (-1, 784)),
)
label_train_set, label_test_set = (
    jax.nn.one_hot(data["train"]["label"], num_classes=10, axis=-1),
    jax.nn.one_hot(data["test"]["label"], num_classes=10, axis=-1),
)

# print(type(train_set))

model = MLP(key=key)

optimizer = optax.adamw(5e-3)
opt_state = optimizer.init(model)

for i in range(epochs * image_train_set.shape[0] // batch_size):
    key, k = jr.split(key)
    permutation = jr.permutation(k, jnp.arange(0, image_train_set.shape[0]))

    x, y = (
        image_train_set[permutation],
        label_train_set[permutation],
    )

    model, opt_state, loss = make_step(model, optimizer, opt_state, x, y)

    print(loss)
