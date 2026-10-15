import argparse
from pathlib import Path

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import optax
from jax import Array

from .modules.affine import Affine
from .modules.conv import Conv2d
from .modules.module import Module
from .modules.norm import BatchNorm2d
from .modules.pooling import GlobalAvgPool2d
from .modules.residual import Residual
from .modules.stack import Stack
from .utils import cifar_batches, normalize, unpickle


def new_model(key: Array) -> Module:
    keys = iter(jax.random.split(key, 22))
    layers: list[Module] = [
        Conv2d.init_conv(next(keys), (3, 3), 3, 16),
        BatchNorm2d.init_batch_norm(16),
        jnn.relu,
    ]

    in_channels = 16
    for channels in (16, 32, 64):
        for block in range(3):
            stride = 2 if channels != 16 and block == 0 else 1
            skip = (
                Stack(
                    Conv2d.init_conv(
                        next(keys), (1, 1), in_channels, channels, stride=stride
                    ),
                    BatchNorm2d.init_batch_norm(channels),
                )
                if stride == 2
                else lambda x: x
            )
            layers.extend(
                (
                    Residual(
                        Conv2d.init_conv(
                            next(keys), (3, 3), in_channels, channels, stride=stride
                        ),
                        BatchNorm2d.init_batch_norm(channels),
                        jnn.relu,
                        Conv2d.init_conv(next(keys), (3, 3), channels, channels),
                        BatchNorm2d.init_batch_norm(channels),
                        skip=skip,
                    ),
                    jnn.relu,
                )
            )
            in_channels = channels

    layers.extend((GlobalAvgPool2d(), Affine.init_linear(next(keys), 64, 10)))
    return Stack(*layers)


def init_norm_state(module: Module):
    if isinstance(module, BatchNorm2d):
        return module.running_mean, module.running_var
    if isinstance(module, Stack):
        return tuple(init_norm_state(child) for child in module.modules)
    if isinstance(module, Residual):
        return (
            tuple(init_norm_state(child) for child in module.modules),
            init_norm_state(module.skip),
        )
    return None


def train_model(module: Module, x: Array, norm_state):
    if isinstance(module, BatchNorm2d):
        return module.train(x, norm_state)
    if isinstance(module, Stack):
        new_states = []
        for child, child_state in zip(module.modules, norm_state, strict=True):
            x, child_state = train_model(child, x, child_state)
            new_states.append(child_state)
        return x, tuple(new_states)
    if isinstance(module, Residual):
        body_states, skip_state = norm_state
        residual, body_states = train_model(Stack(*module.modules), x, body_states)
        skip, skip_state = train_model(module.skip, x, skip_state)
        return residual + skip, (body_states, skip_state)
    return module(x), norm_state


def with_norm_state(module: Module, norm_state) -> Module:
    if isinstance(module, BatchNorm2d):
        mean, var = norm_state
        return BatchNorm2d(module.gamma, module.beta, mean, var)
    if isinstance(module, Stack):
        return Stack(
            *(
                with_norm_state(child, state)
                for child, state in zip(module.modules, norm_state, strict=True)
            )
        )
    if isinstance(module, Residual):
        body_states, skip_state = norm_state
        return Residual(
            *(
                with_norm_state(child, state)
                for child, state in zip(module.modules, body_states, strict=True)
            ),
            skip=with_norm_state(module.skip, skip_state),
        )
    return module


def train(
    data_dir: str | Path,
    *,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.01,
    output_path: str | Path = "resnet20.npz",
) -> Module:
    """Train the CIFAR-10 ResNet and return its learned parameters."""
    key = jax.random.key(0)
    key, model_key = jax.random.split(key)
    model = jax.tree.map(
        lambda leaf: jax.tree_util.Partial(leaf) if callable(leaf) else leaf,
        new_model(model_key),
    )
    params = jax.tree.map(
        lambda leaf: leaf if isinstance(leaf, jax.Array) else None, model
    )
    norm_state = init_norm_state(model)
    steps_per_epoch = 5 * (10_000 // batch_size)
    schedule = optax.cosine_decay_schedule(
        init_value=learning_rate,
        decay_steps=epochs * steps_per_epoch,
        alpha=0.01,
    )
    optimizer = optax.adamw(schedule, weight_decay=weight_decay)
    state = optimizer.init(params)
    test_data = normalize(unpickle(Path(data_dir) / "test_batch"))

    @jax.jit
    def step(params, state, norm_state, images, labels, augmentation_key):
        crop_key, flip_key, noise_key = jax.random.split(augmentation_key, 3)
        padded = jnp.pad(images, ((0, 0), (4, 4), (4, 4), (0, 0)))
        offsets = jax.random.randint(crop_key, (len(images), 2), 0, 9)
        images = jax.vmap(
            lambda image, offset: jax.lax.dynamic_slice(
                image, (offset[0], offset[1], 0), (32, 32, 3)
            )
        )(padded, offsets)
        flipped = jnp.flip(images, axis=2)
        flip = jax.random.bernoulli(flip_key, shape=(len(images), 1, 1, 1))
        images = jnp.where(flip, flipped, images)
        noise = 0.02 * jax.random.normal(noise_key, images.shape)
        images = jnp.clip(images + noise, 0.0, 1.0)

        def objective(current_params):
            current_model = jax.tree.map(
                lambda param, leaf: leaf if param is None else param,
                current_params,
                model,
                is_leaf=lambda leaf: leaf is None,
            )
            logits, updated_norm_state = train_model(current_model, images, norm_state)
            loss = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits, labels)
            )
            return loss, updated_norm_state

        (batch_loss, norm_state), gradients = jax.value_and_grad(
            objective, has_aux=True
        )(params)
        updates, state = optimizer.update(gradients, state, params)
        return optax.apply_updates(params, updates), state, norm_state, batch_loss

    @jax.jit
    def test_step(params, norm_state, images, labels):
        current_model = jax.tree.map(
            lambda param, leaf: leaf if param is None else param,
            params,
            model,
            is_leaf=lambda leaf: leaf is None,
        )
        logits = with_norm_state(current_model, norm_state)(images)
        loss = jnp.sum(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
        correct = jnp.sum(jnp.argmax(logits, axis=-1) == labels)
        return loss, correct

    print("started")

    for epoch in range(epochs):
        key, epoch_key = jax.random.split(key)
        losses = []
        for batch_index, batch in enumerate(
            cifar_batches(data_dir, batch_size, epoch_key), start=1
        ):
            key, augmentation_key = jax.random.split(key)
            params, state, norm_state, batch_loss = step(
                params, state, norm_state, batch.images, batch.labels, augmentation_key
            )
            losses.append(batch_loss)
            if batch_index % 100 == 0:
                print(
                    f"epoch {epoch + 1}/{epochs} batch {batch_index} "
                    f"loss {jnp.mean(jnp.stack(losses[-100:])):.4f}"
                )
        test_losses = []
        test_correct = []
        for start in range(0, len(test_data.labels), batch_size):
            end = start + batch_size
            loss, correct = test_step(
                params,
                norm_state,
                test_data.images[start:end],
                test_data.labels[start:end],
            )
            test_losses.append(loss)
            test_correct.append(correct)
        test_loss = jnp.sum(jnp.stack(test_losses)) / len(test_data.labels)
        test_accuracy = jnp.sum(jnp.stack(test_correct)) / len(test_data.labels)
        print(
            f"epoch {epoch + 1}/{epochs} "
            f"train loss {jnp.mean(jnp.stack(losses)):.4f} "
            f"test loss {test_loss:.4f} test accuracy {test_accuracy:.2%}"
        )

    model = jax.tree.map(
        lambda param, leaf: leaf if param is None else param,
        params,
        model,
        is_leaf=lambda leaf: leaf is None,
    )
    model = with_norm_state(model, norm_state)
    with Path(output_path).open("wb") as checkpoint:
        weights = [
            leaf for leaf in jax.tree.leaves(model) if isinstance(leaf, jax.Array)
        ]
        np.savez(checkpoint, *weights)
    print(f"saved model to {output_path}")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CIFAR-10 ResNet")
    parser.add_argument(
        "data_dir", type=Path, help="extracted CIFAR-10/cifar-10-batches-py directory"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=Path("resnet20.npz"))
    args = parser.parse_args()
    train(
        args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        output_path=args.output,
    )
