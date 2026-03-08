from model import MnistModel

import mnist_loader

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import typing as T

import lossplot
import ascii_util

import os
import time
import model_serialization

from ml_collections import config_flags
from absl import app


def Preprocess(
    x: jax.Array,
    random_key: jax.Array,
):
    random_key, subkey = jax.random.split(random_key)
    horizen_flip_mask = jax.random.bernoulli(subkey, 0.5, (x.shape[0], 1, 1, 1))
    x = jnp.where(horizen_flip_mask, x[:, :, ::-1, :], x)

    random_key, subkey = jax.random.split(random_key)
    shift = jax.random.randint(subkey, (2,), -2, 2)
    x = jnp.roll(x, shift, axis=(1, 2))

    return x


Model_t = T.TypeVar("Model_t", bound=nnx.Module)


@nnx.scan(in_axes=(nnx.Carry, 0, 0, 0), out_axes=(nnx.Carry, 0, 0))
@nnx.jit
def TrainBatch(
    model_optimizer: tuple[Model_t, nnx.Optimizer[Model_t]],
    x: jax.Array,
    y: jax.Array,
    random_key: jax.Array,
):
    model, optimzier = model_optimizer

    x = Preprocess(x, random_key)

    def loss_fn(model: Model_t):
        logits = model(x)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
        accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
        return loss, accuracy

    (loss, accuracy), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimzier.update(model, grads)
    return (model, optimzier), loss, accuracy


@nnx.jit(static_argnames=["batch_size"])
def TrainModel(
    model: Model_t,
    optimizer: nnx.Optimizer[Model_t],
    x: jax.Array,
    y: jax.Array,
    batch_size: int,
    rngs: nnx.Rngs,
    metrics: nnx.Metric,
):
    indices = jnp.arange(x.shape[0])
    indices = jax.random.permutation(rngs.params(), indices)
    x, y = BatchDatas((x[indices], y[indices]), batch_size)

    random_keys = jax.random.split(rngs.params(), x.shape[0])

    _, losses, accuracies = TrainBatch((model, optimizer), x, y, random_keys)

    metrics.update(values=losses, accuracy=accuracies)


def Train(
    model: Model_t,
    optimizer: nnx.Optimizer[Model_t],
    x: jax.Array,
    y: jax.Array,
    batch_size: int,
    rngs: nnx.Rngs,
    epoch_count: int,
    x_test: T.Optional[jax.Array] = None,
    y_test: T.Optional[jax.Array] = None,
    test_batch_size: T.Optional[int] = None,
    state_save_path: T.Optional[str] = None,
    state_save_per_epoch: T.Optional[int] = None,
    model_save_path: T.Optional[str] = None,
    use_graphic: bool = True,
    dashboard_block: bool = False,
):
    train_metrics = nnx.MultiMetric(
        loss=nnx.metrics.Average(), accuracy=nnx.metrics.Average("accuracy")
    )

    if test_batch_size is None:
        test_batch_size = batch_size

    if state_save_per_epoch is not None and state_save_path is not None:
        assert state_save_per_epoch > 0

    progress_bar = ascii_util.ProgressBar("Training", epoch_count, show_percent=False)
    if use_graphic:
        dashboard = lossplot.Dashboard(
            "Dashboard", {"Loss": ["loss"], "Accuracy": ["accuracy", "test_accuracy"]}
        )
    else:
        dashboard = None

    for epoch in range(epoch_count):
        TrainModel(model, optimizer, x, y, batch_size, rngs, train_metrics)
        epoch_metrics = train_metrics.compute()
        train_metrics.reset()

        if state_save_path is not None and state_save_per_epoch is not None:
            if (epoch + 1) % state_save_per_epoch == 0:
                model_serialization.SaveTrainingState(
                    os.path.join(state_save_path),
                    model,
                    optimizer,
                )

        progress_bar_msg = f"loss: {epoch_metrics['loss']}, accuracy: {epoch_metrics['accuracy']}"
        loss_plot_dict = {"loss": epoch_metrics["loss"], "accuracy": epoch_metrics["accuracy"]}
        if x_test is not None and y_test is not None:
            model.eval()
            test_accuracy = TestModel(model, x_test, y_test, test_batch_size)
            model.train()
            progress_bar_msg += f", test_accuracy: {test_accuracy}"
            loss_plot_dict["test_accuracy"] = test_accuracy
        progress_bar.Update(epoch + 1, progress_bar_msg)
        if dashboard is not None:
            dashboard.Update(loss_plot_dict)

    if model_save_path is not None:
        model_serialization.SaveModel(model_save_path, model)

    progress_bar.End()

    if dashboard is not None and dashboard_block:
        dashboard.fig.show()


@nnx.scan(in_axes=(None, 0, 0), out_axes=0)
@nnx.jit
def TestBatch(model: nnx.Module, x: jax.Array, y: jax.Array):
    logits = model(x)
    return jnp.sum(jnp.argmax(logits, axis=-1) == y)


@nnx.jit(static_argnames=["batch_size"])
def TestModel(model: nnx.Module, x: jax.Array, y: jax.Array, batch_size: int):
    testset_size = x.shape[0]
    x, y = BatchDatas((x, y), batch_size)
    res = TestBatch(model, x, y)
    return res.sum() / testset_size


def BatchDatas(xs: T.Sequence[jax.Array], batch_size: int):
    return [x.reshape(x.shape[0] // batch_size, batch_size, *x.shape[1:]) for x in xs]


def CountModuleParams(module: nnx.Module):
    params = nnx.state(module, nnx.Param)
    leaves = jax.tree_util.tree_leaves(params)
    return sum(leaf.size for leaf in leaves)


_CONFIG = config_flags.DEFINE_config_file(
    "config", "config.py", "Configuration file for training the model."
)


def main(_):
    config = _CONFIG.value

    print("Initing model")
    rngs = nnx.Rngs(config.seed)

    print("Loading data")
    (x_train, y_train), (x_test, y_test) = mnist_loader.LoadFashionMnist()
    trainset_size = x_train.shape[0]
    testset_size = x_test.shape[0]

    if not config.test_only:
        if config.use_training_model:
            model, optimizer = model_serialization.LoadTrainingState(
                config.train_state_path,
                lambda: MnistModel(
                    config.model_features, config.num_heads, config.num_encoders, nnx.Rngs(0)
                ),
                lambda model: nnx.Optimizer(
                    model, optax.adamw(config.learning_rate), wrt=nnx.Param
                ),
            )
        else:
            model = MnistModel(config.model_features, config.num_heads, config.num_encoders, rngs)
            total_steps = config.epoch_count * (trainset_size // config.train_batch_size)
            optimizer_schedule = optax.warmup_cosine_decay_schedule(
                0.0,
                config.learning_rate,
                decay_steps=total_steps,
                warmup_steps=total_steps // 10,
                end_value=1e-6,
            )
            optimizer = nnx.Optimizer(
                model, optax.adamw(optimizer_schedule, weight_decay=1e-2), wrt=nnx.Param
            )
        print(f"Model param count: {CountModuleParams(model)}")
        print("Starting training")
        Train(
            model,
            optimizer,
            x_train,
            y_train,
            config.train_batch_size,
            rngs,
            epoch_count=config.epoch_count,
            x_test=x_test,
            y_test=y_test,
            test_batch_size=config.test_batch_size,
            state_save_path=config.train_state_path,
            state_save_per_epoch=config.state_save_per_epoch,
            model_save_path=os.path.join(
                config.model_save_dir, f"{time.time()}.{config.model_suffix}"
            ),
        )
    else:
        model = model_serialization.LoadNewestModel(
            config.model_save_dir,
            config.model_suffix,
            lambda: MnistModel(config.model_features, config.num_heads, config.num_encoders, rngs),
        )

    print("Start testing")
    model.eval()
    print(f"Test accuracy: {TestModel(model, x_test, y_test, config.test_batch_size) * 100:.4f}%")


if __name__ == "__main__":
    app.run(main)
