from model import MnistModel

import mnist_loader

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import typing as T

import lossplot
import ascii_util


@nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0))
@nnx.jit
def TrainBatch(
    model_optimizer: tuple[MnistModel, nnx.Optimizer[MnistModel]],
    x: jax.Array,
    y: jax.Array,
):
    model, optimzier = model_optimizer

    def loss_fn(model: MnistModel):
        logits = model(x)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y))
        return loss

    loss, grads = nnx.value_and_grad(loss_fn)(model)
    optimzier.update(model, grads)
    return (model, optimzier), loss


@nnx.jit
def TrainModel(
    model: MnistModel,
    optimizer: nnx.Optimizer[MnistModel],
    x: jax.Array,
    y: jax.Array,
    metrics: nnx.Metric,
):
    _, losses = TrainBatch((model, optimizer), x, y)
    metrics.update(values=losses)


def Train(
    model: MnistModel,
    optimizer: nnx.Optimizer[MnistModel],
    x: jax.Array,
    y: jax.Array,
    epoch_count: int,
):
    train_metrics = nnx.MultiMetric(loss=nnx.metrics.Average())

    progress_bar = ascii_util.ProgressBar("Training", epoch_count, show_percent=False)
    loss_plot = lossplot.LossPlot("Training Loss")

    for epoch in range(epoch_count):
        TrainModel(model, optimizer, x, y, train_metrics)
        epoch_metrics = train_metrics.compute()
        train_metrics.reset()

        progress_bar.Update(epoch + 1, f"loss: {epoch_metrics['loss']}")
        loss_plot.Update({"loss": epoch_metrics["loss"]})
    progress_bar.End()


@nnx.scan(in_axes=(None, 0, 0), out_axes=0)
@nnx.jit
def TestBatch(model: MnistModel, x: jax.Array, y: jax.Array):
    logits = model(x)
    return jnp.sum(jnp.argmax(logits, axis=-1) == y)


@nnx.jit
def TestModel(model: MnistModel, x: jax.Array, y: jax.Array):
    testset_size = x.shape[0] * x.shape[1]
    res = TestBatch(model, x, y)
    return res.sum() / testset_size


def BatchDatas(xs: T.Sequence[jax.Array], batch_size: int):
    assert xs[0].shape[0] % batch_size == 0
    for i in range(1, len(xs)):
        assert xs[i].shape[0] == xs[0].shape[0]
    return [x.reshape(x.shape[0] // batch_size, batch_size, *x.shape[1:]) for x in xs]


(x_train, y_train), (x_test, y_test) = mnist_loader.LoadFashionMnist()

rngs = nnx.Rngs(666)
model = MnistModel(64, 1, 1, 32, rngs)
optimizer = nnx.Optimizer(model, optax.adam(0.0001), wrt=nnx.Param)

EPOCH_COUNT = 100
TRAIN_BATCH_SIZE = 20
TEST_BATCH_SIZE = TRAIN_BATCH_SIZE
Train(
    model,
    optimizer,
    *BatchDatas([x_train, y_train], TRAIN_BATCH_SIZE),
    epoch_count=EPOCH_COUNT,
)
print(
    f"Test accuracy: {TestModel(model, *BatchDatas([x_test, y_test], TEST_BATCH_SIZE)) * 100:.4f}%"
)
