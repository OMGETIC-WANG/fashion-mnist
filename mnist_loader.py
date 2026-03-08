import os
import kagglehub
import pandas as pd
import jax
import jax.numpy as jnp
import numpy as np


def LoadMnist() -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    dataset_path = "./cache/datasets"
    trainset_path = os.path.join(dataset_path, "mnist_train.csv")
    testset_path = os.path.join(dataset_path, "mnist_test.csv")
    if not os.path.exists(trainset_path) or not os.path.exists(testset_path):
        total_try = 4
        for i in range(total_try):
            try:
                kagglehub.dataset_download("oddrationale/mnist-in-csv", output_dir=dataset_path)
                break
            except:
                print(f"Failed to download dataset. Retrying... ({i + 1}/{total_try})")

    train_df = pd.read_csv(trainset_path)
    y_train_raw = train_df.iloc[:, 0].values
    x_train_raw = train_df.iloc[:, 1:].values

    test_df = pd.read_csv(testset_path)
    y_test_raw = test_df.iloc[:, 0].values
    x_test_raw = test_df.iloc[:, 1:].values

    x_train = jnp.array(x_train_raw).astype(jnp.float32) / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    y_train = jnp.array(y_train_raw).astype(jnp.int32)

    x_test = jnp.array(x_test_raw).astype(jnp.float32) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1)
    y_test = jnp.array(y_test_raw).astype(jnp.int32)

    return (x_train, y_train), (x_test, y_test)


def _LoadFashionMnistFromKaggle(
    path: str, dataset_path: str, cache_path: str
) -> tuple[jax.Array, jax.Array]:
    filepath = os.path.join(dataset_path, path)
    if not os.path.exists(filepath):
        success = False
        for i in range(4):
            try:
                kagglehub.dataset_download(
                    "zalando-research/fashionmnist",
                    path=path,
                    output_dir=dataset_path,
                )
                success = True
                break
            except Exception as e:
                print(f"Retry {i + 1}/4: {e}")
        if not success:
            raise FileNotFoundError(f"Failed to download {path} after multiple attempts.")

    df = pd.read_csv(filepath)
    x_raw = df.iloc[:, 1:].values.astype(np.uint8)
    y_raw = df.iloc[:, 0].values.astype(np.uint8)

    np.savez(cache_path, x=np.array(x_raw), y=np.array(y_raw))

    x = jnp.array(x_raw, dtype=jnp.float32) / 255.0
    y = jnp.array(y_raw, dtype=jnp.int32)

    return x.reshape(-1, 28, 28, 1), y


def _LoadFashionMnistSerialized(
    filepath: str, dataset_path: str, kaggle_path: str
) -> tuple[jax.Array, jax.Array]:
    train_serialized_path = os.path.join(dataset_path, filepath)
    if os.path.exists(train_serialized_path):
        with np.load(train_serialized_path) as data:
            x_train = jnp.array(data["x"]).reshape(-1, 28, 28, 1) / 255.0
            y_train = jnp.array(data["y"])
    else:
        x_train, y_train = _LoadFashionMnistFromKaggle(
            kaggle_path, dataset_path, train_serialized_path
        )
    return x_train, y_train


def LoadFashionMnist(
    dataset_path: str = "./cache/datasets/fashion",
) -> tuple[tuple[jax.Array, jax.Array], tuple[jax.Array, jax.Array]]:
    x_train, y_train = _LoadFashionMnistSerialized(
        "train.npz", dataset_path, "fashion-mnist_train.csv"
    )
    x_test, y_test = _LoadFashionMnistSerialized("test.npz", dataset_path, "fashion-mnist_test.csv")
    return (x_train, y_train), (x_test, y_test)
