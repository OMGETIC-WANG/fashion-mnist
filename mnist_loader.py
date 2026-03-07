import os
import kagglehub
import pandas as pd
import jax.numpy as jnp


def LoadMnist():
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


def LoadFashionMnist():
    dataset_path = "./cache/datasets/fashion"
    trainset_path = os.path.join(dataset_path, "fashion-mnist_train.csv")
    testset_path = os.path.join(dataset_path, "fashion-mnist_test.csv")

    if not os.path.exists(trainset_path):
        for i in range(4):
            try:
                kagglehub.dataset_download(
                    "zalando-research/fashionmnist",
                    path="fashion-mnist_train.csv",
                    output_dir=dataset_path,
                )
                kagglehub.dataset_download(
                    "zalando-research/fashionmnist",
                    path="fashion-mnist_test.csv",
                    output_dir=dataset_path,
                )
                break
            except Exception as e:
                print(f"Retry {i + 1}/4: {e}")

    def process_csv(path):
        df = pd.read_csv(path)
        y = jnp.array(df.iloc[:, 0].values, dtype=jnp.int32)
        x = jnp.array(df.iloc[:, 1:].values, dtype=jnp.float32) / 255.0
        return x.reshape(-1, 28, 28, 1), y

    return process_csv(trainset_path), process_csv(testset_path)
