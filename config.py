from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    config.seed = 666

    config.model_features = 64
    config.num_encoders = 1

    config.train_batch_size = 32
    config.epoch_count = 100

    config.test_batch_size = 100

    config.learning_rate = 0.0001

    return config
