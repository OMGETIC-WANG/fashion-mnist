from ml_collections import ConfigDict


def get_config():
    config = ConfigDict()

    config.seed = 666

    config.test_only = False
    config.use_training_model = False
    config.train_state_path = "./cache/latest.trainstate"

    config.model_features = 64
    config.num_heads = 2
    config.num_encoders = 1

    config.train_batch_size = 32
    config.epoch_count = 100

    config.test_batch_size = 100

    config.learning_rate = 0.0001

    config.state_save_per_epoch = 10
    config.model_save_dir = "./cache"
    config.model_suffix = ".model"

    config.use_graphic = True

    return config
