import re
import numpy as np
import tensorflow as tf


def tpu_init():
    print("Tensorflow version " + tf.__version__)

    # Detect TPU, return appropriate distribution strategy
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print("Running on TPU ", tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy()

    print("REPLICAS: ", strategy.num_replicas_in_sync)

    return strategy


from constants import TRAINING_FILENAMES, VALIDATION_FILENAMES, TEST_FILENAMES


def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec
    # files, i.e. flowers00-230.tfrec = 230 data items
    n = [
        int(re.compile(r"-(\d*)\.").search(filename).group(1)) for filename in filenames
    ]
    return np.sum(n)


from dataset_loader import (
    get_training_dataset,
    get_validation_dataset,
    get_test_dataset,
)
from visualization import display_batch_of_images


def main():
    strategy = tpu_init()

    NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
    NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
    NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
    print(
        f"Dataset: {NUM_TRAINING_IMAGES} training images, {NUM_VALIDATION_IMAGES} validation images, {NUM_TEST_IMAGES} unlabelled test images"
    )

    # Define the batch size. This will be 16 with TPU off and 128 (=16*8) with TPU on
    BATCH_SIZE = 16 * strategy.num_replicas_in_sync

    ds_train = get_training_dataset(BATCH_SIZE)
    ds_valid = get_validation_dataset(BATCH_SIZE)
    ds_test = get_test_dataset(BATCH_SIZE)

    print("Training:", ds_train)
    print("Validation:", ds_valid)
    print("Test:", ds_test)

    np.set_printoptions(threshold=15, linewidth=80)

    print("Training data shapes:")
    for image, label in ds_train.take(3):
        print(image.numpy().shape, label.numpy().shape)
    print("Training data label examples:", label.numpy())

    print("Test data shapes:")
    for image, id_num in ds_test.take(3):
        print(image.numpy().shape, id_num.numpy().shape)
    print("Test data IDs:", id_num.numpy().astype("U"))  # U=unicode string

    ds_iter = iter(ds_train.unbatch().batch(20))

    one_batch = next(ds_iter)
    display_batch_of_images(one_batch)


if __name__ == "__main__":
    main()
