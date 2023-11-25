import numpy as np
import tensorflow as tf


def get_strategy():
    # Detect TPU, return appropriate distribution strategy
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print("Running on TPU ", tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy()

    print("REPLICAS: ", strategy.num_replicas_in_sync)

    return strategy


from dataset_loader import (
    get_training_dataset,
    get_validation_dataset,
    get_test_dataset,
    count_data_items,
)
from visualization import display_batch_of_images


def visualize_data(ds_train, ds_valid, ds_test):
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

    print("PRINTING ONE BATCH OF IMAGES")
    one_batch = next(ds_iter)
    display_batch_of_images(one_batch)


def main():
    print("Tensorflow version " + tf.__version__)

    strategy = get_strategy()

    BATCH_SIZE = 16 * strategy.num_replicas_in_sync
    IMAGE_SIZE_512 = [512, 512]
    DS_PATH = "data"
    PATH_512 = DS_PATH + "/tfrecords-jpeg-512x512"

    TRAINING_FILENAMES_512 = tf.io.gfile.glob(PATH_512 + "/train/*.tfrec")
    VALIDATION_FILENAMES_512 = tf.io.gfile.glob(PATH_512 + "/val/*.tfrec")
    TEST_FILENAMES_512 = tf.io.gfile.glob(PATH_512 + "/test/*.tfrec")  # 100 - 10

    NUM_TRAINING_IMAGES_512 = count_data_items(TRAINING_FILENAMES_512)
    NUM_VALIDATION_IMAGES_512 = count_data_items(VALIDATION_FILENAMES_512)
    NUM_TEST_IMAGES_512 = count_data_items(TEST_FILENAMES_512)

    print(
        "Dataset - 512 x 512: ",
        f"{NUM_TRAINING_IMAGES_512} training images",
        f"{NUM_VALIDATION_IMAGES_512} validation images",
        f"{NUM_TEST_IMAGES_512} unlabelled test images",
    )

    ds_train_512 = get_training_dataset(
        TRAINING_FILENAMES_512,
        IMAGE_SIZE_512,
        BATCH_SIZE,
    )
    ds_valid_512 = get_validation_dataset(
        VALIDATION_FILENAMES_512,
        IMAGE_SIZE_512,
        BATCH_SIZE,
    )
    ds_test_512 = get_test_dataset(
        TEST_FILENAMES_512,
        IMAGE_SIZE_512,
        BATCH_SIZE,
    )

    visualize_data(ds_train_512, ds_valid_512, ds_test_512)


if __name__ == "__main__":
    main()
