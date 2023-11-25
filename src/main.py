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
        f"Dataset - 512 x 512: {NUM_TRAINING_IMAGES_512} training images",
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

    print("Training - 512:", ds_train_512)
    print("Validation - 512:", ds_valid_512)
    print("Test - 512:", ds_test_512)

    np.set_printoptions(threshold=15, linewidth=80)

    print("Training - 512 data shapes:")
    for image, label in ds_train_512.take(3):
        print(image.numpy().shape, label.numpy().shape)
    print("Training - 512 data label examples:", label.numpy())

    print("Test - 512 data shapes:")
    for image, id_num in ds_test_512.take(3):
        print(image.numpy().shape, id_num.numpy().shape)
    print("Test - 512 data IDs:", id_num.numpy().astype("U"))  # U=unicode string

    ds_iter_512 = iter(ds_train_512.unbatch().batch(20))

    print("PRINTING ONE BATCH OF 512 x 512 SIZE IMAGES")
    one_batch_512 = next(ds_iter_512)
    display_batch_of_images(one_batch_512)


if __name__ == "__main__":
    main()
