import re
import tensorflow as tf
import numpy as np

AUTO = tf.data.experimental.AUTOTUNE


def decode_image(image_data, image_size):
    image = tf.image.decode_jpeg(image_data, channels=3)
    # convert image to floats in [0, 1] range
    image = tf.cast(image, tf.float32) / 255.0
    # explicit size needed for TPU
    image = tf.reshape(image, [*image_size, 3])
    return image


def read_labelled_tfrecord(example, image_size):
    LABELLED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELLED_TFREC_FORMAT)
    image = decode_image(example["image"], image_size)
    label = tf.cast(example["class"], tf.int32)
    return image, label  # returns a dataset of (image, label) pairs


def read_unlabelled_tfrecord(example, image_size):
    UNLABELLED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        # class is missing, this competitions's challenge is to predict flower classes for the test dataset
    }
    example = tf.io.parse_single_example(example, UNLABELLED_TFREC_FORMAT)
    image = decode_image(example["image"], image_size)
    id_num = example["id"]
    return image, id_num  # returns a dataset of image(s)


def load_dataset(filenames, image_size, labelled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False  # disable order, increase speed

    # automatically interleaves reads from multiple files
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)

    # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(
        lambda x: read_labelled_tfrecord(x, image_size)
        if labelled
        else read_unlabelled_tfrecord(x, image_size),
        num_parallel_calls=AUTO,
    )
    # returns a dataset of (image, label) pairs if labelled=True or (image, id) pairs if labelled=False
    return dataset


def data_augment(image, label):
    # Thanks to the dataset.prefetch(AUTO)
    # statement in the next function (below), this happens essentially
    # for free on TPU. Data pipeline code is executed on the "CPU"
    # part of the TPU while the TPU itself is computing gradients.
    image = tf.image.random_flip_left_right(image)
    # image = tf.image.random_saturation(image, 0, 2)
    return image, label


def get_training_dataset(train_file_name, image_size, batch_size):
    dataset = load_dataset(train_file_name, image_size, labelled=True)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat()  # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(batch_size)
    # prefetch next batch while training (autotune prefetch buffer size)
    dataset = dataset.prefetch(AUTO)
    return dataset


def get_validation_dataset(validation_file_name, image_size, batch_size, ordered=False):
    dataset = load_dataset(
        validation_file_name, image_size, labelled=True, ordered=ordered
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO)
    return dataset


def get_test_dataset(test_file_name, image_size, batch_size, ordered=False):
    dataset = load_dataset(test_file_name, image_size, labelled=False, ordered=ordered)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTO)
    return dataset


def count_data_items(filenames):
    # the number of data items is written in the name of the .tfrec
    # files, i.e. flowers00-230.tfrec = 230 data items
    n = [
        int(re.compile(r"-([0-9]*)\.").search(filename).group(1))
        for filename in filenames
    ]
    return np.sum(n)
