import math, re, os
import numpy as np
import tensorflow as tf

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

GCS_DS_PATH = "gs://kds-15a35095dc89b4ea0cfbc0b805269c9ef228f63e9a0bd4432b2dedc9"

IMAGE_SIZE = [512, 512]
GCS_PATH = GCS_DS_PATH + "/tfrecords-jpeg-512x512"
AUTO = tf.data.experimental.AUTOTUNE

TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/train/*.tfrec")
VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/val/*.tfrec")
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + "/test/*.tfrec")

CLASSES = [
    "pink primrose",  # 0
    "hard-leaved pocket orchid",  # 1
    "canterbury bells",  # 2
    "sweet pea",  # 3
    "wild geranium",  # 4
    "tiger lily",  # 5
    "moon orchid",  # 6
    "bird of paradise",  # 7
    "monkshood",  # 8
    "globe thistle",  # 9
    "snapdragon",  # 10
    "colt's foot",  # 11
    "king protea",  # 12
    "spear thistle",  # 13
    "yellow iris",  # 14
    "globe-flower",  # 15
    "purple coneflower",  # 16
    "peruvian lily",  # 17
    "balloon flower",  # 18
    "giant white arum lily",  # 19
    "fire lily",  # 20
    "pincushion flower",  # 21
    "fritillary",  # 22
    "red ginger",  # 23
    "grape hyacinth",  # 24
    "corn poppy",  # 25
    "prince of wales feathers",  # 26
    "stemless gentian",  # 27
    "artichoke",  # 28
    "sweet william",  # 29
    "carnation",  # 30
    "garden phlox",  # 31
    "love in the mist",  # 32
    "cosmos",  # 33
    "alpine sea holly",  # 34
    "ruby-lipped cattleya",  # 35
    "cape flower",  # 36
    "great masterwort",  # 37
    "siam tulip",  # 38
    "lenten rose",  # 39
    "barberton daisy",  # 40
    "daffodil",  # 41
    "sword lily",  # 42
    "poinsettia",  # 43
    "bolero deep blue",  # 44
    "wallflower",  # 45
    "marigold",  # 46
    "buttercup",  # 47
    "daisy",  # 48
    "common dandelion",  # 49
    "petunia",  # 50
    "wild pansy",  # 51
    "primula",  # 52
    "sunflower",  # 53
    "lilac hibiscus",  # 54
    "bishop of llandaff",  # 55
    "gaura",  # 56
    "geranium",  # 57
    "orange dahlia",  # 58
    "pink-yellow dahlia",  # 59
    "cautleya spicata",  # 60
    "japanese anemone",  # 61
    "black-eyed susan",  # 62
    "silverbush",  # 63
    "californian poppy",  # 64
    "osteospermum",  # 65
    "spring crocus",  # 66
    "iris",  # 67
    "windflower",  # 68
    "tree poppy",  # 69
    "gazania",  # 70
    "azalea",  # 71
    "water lily",  # 72
    "rose",  # 73
    "thorn apple",  # 74
    "morning glory",  # 75
    "passion flower",  # 76
    "lotus",  # 77
    "toad lily",  # 78
    "anthurium",  # 79
    "frangipani",  # 80
    "clematis",  # 81
    "hibiscus",  # 82
    "columbine",  # 83
    "desert-rose",  # 84
    "tree mallow",  # 85
    "magnolia",  # 86
    "cyclamen ",  # 87
    "watercress",  # 88
    "canna lily",  # 89
    "hippeastrum ",  # 90
    "bee balm",  # 91
    "pink quill",  # 92
    "foxglove",  # 93
    "bougainvillea",  # 94
    "camellia",  # 95
    "mallow",  # 96
    "mexican petunia",  # 97
    "bromelia",  # 98
    "blanket flower",  # 99
    "trumpet creeper",  # 100
    "blackberry lily",  # 101
    "common tulip",  # 102
    "wild rose",  # 103
]


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = (
        tf.cast(image, tf.float32) / 255.0
    )  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [*IMAGE_SIZE, 3])  # explicit size needed for TPU
    return image


def read_labelled_tfrecord(example):
    LABELLED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
        "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELLED_TFREC_FORMAT)
    image = decode_image(example["image"])
    label = tf.cast(example["class"], tf.int32)
    return image, label  # returns a dataset of (image, label) pairs


def read_unlabelled_tfrecord(example):
    UNLABELLED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
        "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        # class is missing, this competitions's challenge is to predict flower classes for the test dataset
    }
    example = tf.io.parse_single_example(example, UNLABELLED_TFREC_FORMAT)
    image = decode_image(example["image"])
    id_num = example["id"]
    return image, id_num  # returns a dataset of image(s)


def load_dataset(filenames, labelled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False  # disable order, increase speed

    dataset = tf.data.TFRecordDataset(
        filenames, num_parallel_reads=AUTO
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        read_labelled_tfrecord if labelled else read_unlabelled_tfrecord,
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


def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES, labelled=True)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat()  # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(
        AUTO
    )  # prefetch next batch while training (autotune prefetch buffer size)
    return dataset


def get_validation_dataset(ordered=False):
    dataset = load_dataset(VALIDATION_FILENAMES, labelled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO)
    return dataset


def get_test_dataset(ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labelled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
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


NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_VALIDATION_IMAGES = count_data_items(VALIDATION_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
print(
    "Dataset: {} training images, {} validation images, {} unlabelled test images".format(
        NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES
    )
)

# Define the batch size. This will be 16 with TPU off and 128 (=16*8) with TPU on
BATCH_SIZE = 16 * strategy.num_replicas_in_sync

ds_train = get_training_dataset()
ds_valid = get_validation_dataset()
ds_test = get_test_dataset()

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

from matplotlib import pyplot as plt


def batch_to_numpy_images_and_labels(data):
    images, labels = data
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    if numpy_labels.dtype == object:  # binary string in this case,
        # these are image ID strings
        numpy_labels = [None for _ in enumerate(numpy_images)]
    # If no labels, only image IDs, return None for labels (this is
    # the case for test data)
    return numpy_images, numpy_labels


def title_from_label_and_target(label, correct_label):
    if correct_label is None:
        return CLASSES[label], True
    correct = label == correct_label
    return (
        "{} [{}{}{}]".format(
            CLASSES[label],
            "OK" if correct else "NO",
            "\u2192" if not correct else "",
            CLASSES[correct_label] if not correct else "",
        ),
        correct,
    )


def display_one_flower(image, title, subplot, red=False, title_size=16):
    plt.subplot(*subplot)
    plt.axis("off")
    plt.imshow(image)
    if len(title) > 0:
        plt.title(
            title,
            fontsize=int(title_size) if not red else int(title_size / 1.2),
            color="red" if red else "black",  #
            fontdict={"verticalalignment": "center"},
            pad=int(title_size / 1.5),
        )
    return (subplot[0], subplot[1], subplot[2] + 1)


def display_batch_of_images(data_batch, predictions=None):
    """This will work with:
    display_batch_of_images(images)
    display_batch_of_images(images, predictions)
    display_batch_of_images((images, labels))
    display_batch_of_images((images, labels), predictions)
    """
    # data
    images, labels = batch_to_numpy_images_and_labels(data_batch)
    if labels is None:
        labels = [None for _ in enumerate(images)]

    # auto-squaring: this will drop data that does not fit into square
    # or square-ish rectangle
    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows

    # size and spacing
    FIGSIZE = 13.0
    SPACING = 0.1
    subplot = (rows, cols, 1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE, FIGSIZE / cols * rows))
    else:
        plt.figure(figsize=(FIGSIZE / rows * cols, FIGSIZE))

    # display
    for i, (image, label) in enumerate(
        zip(images[: rows * cols], labels[: rows * cols])
    ):
        title = "" if label is None else CLASSES[label]
        correct = True
        if predictions is not None:
            title, correct = title_from_label_and_target(predictions[i], label)
        dynamic_title_size = (
            FIGSIZE * SPACING / max(rows, cols) * 40 + 3
        )  # magic formula tested to work from 1x1 to 10x10 images
        subplot = display_one_flower(
            image, title, subplot, not correct, title_size=dynamic_title_size
        )

    # layout
    plt.tight_layout()
    if label is None and predictions is None:
        plt.subplots_adjust(wspace=0, hspace=0)
    else:
        plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    plt.savefig("docs/flowers.png")
    plt.show()


def display_training_curves(training, validation, title, subplot):
    if subplot % 10 == 1:  # set up the subplots on the first call
        plt.subplots(figsize=(10, 10), facecolor="#F0F0F0")
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor("#F8F8F8")
    ax.plot(training)
    ax.plot(validation)
    ax.set_title("model " + title)
    ax.set_ylabel(title)
    # ax.set_ylim(0.28,1.05)
    ax.set_xlabel("epoch")
    ax.legend(["train", "valid."])


ds_iter = iter(ds_train.unbatch().batch(20))

one_batch = next(ds_iter)
display_batch_of_images(one_batch)
