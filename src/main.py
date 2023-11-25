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


from visualization import CLASSES
from models import create_vgg16_model, compile_model
from metrics import display_training_curves, generate_confusion_matrix
from prediction import generate_submission


def main():
    print("Tensorflow version " + tf.__version__)

    strategy = get_strategy()

    BATCH_SIZE = 16 * strategy.num_replicas_in_sync
    IMAGE_SIZE_192 = [192, 192]
    DS_PATH = "data"
    PATH_192 = DS_PATH + "/tfrecords-jpeg-192x192"

    TRAINING_FILENAMES_192 = tf.io.gfile.glob(PATH_192 + "/train/*.tfrec")
    VALIDATION_FILENAMES_192 = tf.io.gfile.glob(PATH_192 + "/val/*.tfrec")
    TEST_FILENAMES_192 = tf.io.gfile.glob(PATH_192 + "/test/*.tfrec")  # 100 - 10

    NUM_TRAINING_IMAGES_192 = count_data_items(TRAINING_FILENAMES_192)
    NUM_VALIDATION_IMAGES_192 = count_data_items(VALIDATION_FILENAMES_192)
    NUM_TEST_IMAGES_192 = count_data_items(TEST_FILENAMES_192)

    print(
        "Dataset - 192 x 192: ",
        f"{NUM_TRAINING_IMAGES_192} training images",
        f"{NUM_VALIDATION_IMAGES_192} validation images",
        f"{NUM_TEST_IMAGES_192} unlabelled test images",
    )

    ds_train_192 = get_training_dataset(
        TRAINING_FILENAMES_192,
        IMAGE_SIZE_192,
        BATCH_SIZE,
    )
    ds_valid_192 = get_validation_dataset(
        VALIDATION_FILENAMES_192,
        IMAGE_SIZE_192,
        BATCH_SIZE,
    )
    ds_test_192 = get_test_dataset(
        TEST_FILENAMES_192,
        IMAGE_SIZE_192,
        BATCH_SIZE,
    )

    visualize_data(ds_train_192, ds_valid_192, ds_test_192)

    with strategy.scope():
        model_192 = create_vgg16_model(strategy, IMAGE_SIZE_192 + [3], len(CLASSES))

    optimizer_192 = tf.keras.optimizers.Adam()

    with strategy.scope():
        compile_model(model_192, optimizer_192)
        model_192.summary()

    EPOCHS = 16
    STEPS_PER_EPOCH_192 = NUM_TRAINING_IMAGES_192 // BATCH_SIZE

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        "model.h5",
        save_best_only=True,
        monitor="val_sparse_categorical_accuracy",
        mode="max",
    )
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_sparse_categorical_accuracy",
        patience=4,
        mode="max",
    )

    history_192 = model_192.fit(
        ds_train_192,
        validation_data=ds_valid_192,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH_192,
        callbacks=[model_checkpoint_callback, early_stopping_callback],
    )

    display_training_curves(
        history_192.history["loss"],
        history_192.history["val_loss"],
        "loss",
        211,
    )
    display_training_curves(
        history_192.history["sparse_categorical_accuracy"],
        history_192.history["val_sparse_categorical_accuracy"],
        "accuracy",
        212,
    )

    generate_confusion_matrix(
        model_192,
        ds_valid_192,
        NUM_VALIDATION_IMAGES_192,
    )

    dataset_192 = dataset_192.unbatch().batch(20)
    batch_192 = iter(dataset_192)

    images, labels = next(batch_192)
    probabilities = model_192.predict(images)
    predictions = np.argmax(probabilities, axis=-1)
    display_batch_of_images((images, labels), predictions)

    generate_submission(model_192, ds_test_192, "submission.csv", NUM_TEST_IMAGES_192)


if __name__ == "__main__":
    main()
