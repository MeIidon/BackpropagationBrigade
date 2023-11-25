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

    with strategy.scope():
        model_512 = create_vgg16_model(strategy, IMAGE_SIZE_512 + [3], len(CLASSES))

    optimizer_512 = tf.keras.optimizers.Adam()

    with strategy.scope():
        compile_model(model_512, optimizer_512)
        model_512.summary()

    EPOCHS = 16
    STEPS_PER_EPOCH_512 = NUM_TRAINING_IMAGES_512 // BATCH_SIZE

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

    history_512 = model_512.fit(
        ds_train_512,
        validation_data=ds_valid_512,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH_512,
        callbacks=[model_checkpoint_callback, early_stopping_callback],
    )

    display_training_curves(
        history_512.history["loss"],
        history_512.history["val_loss"],
        "loss",
        211,
    )
    display_training_curves(
        history_512.history["sparse_categorical_accuracy"],
        history_512.history["val_sparse_categorical_accuracy"],
        "accuracy",
        212,
    )

    generate_confusion_matrix(
        model_512,
        ds_valid_512,
        NUM_VALIDATION_IMAGES_512,
    )

    dataset_512 = dataset_512.unbatch().batch(20)
    batch_512 = iter(dataset_512)

    images, labels = next(batch_512)
    probabilities = model_512.predict(images)
    predictions = np.argmax(probabilities, axis=-1)
    display_batch_of_images((images, labels), predictions)

    generate_submission(model_512, ds_test_512, "submission.csv", NUM_TEST_IMAGES_512)


if __name__ == "__main__":
    main()
