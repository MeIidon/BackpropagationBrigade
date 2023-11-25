import tensorflow as tf


def create_vgg16_model(strategy, input_shape, n_classes):
    with strategy.scope():
        pretrained_model = tf.keras.applications.VGG16(
            weights="imagenet", include_top=False, input_shape=input_shape
        )
        pretrained_model.trainable = True

        model = tf.keras.Sequential(
            [
                pretrained_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.Dense(n_classes, activation="softmax"),
            ]
        )

        return model


def compile_model(model, optimizer):
    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
