import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from visualization import CLASSES


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


def display_confusion_matrix(cmat, score, precision, recall):
    plt.figure(figsize=(15, 15))
    ax = plt.gca()
    ax.matshow(cmat, cmap="Reds")
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, fontdict={"fontsize": 7})
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES, fontdict={"fontsize": 7})
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    title = ""
    if score is not None:
        title += "f1 = {:.3f} ".format(score)
    if precision is not None:
        title += "\nprecision = {:.3f} ".format(precision)
    if recall is not None:
        title += "\nrecall = {:.3f} ".format(recall)
    if len(title) > 0:
        ax.text(
            101,
            1,
            title,
            fontdict={
                "fontsize": 18,
                "horizontalalignment": "right",
                "verticalalignment": "top",
                "color": "#804040",
            },
        )
    plt.show()


def generate_confusion_matrix(model, validation_dataset, number_validation_images):
    images_ds = validation_dataset.map(lambda image, label: image)
    labels_ds = validation_dataset.map(lambda image, label: label).unbatch()

    correct_labels = next(iter(labels_ds.batch(number_validation_images))).numpy()
    probabilities = model.predict(images_ds)
    predictions = np.argmax(probabilities, axis=-1)

    labels = range(len(CLASSES))
    cmat = confusion_matrix(correct_labels, predictions, labels=labels)
    cmat = (cmat.T / cmat.sum(axis=1)).T  # normalize

    f1 = f1_score(correct_labels, predictions, labels=labels, average="macro")
    precision = precision_score(
        correct_labels, predictions, labels=labels, average="macro"
    )
    recall = recall_score(correct_labels, predictions, labels=labels, average="macro")

    display_confusion_matrix(cmat, f1, precision, recall)
