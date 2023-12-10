import numpy as np


def compute_predictions(model, dataset):
    images_ds = dataset.map(lambda image, idnum: image)
    probabilities = model.predict(images_ds)
    predictions = np.argmax(probabilities, axis=-1)
    return predictions


def generate_submission(model, test_dataset, output_filename, number_test_images):
    predictions = compute_predictions(model, test_dataset)
    test_ids_ds = test_dataset.map(lambda image, idnum: idnum).unbatch()
    test_ids = next(iter(test_ids_ds.batch(number_test_images))).numpy().astype("U")
    np.savetxt(
        output_filename,
        np.rec.fromarrays([test_ids, predictions]),
        fmt=["%s", "%d"],
        delimiter=",",
        header="id,label",
        comments="",
    )

    print(f"Generated {output_filename} file successfully!")
