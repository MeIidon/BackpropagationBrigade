from flask import Flask, request
from werkzeug.datastructures import FileStorage
from PIL import Image
import tensorflow as tf

app = Flask(__name__)


@app.post("/predict")
def predict():
    image_file: FileStorage = request.files["image"]
    image: Image = Image.open(image_file)
    image_tensor: tf.Tensor = tf.convert_to_tensor(image, dtype=tf.uint8)
    print(image_tensor.shape)
    return "Rose"
