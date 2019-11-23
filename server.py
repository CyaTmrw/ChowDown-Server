from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from skimage import transform

UPLOAD_FOLDER = '/Users/ivan.zhang/hackathon/chow_down/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
CATEGORIES = ['hamburger', 'pizza', 'sushi']
IMG_SIZE = 100

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
model = load_model("food_model")

@app.route("/image-analyze", methods=["POST"])
def image_analyze():
    
    image = Image.open(request.files["image"])
    image = np.array(image).astype('float32')/255
    image = transform.resize(image, (IMG_SIZE, IMG_SIZE, 1))
    image = np.expand_dims(image, axis=0)

    index = np.argmax(model.predict(image))
    result = CATEGORIES[index]
    return jsonify({
        "result": result
    })

if __name__ == '__main__':
    app.run()