from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from flask_cors import CORS, cross_origin
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
from skimage import transform
import google_api
import dishToRecipe
import ingredientsToRecipe

UPLOAD_FOLDER = '/Users/ivan.zhang/hackathon/chow_down/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
CATEGORIES = ['hamburger', 'pizza', 'sushi']
IMG_SIZE = 100

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config["CACHE_TYPE"] = "null"
model = load_model("food_model.h5")

@app.route("/image-analyze/place", methods=["POST"])
@cross_origin()
def place():
    index = image_analyze()
    result = google_api.get(CATEGORIES[index])
    return jsonify(result)

@app.route("/image-analyze/recipe", methods=["POST"])
@cross_origin()
def recipe():
    index = image_analyze()
    result = dishToRecipe.get(CATEGORIES[index])
    return jsonify(result)

@app.route("/name/place", methods=["POST"])
@cross_origin()
def name_place():
    result = google_api.get(request.json["place"])
    return jsonify(result)

@app.route("/name/recipe", methods=["POST"])
@cross_origin()
def name_recipe():
    result = dishToRecipe.get(request.json["recipe"])
    return jsonify(result)

@app.route("/ingredients", methods=["POST"])
@cross_origin()
def ingredients():
    result = ingredientsToRecipe.get(request.json["ingredients"])
    return jsonify(result)

def image_analyze():
    image = Image.open(request.files['image'])
    image = np.array(image).astype('float32')/255
    image = transform.resize(image, (IMG_SIZE, IMG_SIZE, 1))
    image = np.expand_dims(image, axis=0)

    index = np.argmax(model.predict(image))
    return index

if __name__ == '__main__':
    app.run()