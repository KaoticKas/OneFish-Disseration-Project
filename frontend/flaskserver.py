import os
import uuid
import flask
import urllib
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask , render_template  , request , send_file
from tensorflow.keras.preprocessing.image import load_img , img_to_array

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/model')
def cnnModel():
    return render_template('model.html')
@app.route('/transferModel')
def transferModel():
    return True
@app.route('/rCNNModel')
def rcnnModel():
    return True
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000,debug = True)