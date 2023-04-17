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
    title = 'CNN Model'
    return render_template('model.html', title = title)

@app.route('/transferModel')
def transferModel():
    title = 'Transfer Model'
    return render_template('model.html', title = title)

@app.route('/rCNNModel')
def rcnnModel():
    title = 'RCNN Model'
    return render_template('model.html', title = title)


@app.route('/result')
def result():
    
    return render_template('result.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000,debug = True)