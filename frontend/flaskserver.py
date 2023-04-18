import os
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask , render_template  , request , send_file
from tensorflow.keras.preprocessing.image import load_img , img_to_array
#improting the required libraries for the controller to work and be able to use the models to classify

app = Flask(__name__)
#initalising the flask server

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png' , 'jfif'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT


img_dir = "C:/Users/Kacper/Desktop/github/OneFish-Disseration-Project/Model/img"

classes =[] # creates categories based on the names of the folders

for x in os.listdir(img_dir):
    classes.append(x)

models = {}  # create a dictionary to hold the models

def predict(filename , model):
    img = load_img(filename , target_size = (224 , 224))
    img = img_to_array(img)
    img = img.reshape(1 , 224 ,224 ,3)

    img = img.astype('float32')
    img = img/255.0
    result = model.predict(img)

    dict_result = {}
    for i in range(30):
        dict_result[result[0][i]] = classes[i]

    res = result[0]
    res.sort()
    res = res[::-1]
    prob = res[:3]
    
    prob_result = []
    class_result = []
    for i in range(3):
        prob_result.append((prob[i]*100).round(2))
        class_result.append(dict_result[prob[i]])

    return class_result , prob_result

@app.route('/') # routes to the front page

def home():
    return render_template('index.html')

@app.route('/model')

def cnnModel():
    title = 'CNN Model'
    models[title] = load_model(os.path.join(BASE_DIR , 'modelt.hdf5'))

    return render_template('model.html', title = title )

@app.route('/transferModel')

def transferModel():
    title = 'Transfer Model'
    models[title] = load_model(os.path.join(BASE_DIR , 'modelt.hdf5'))

    return render_template('model.html', title = title)

@app.route('/rCNNModel')

def rcnnModel():
    title = 'RCNN Model'

    models[title] = load_model(os.path.join(BASE_DIR , 'modelt.hdf5'))
    return render_template('model.html', title = title)


@app.route('/result', methods =["GET","POST"])
def result():
    target_img = os.path.join(os.getcwd() , 'static/images')
    file = request.files['file']
    if file and allowed_file(file.filename):
        file.save(os.path.join(target_img , file.filename))
        img_path = os.path.join(target_img , file.filename)
        img = file.filename

    if request.referrer.endswith('/model'):
        model = models['CNN Model']
    elif request.referrer.endswith('/transferModel'):
        model = models['Transfer Model']
    elif request.referrer.endswith('/rCNNModel'):
        model = models['RCNN Model']

    class_result , prob_result = predict(img_path , model)
    predictions = {
        "class1":class_result[0],
        "class2":class_result[1],
        "class3":class_result[2],
        "prob1": prob_result[0],
        "prob2": prob_result[1],
        "prob3": prob_result[2],}

    return render_template('result.html', img = img, predictions = predictions)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000,debug = True)