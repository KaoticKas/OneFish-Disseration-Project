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
#checks if the file passed to the server is of a required type


img_dir = "C:/Users/Kacper/Desktop/github/OneFish-Disseration-Project/Model/img"

classes =[] # creates categories based on the names of the folders

for x in os.listdir(img_dir):
    classes.append(x)
# makes a list of fish species based on folder names

models = {}  # create a dictionary to hold the models

def predict(filename , model, img_size):
    num_probablities = 3
    img = load_img(filename , target_size = (img_size , img_size))
    img = img_to_array(img)
    img = img.reshape(1 , img_size ,img_size ,3)
    # loads the image and resizes it to what the models were trained on
    img = img.astype('float32')
    img = img/255.0
    result = model.predict(img)
    #normalises the image and passes it to the model
    dict_result = {}
    for i in range(30):
        dict_result[result[0][i]] = classes[i]
    #takes the model results and makes a dictonary with the key being the class name
    res = result[0]
    res.sort()
    res = res[::-1]
    prob = res[:num_probablities]
    
    prob_result = []
    class_result = []
    for i in range(num_probablities):
        prob_result.append((prob[i]*100).round(2))
        class_result.append(dict_result[prob[i]])
    # makes two lists with the percentage probablility of the image and what top 3 classes it is.
    return class_result , prob_result

# underneath are a list of routes that the server takes based on the urls. Most of the routes are Get pages as no data is expected to return

@app.route('/') # routes to the front page

def home():
    return render_template('index.html')

@app.route('/model')

def cnnModel():
    title = 'CNN Model'
    models[title] = load_model(os.path.join(BASE_DIR , 'modelV3.hdf5'))

    return render_template('model.html', title = title )

@app.route('/transferModel')

def transferModel():
    title = 'Transfer Model'
    models[title] = load_model(os.path.join(BASE_DIR , 'modelTransfer.hdf5'))

    return render_template('model.html', title = title)

@app.route('/rCNNModel')

def rcnnModel():
    title = 'RCNN Model'

    models[title] = load_model(os.path.join(BASE_DIR , 'modelV3.hdf5'))
    return render_template('model.html', title = title)


@app.route('/result', methods =["GET","POST"])
def result():
    #target_img = os.path.join(os.getcwd() , 'static/images')
    target_img = (r"C:\Users\Kacper\Desktop\github\CIFAR-10-image-classification\CIFAR-10-image-classification\static")
    file = request.files['file']
    if file and allowed_file(file.filename):
        file.save(os.path.join(target_img , file.filename))
        img_path = os.path.join(target_img , file.filename)
        img = file.filename
    #Loads the file name into a static/images folder to be used for the prediction 
    if request.referrer.endswith('/model'):
        model = models['CNN Model']
        img_size = 256
    elif request.referrer.endswith('/transferModel'):
        model = models['Transfer Model']
        img_size = 224
    elif request.referrer.endswith('/rCNNModel'):
        model = models['RCNN Model']
        img_size = 256
    # if else statement to determin which model to laod
    class_result , prob_result = predict(img_path , model, img_size)
    predictions = {
        "class1":class_result[0],
        "class2":class_result[1],
        "class3":class_result[2],
        "prob1": prob_result[0],
        "prob2": prob_result[1],
        "prob3": prob_result[2],}
    #produces a list of predictions
    return render_template('result.html', img = img, predictions = predictions)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000,debug = True)