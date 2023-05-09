import os
import uuid
from tensorflow.keras.models import load_model
from flask import Flask , render_template  , request
from tensorflow.keras.preprocessing.image import load_img , img_to_array
#improting the required libraries for the controller to work and be able to use the models to classify

app = Flask(__name__)
#initalising the flask server

root_dir = os.path.dirname(os.path.abspath(__file__))
extensions = set(['jpg' , 'jpeg' , 'png' , 'jfif']) # allowed file formats

def allowed_file(filename):
    return '.' in filename and \
     filename.rsplit('.', 1)[1].lower() in extensions
#checks if the file passed to the server is of a required type


img_dir = os.path.join(os.getcwd() , 'Model\img')
classes = [] # creates categories based on the names of the folders


print(img_dir)
for x in os.listdir(img_dir):
    classes.append(x)
# makes a list of fish species based on folder names

models = {}  # create a dictionary to hold the models

def predict(filename , model, img_size):
    display_probs = 3
    img = load_img(filename , target_size = (img_size , img_size))
    img = img_to_array(img).reshape(1 , img_size ,img_size ,3)
    # loads the image and resizes it to what the models were trained on
    img = img/255.0
    result = model.predict(img)
    #normalises the image and passes it to the model
    dict_result = {}
    for i in range(len(classes)):
        dict_result[result[0][i]] = classes[i]
    #takes the model results and makes a dictonary with the key being the class name
    res = result[0]
    res.sort()
    res = res[::-1]
    prob = res[:display_probs]
    #takes top 3 highest model predicitons into a list.    
    prob_result = []
    class_result = []
    for i in range(display_probs):
        prob_result.append((prob[i]*100).round(2))
        class_result.append(dict_result[prob[i]])
    # makes two lists with the percentage probablility of the image and what top 3 classes it is.
    return class_result , prob_result

# underneath are a list of routes that the server takes based on the urls. Most of the routes are Get pages as no data is expected to return

@app.route('/') # routes to the front page

def home():
    return render_template('index.html')

@app.route('/cnnModel')

def cnnModel():
    title = 'CNN Model'
    models[title] = load_model(os.path.join(root_dir , 'models\ModelV3.hdf5'))
    # loads the model into dictonary models with the Models name as the key
    return render_template('model.html', title = title )

@app.route('/transferModel')

def transferModel():
    title = 'Transfer Model'
    models[title] = load_model(os.path.join(root_dir , 'models\modelTransfer.hdf5'))

    return render_template('model.html', title = title)

@app.route('/model3')

def model3():
    title = 'Model3'

    models[title] = load_model(os.path.join(root_dir , 'models\modelV3.hdf5'))
    return render_template('model.html', title = title)
    #routes to one of the models and preloads the model.
@app.route('/help')

def help():
    return render_template('help.html')

@app.route('/result', methods =["GET","POST"])
def result():
    if request.method == "POST":
        error_msg = ""
        target_img = os.path.join(root_dir, 'static\images') 
        #assigns the image directory for the image to be classified by using the OS Module.
        file = request.files['file']
        if file and allowed_file(file.filename):
            unique_name = str(uuid.uuid4())[:8] + "_" +file.filename
            file.save(os.path.join(target_img , unique_name))
            img_path = os.path.join(target_img , unique_name)
            img = unique_name
            #Loads the file name into a static/images folder to be used for the prediction
            if request.referrer.endswith('/cnnModel'):
                model = models['CNN Model']
                img_size = 256 # set image size to required size
            elif request.referrer.endswith('/transferModel'):
                model = models['Transfer Model']
                img_size = 224
            elif request.referrer.endswith('/model3'):
                model = models['Model3']
                img_size = 256
            else:
                model = models['CNN Model']
                img_size = 256
    # if else statement to determin which model to load 
            class_result , prob_result = predict(img_path, model, img_size)
            predictions = {
                "class1":class_result[0],
                "class2":class_result[1],
                "class3":class_result[2],
                "prob1": prob_result[0],
                "prob2": prob_result[1],
                "prob3": prob_result[2],
                }
        #produces a list of predictions to feed into results page
            return render_template('result.html', img = img, predictions = predictions)
        else:
            error_msg = "the was no image uploaded or the file type was not supported, Please Choose an Image that is of type PNG, JPG or JFIF"
            return render_template('index.html', error_msg = error_msg)
        #error catch
    else:
        error_msg = "method was not POST"
        return render_template('index.html', error_msg = error_msg)
    #error catch

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000,debug = True)