#our web app framework!

#you could also generate a skeleton from scratch via
#http://flask-appbuilder.readthedocs.io/en/latest/installation.html

#Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
#HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine 
#for you automatically.
#requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template,request
import numpy as np
import sys 
import os
sys.path.append(os.path.abspath("./models"))
app = Flask(__name__)
import re 
from PIL import Image

from classifier import MyRandomForestClassifier, load_model
import cv2
import base64 
import  urllib.request
import io

global my_clf
my_clf = MyRandomForestClassifier()
my_clf = load_model("random_forest_classifier.pkl")


def convertImage(imgData1):
    #https://www.base64decoder.io/python/
     # url= data:image/png; base64, ..."
    #print(type(imgData1))
    base64_string = imgData1.decode("utf-8")
    imgstr = re.search(r'base64,(.*)',base64_string).group(1)
    decodedBytes = base64.b64decode(imgstr)
    image = np.array(Image.open(io.BytesIO(decodedBytes)))
    cv2.imwrite("output.png",image)

    '''
    with open('output.png','wb') as output:
        output.write(imgstr.decode('base64'))
	'''

@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():
    #whenever the predict method is called, we're going
    #to input the user drawn character as an image into the model
    #perform inference, and return the classification
    #get the raw data format of the image
    imgData = request.get_data()
    #encode it into a suitable format
    convertImage(imgData)

    img = cv2.imread("output.png",0)

    dst = cv2.bitwise_not(img)
    #cv2.imshow("Image", dst)
    #cv2.waitKey(0)
    #gray = dst/255
    resized_img =  np.asarray(cv2.resize(dst, (28, 28)))
    inp_img = resized_img.flatten()
    Y_pred = my_clf.predict([inp_img])
    response = np.array_str(Y_pred)
    print(Y_pred)
    return response	
	

if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
	#app.run(debug=True)