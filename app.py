#!/usr/bin/env python
# coding: utf-8

# In[1]:


# !pip install flask


# In[2]:


from flask import Flask, url_for, redirect
import os
import shutil


# In[3]:


def make_over_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


# In[ ]:





# In[4]:


#os.chdir('C:\\Users\\Consultant\\Desktop\\vision api')

s = os.getcwd()

make_over_dir(os.path.join(s, 'templates'))
make_over_dir(os.path.join(s,'uploads'))


# In[5]:


a = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/css/bootstrap.min.css">
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.1.1/jquery.min.js" type="text/javascript"></script>
<script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.7/js/bootstrap.min.js" type="text/javascript"></script>
  <title>Image Recognition Server</title>
  <body class style="margin:10px;padding:10px">
    <div class="page-header" id="banner">
      <div class="row">
        <div class="col-lg-8 col-md-7 col-sm-6">
          <h3>Image Recognition Server</h3>
          <p class="lead">Upload the image and find out what an animal is located on it?</p>
        </div>
      </div>
      <form action="" method=post enctype=multipart/form-data>
        <input type=file name=file>
        <input type=submit value=Upload>
    </form>
    </div>
    <p style="margin-bottom:2cm;"></p>
    <div class="row">
        <div class="col-lg-4">
          <div class="page-header">
            <h3 id="tables">Result</h3>
          </div>
          <div>
          
          </div>
          <div class="bs-component">
            <table class="table table-hover">
                <tr class="table-active">
                 
                  <th scope="col">Predict</th>
                </tr>
                
                <tr>
                    
                         <td> {{label}} </td>
                  </tr>
            </table> 
        </div>
      </div>
  </body>
  '''


# In[6]:

s = os.path.join(s,'templates')
html_file = open(os.path.join(s,'index.html'),'w')
html_file.write(a)
html_file.close()


# In[7]:


#pip install tensorflow


# In[8]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from keras_preprocessing import image

import numpy as np


# In[9]:


allowed_extensions = set(['jpg','jpeg','png'])
image_size = (224,224)
upload_folder = 'uploads'
model = None


# In[10]:


def load_model():
    global model
    model = ResNet50(weights = "imagenet")


# In[11]:


def allowed_file (file_name):
    return '.' in file_name and file_name.split('.',1)[1]in allowed_extensions


# In[12]:


def predict(file):
    img = image.load_img(file,target_size = image_size)
    img = img_to_array(img)
    img = np.expand_dims(img,axis = 0)
    
    img = preprocess_input(img)
    
    probs = model.predict(img)
    
    
    topfive_labels = " "
    
    for (imagenetID, label, prob) in decode_predictions(probs, top = 5)[0]:
        topfive_labels  += str(" "+ label + ", " + str(int(100*prob)) + "%, ")
    
    return topfive_labels


# In[13]:


from werkzeug.utils import secure_filename
from flask import request, redirect, url_for, send_from_directory, render_template


# In[14]:


app = Flask(__name__)
app.config['upload_folder'] = upload_folder


# In[15]:


@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('',filename='uploads'+filename),code = 301)


# In[16]:



@app.route('/', methods = ['POST'])
def upload_file():
    file = request.files['file']
    filename = ""
    output = ""
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['upload_folder'],filename)
        file.save(file_path)
    
        output = predict(file_path)
    return render_template("index.html", label = output)


# In[17]:


@app.route('/')
def template_test():
    return render_template("index.html", label = "Mariachis Were Here!", imagesource = 'file://null')


# In[ ]:





# In[ ]:


if __name__ == "__main__":
    load_model()
    app.run(host='0.0.0.0',port=80)


# In[ ]:





# In[ ]:




