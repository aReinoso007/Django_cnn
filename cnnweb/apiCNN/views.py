from django.shortcuts import render

#Para guardar la imagen en django
from django.core.files.storage import FileSystemStorage

#Aqui es para cargar el modelo de la red
#import tensorflow as tf
#from keras import backend as k
from keras.models import load_model, model_from_json
from keras.preprocessing import image
import json
import json
from tensorflow import Graph, Session
import tensorflow as tf
import numpy as np
#from skimage.transform import resize
import matplotlib as plt

#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#x_train = x_train.reshape(60000, 28, 28, 1)
#x_train = x_train.astype('float32')
#x_train /= 255

img_height, img_width=28,28





with open('./models/imagenet_classes.json','r') as f:
    labelInfo=f.read()
labelInfo=json.loads(labelInfo)


model_graph = Graph()
with model_graph.as_default():
    tf_session = Session()
    with tf_session.as_default():
        model=load_model('./models/modelo.h5')


#predicts = model.predict(x_train)

def index(request):
    context={'a':1}
    return render(request, 'index.html', context)

def predictImage(request):
    print (request)
    print (request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    testimage='.'+filePathName
    img = image.load_img(testimage, target_size=(img_height, img_width))
    x = image.img_to_array(img)
    print(x.shape)
    x=x/255
    x=x.reshape(-1,img_height, img_width,1)
    with model_graph.as_default():
        with tf_session.as_default():
            predi=model.predict(x)
            proba = predi[:,np.argmax(model.predict(x))]
            proba = proba*100
    print("Probabilidad",proba)
    print("Prediccion",np.argmax(predi[1]))
    predictedLabel=labelInfo[str(np.argmax(predi[0]))]
    #predictedLabel= predicts(np.array( [x] ))
    print("label prediccion",predictedLabel[0])
    context={'filePathName':filePathName,'predictedLabel':predictedLabel[1], 'Probabilidad':proba}
    return render(request,'index.html',context) 

def viewDataBase(request):
    import os
    listOfImages=os.listdir('./media/')
    listOfImagesPath=['./media/'+i for i in listOfImages]
    context={'listOfImagesPath':listOfImagesPath}
    return render(request,'viewDB.html',context) 