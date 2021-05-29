from django.shortcuts import render
import numpy as np
import tensorflow as tf
from PIL import Image
import keras
import os
from keras.models import load_model
import requests
import urllib
from django.contrib.staticfiles.storage import staticfiles_storage
# Create your views here.
def DeepModel(request):
    result=""
    test_input=''
    url = staticfiles_storage.url("cat_dog_model.h5")
    model=load_model(url)
    if request.method == "POST":
        try:
            test_input=request.POST["input_url"]
            new_image=Image.open(requests.get(test_input, stream=True).raw)
            new_image=new_image.resize((64,64))
            new_image=np.array(new_image)
            predictions=model.predict(np.expand_dims(new_image, 0))
            print(predictions)
            if (predictions>0.5):
                result="Dog"

            else:
                result="Cat"
        except:
            result="Image not loaded properly"
        print(result)
    return render(request,"home.html",context={'result':result,'result_img':test_input})
