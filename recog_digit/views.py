from django.shortcuts import render
from django.http import HttpResponse
import base64
import numpy as np
import pickle
from PIL import Image
from io import BytesIO
from django.views.decorators.csrf import csrf_exempt
from mlutils.mycnn import model as cnn_model, optimizer
from mlutils.myann import ArtificialNeuralNetwork
import torch



picklefile = open('media/recog_digit/ANN_params.pkl', 'rb')
ann_params = pickle.load(picklefile)
picklefile.close()

ann_model = ArtificialNeuralNetwork(ann_params['inodes'], ann_params['hnodes'], ann_params['onodes'], ann_params['lr'])
ann_model.assign_weights(ann_params['wih'], ann_params['who'])

checkpoint = torch.load("media/recog_digit/CNN_params.pth.tar")
cnn_model.load_state_dict(checkpoint['state_dict'])
optimizer.load_state_dict(checkpoint['optimizer'])


def parseImage(imgData):
    dimensions = (28, 28)
    imgstr = imgData.split(",")[1]
    encoded_image = imgstr
    decoded_image = base64.b64decode(encoded_image)
    img = Image.open(BytesIO(decoded_image)).convert('LA')      # image is (280, 280)
    img = img.resize(dimensions, Image.ANTIALIAS)               # image is (28, 28)
    pixels = np.asarray(img, dtype='uint8')                 # pixels.shape == (28, 28, 2)
    pixels = pixels[:, :, 0]
    # img = Image.fromarray(pixels)         # to display the img
    # img.show()
    return pixels


@csrf_exempt
def ann(request):
    if request.method == "POST":
        img_array = parseImage(request.body.decode("utf-8"))
        label = ann_model.identify_num(img_array) 
        return HttpResponse(label)
    return render(request, 'digit/index.html')


@csrf_exempt
def cnn(request):
    if request.method == "POST":
        img_array = parseImage(request.body.decode("utf-8"))
        label = cnn_model.identify_num(img_array)
        return HttpResponse(label)
    return render(request, 'digit/index.html')