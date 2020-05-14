from django.shortcuts import render
from django.http import HttpResponse
import base64
import numpy as np
import pickle
from PIL import Image
from io import BytesIO
from django.views.decorators.csrf import csrf_exempt


def parseImage(imgData):
    dimensions = (28, 28)
    imgstr = imgData.split(",")[1]
    encoded_image = imgstr
    decoded_image = base64.b64decode(encoded_image)
    img = Image.open(BytesIO(decoded_image)).convert('LA')      # image is (280, 280)
    img = img.resize(dimensions, Image.ANTIALIAS)               # image is (28, 28)
    pixels = np.asarray(img, dtype='uint8')                 # pixels.shape == (28, 28, 2)
    pixels = pixels[:, :, 0]
    # img = Image.fromarray(pixels)
    # img.show()
    return pixels


class neuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        fileobj = open("media/recog_digit/model_param.pkl", 'rb')
        model_param = pickle.load(fileobj)
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate
        self.activation_function = lambda z: 1 / (1 + np.exp(-z))

        # loading trained weight matrices wih, who in which w_i_j, is link from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = model_param['wih']
        self.who = model_param['who']

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T

        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


@csrf_exempt
def index(request):
    if (request.method == "POST"):
        img_array = parseImage(request.body.decode("utf-8"))
        input_nodes = 784
        hidden_nodes = 200
        output_nodes = 10
        learning_rate = 0.01

        # create instance of neural network
        n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
        img_data = 255.0 - img_array.reshape(784)
        img_data = (img_data / 255.0 * 0.99) + 0.01  # Scale the data

        # query the network
        outputs = n.query(img_data)
        label = np.argmax(outputs)  # index of the highest value corresponds to the label

        return HttpResponse(label)
    return render(request, 'digit/index.html')

