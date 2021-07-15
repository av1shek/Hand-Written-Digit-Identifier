import pickle
import imageio
import matplotlib.pyplot
import numpy as np
import scipy.ndimage


class ArtificialNeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        self.lr = learning_rate
        
        # initialising weight matrices wih, who in which w_i_j, is link from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc 
        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))
        
    def activation_function(self, z):
        return 1/(1 + np.exp(-z))

    def train(self, inputs_list, targets_list):
        '''Train the model for one epoch and update the weights'''
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors, split by weights
        # in their ratios of weights to inputs for each node
        hidden_errors = np.dot(self.who.T, output_errors)
        
        # update the weights
        self.who += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))
        
    def start_training(self, training_data_list, epochs):
        '''Train the model for given number of epochs with given input'''
        for e in range(epochs):
            print("iteration =>", e)
            for record in training_data_list:
                all_values = record.split(',')
                inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01  # scale and shift the inputs
                targets = np.zeros(self.onodes) + 0.01
                targets[int(all_values[0])] = 0.99
                
                # without rotating
                self.train(inputs, targets)

                # rotated anticlockwise by x=10 degrees
                inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), 10, cval=0.01, order=1, reshape=False)
                self.train(inputs_plusx_img.reshape(784), targets)

                # rotated clockwise by x=-10 degrees
                inputs_minusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), -10, cval=0.01, order=1, reshape=False)
                self.train(inputs_minusx_img.reshape(784), targets)
    
    def test(self, test_data_list):
        scorecard = []
        for record in test_data_list:
            all_values = record.split(',')      # split the record by the ',' commas
            correct_label = int(all_values[0])  # First value of each line is label in dataset
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01   # scale and shift the inputs
            outputs = self.query(inputs)
            label = np.argmax(outputs)
            if (label == correct_label):
                scorecard.append(1)
            else:
                scorecard.append(0)
                
        # calculate the performance score, the fraction of correct answers
        scorecard_array = np.asarray(scorecard)
        print ("performance = ", scorecard_array.sum() / scorecard_array.size)


    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs

    def identify_num(self, img_array):
        # Invert the color
        img_data  = 255.0 - img_array.reshape(784)
        img_data = (img_data / 255.0 * 0.99) + 0.01  # Scale the data

        # plot image
        matplotlib.pyplot.imshow(img_data.reshape(28,28), cmap='Greys', interpolation='None')

        # query the network
        outputs = self.query(img_data)
        label = np.argmax(outputs)   # index of the highest value corresponds to the label
        print("Model says ", label)
        return label

    def save_params(self):
        parameters = {'who':self.who, 'wih':self.wih, 'inodes':self.inodes, 'hnodes':self.hnodes, 'onodes':self.onodes, 'lr':self.lr}
        file = '/content/drive/My Drive/machine learning/makeyourownneuralnetwork-master/ANN_params.pkl'
        fileobj = open(file, 'wb')
        pickle.dump(parameters, fileobj)
        fileobj.close()

    def assign_weights(self, wih, who):
        self.who = who
        self.wih = wih




if __name__ == '__main__':
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    learning_rate = 0.01
    epochs = 10

    # load the mnist training data CSV file into a list
    training_data_file = open("/content/drive/My Drive/machine learning/makeyourownneuralnetwork-master/mnist_dataset/mnist_train.csv", 'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    # load the mnist test data CSV file into a list
    test_data_file = open("/content/drive/My Drive/machine learning/makeyourownneuralnetwork-master/mnist_dataset/mnist_test.csv", 'r')
    test_data_list = test_data_file.readlines()
    test_data_file.close()


    model = ArtificialNeuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)
    model.start_training(training_data_list, epochs)
    model.test(test_data_list)

    img_array = imageio.imread('/content/drive/My Drive/machine learning/makeyourownneuralnetwork-master/my_own_images/2828_my_own_noisy_6.png', as_gray=True)
    model.identify_num(img_array)      # test the neural network on single image

    model.save_params()