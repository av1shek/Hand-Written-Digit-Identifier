## Hand Written Digit Identifier from Scratch
A Django web app to identify hand written Digit using machine learning model and trained over MNIST handwritten digit training dataset and has an accuraccy of 97.6% for ANN model and 98.3% for CNN model on MNIST test data set.<br>
ANN model is implemented from scratch while CNN model is implemented using Pytorch.<br>
Trained weights are stored in ```model_param.pkl``` and ```my_checkpoint.pth.tar``` file.<br>
#### [Click Here](https://mlapps.herokuapp.com/digit) to visit the site.
Some observations:<br>
i) Works well when digits are drawn in mid of canvas with full size.<br>
ii) CNN model gives better performance than ANN (as expected). 
