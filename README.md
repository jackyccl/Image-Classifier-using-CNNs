# image-classifier-using-cnn

In this project, we are going to perform deep learning with Python, Tensorflow and Keras. Following the release of deep learning libraries (i.e. Tensorflow), one of the higher-level API-like libraries that sit on top of tensorflow and has easily become the most popular is Keras.

Previously, I did a tutorial on using [ImageNet for transfer learning](https://github.com/jackyccl/Image-Classifier-Using-Transfer-Learning). In this tutorial, I will build my own model to classify the pictures between dogs and cats. Now, let's get start.

### Step 1: Install Tensorflow and Keras
To install TensorFlow, simply do a:
```
pip install --upgrade tensorflow
```
Keras is now a superset, included with Tensorflow now, and we can just issue the command as:
```
import tensorflow.keras as keras  or import tensorflow.python.keras as keras 
```

### Step 2: Loading in your own dataset
I grabbed the [Dogs vs Cats dataset](https://www.microsoft.com/en-us/download/confirmation.aspx?id=54765) from Microsoft. Next, unzip the dataset and stored it in a directory called *PetImages*. Next, we need to convert this dataset to training data. The largest issue is not all of these images are the same size. Hence, we need to reshape the image to has the same dimensions. As always, we change the colourful picture to grayscale using cv2 library. we will use matplotlib to display the images, just for reference and make sure the images change to grayscale.

Make sure to install matplotlib and opencv using the commands : ```pip install matpltlib``` and ```pip install opencv-python```

Regarding the dataset, one thing we want to do is to make sure our data is balanced. By balanced, I mean there are the same number of examples for each class(same number of dogs and cats) in our case. If we do not balance, the model will initially learn that the best thing to do is to predict only one class, whichever is the most common. 

Besides, we also need to shuffle the data. This is to avoid the condition that the classifier will learn to just predict dogs always if dog's images come first. Hence, we need to import ```random``` to shuffle our dataset.

Last but not least, we save our data in ```.pickle``` so that we don't need to keep calculating it every time. 

#### Note: The python file named "training_data.py" basically covers the whole step 2, and I wrote some comments at the side of the code. 

### Step 3: Convolutional Neural Networks
The Convolutional Neural Netwrok is currently the state of the art for detecting what an image is or what is contained in the image.

The basic CNN structure is as follows: Convolution -> Pooling -> Convolution -> Pooling -> Fully Connected Layer -> Output
*Convolution - take the original data and creating feature maps from it
*Pooling - down sampling, most often in the form of "max-pooling", where we select a region and then take the maximum value in that region, and that becomes the new value for the entire region.
*Fully connected layers - typical nueral networkds, where all nodes are "fully connected". The convolutional layers are not fully connected like a traditional neural network.
