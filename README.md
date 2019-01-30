# image-classifier-using-CNNs

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
* Convolution - take the original data and creating feature maps from it
* Pooling - down sampling, most often in the form of "max-pooling", where we select a region and then take the maximum value in that region, and that becomes the new value for the entire region.
* Fully connected layers - typical nueral networkds, where all nodes are "fully connected". The convolutional layers are not fully connected like a traditional neural network.

Follow this [link](https://pythonprogramming.net/convolutional-neural-network-deep-learning-python-tensorflow-keras/?completed=/loading-custom-data-deep-learning-python-tensorflow-keras/) for more information.

After just three epochs, we have around 71% validation accuracy. One way to increase the accuracy is that we could increase the epochs.  In this case, we can also use TensorBoard, which comes with TensorFlow which helps us visualize our models as they trained.

#### The python file named "cnn.py" covers step 3

### Step 4: Analyzing Models with TensorBoard
To begin, we need to add the following to our imports:
```
from tensorflow.keras.callbacks import TensorBoard
```
Next, make TensorBoard callback object:
```
Name = "Cats-vs-Dogs-cnn-64x2-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(Name))
```
Then, add this callback to model by adding into ```.fit``` method and ```callback``` is a list, so we can pass other callbacks into this list as well. In our case, we do like:
```
model.fit(X,y,batch_size = 32,epochs=10,validation_split=0.3,callbacks=[tensorboard])
```

Hence, I have attached a few results from tensorboard with 10 epochs as shown below.
<img width="348" alt="second-model-tensorboard" src="https://user-images.githubusercontent.com/46767764/51964027-3a1f2e00-24a0-11e9-916e-71850fe09433.png">
