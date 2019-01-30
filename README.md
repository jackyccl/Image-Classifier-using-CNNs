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
<p align="center"> 
<img src="https://user-images.githubusercontent.com/46767764/51964027-3a1f2e00-24a0-11e9-916e-71850fe09433.png">
</p>

#### Discussion:
Notice the shape of validation loss. Loss is the measure of error. After 4th epoch, the validation loss starts to increase, but interestingly, the validation accuracy continued to hold. This should alert you that you are almost certainly beginning to over-fit. The reason is the model is constantly trying to decrease our in-sample loss, at some point, rather than learning general patterns about the actual datas, the model begins to memorize input data. In this case, any new data attempt to feed the model, it will results in poor judgement.

#### The python file named "cnn.py" covers step 3 and 4

### Step 5: Optimizing our Model based on TensorBoard
The most basic things is to modify nodes per layer and layers, as well as 0,1,or 2 dense layers.
we can do it using a simple [for-loop]{https://pythonprogramming.net/tensorboard-optimizing-models-deep-learning-python-tensorflow-keras/?completed=/tensorboard-analysis-deep-learning-python-tensorflow-keras/} like:

```
import time

dense_layers = [0,1,2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(time.time()))
            print(NAME)
```

and we will combine the above for loop into out model and named it as "cnn2.py" and train the model. The results from the tensorboard are:
<p align="center"> 
<img src="https://user-images.githubusercontent.com/46767764/51964822-a864f000-24a2-11e9-8d04-f023d990f994.png">
</p>

#### Discussion
Although the results are not similar to [this](https://pythonprogramming.net/tensorboard-optimizing-models-deep-learning-python-tensorflow-keras/?completed=/tensorboard-analysis-deep-learning-python-tensorflow-keras/), however it shows almost the same trends. Normally, it is tempting to take highest validation accuracy model, but I tend to choose lowest (best) validation loss models. The models with 0 dense layers seemed to do better overall.

Zooming into validation accuracy graph, Here are the top 5:
3 conv, 64 nodes per layer, 0 dense
3 conv, 128 nodes per layer, 0 dense
3 conv, 32 nodes per layer, 0 dense
3 conv, 32 nodes per layer, 1 dense
3 conv, 64 nodes per layer, 2 dense

From here, we can be comfortable with 0 dense, and 3 convolutional layers. Results for top 3 models:
<p align="center"> 
<img src="https://user-images.githubusercontent.com/46767764/51965493-b4ea4800-24a4-11e9-95f6-28374b9c2f53.png">
</p>

#### The python file named "cnn2.py" covers step 5


### Step 6: Use Trained Model

#### The python file named "cnn_final.py" covers step 6
The model we choose is 3 conv layers with 64x64 kernel sizes and without dense layer. It gives me:
```
loss: 0.3009 - acc: 0.8705 - val_loss: 0.4452 - val_acc: 0.8009
```


