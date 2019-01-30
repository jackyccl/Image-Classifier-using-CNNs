from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard
import pickle
import time

Name = "Cats-vs-Dogs-cnn-64x2-{}".format(int(time.time()))

tensorboard = TensorBoard(log_dir='logs/{}'.format(Name))

#if you want to run multiple models on your gpu, we can limit how much the vram will consume while running the model
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

with open('X.pickle','rb') as f:
	X = pickle.load(f)

with open('y.pickle','rb') as f:
	y = pickle.load(f)

#normalizing data, scale that data. Hence for pixels, the min is 0 and the max is 255
# actually we can just divided by 255

X = X/255.0

model = Sequential()
#one layer
model.add(Conv2D(64,(3,3),input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
#two layer
model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
#third layer
model.add(Flatten())  #this converts 3D feature maps to 1D feature vectors
# model.add(Dense(64))
# model.add(Activation("relu"))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(loss="binary_crossentropy",
	optimizer="adam",
	metrics=['accuracy'])

model.fit(X,y,batch_size = 32,epochs=3,validation_split=0.3,callbacks=[tensorboard])
