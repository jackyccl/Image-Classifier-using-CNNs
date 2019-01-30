
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.callbacks import TensorBoard
import pickle
import time

# Name1 = "Cats-vs-Dogs-cnn-64x2-{}".format(int(time.time()))


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

dense_layers = [0,1,2]
layer_sizes = [32,64,128]
conv_layers = [1,2,3] 

for dense_layer in dense_layers:
	for layer_size in layer_sizes:
		for conv_layer in conv_layers:
			Name = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer,layer_size,dense_layer,time.time())
			tensorboard = TensorBoard(log_dir='logs/{}'.format(Name))
			print(Name)
			model = Sequential()
			#it is the first layer
			model.add(Conv2D(layer_size,(3,3),input_shape = X.shape[1:]))
			model.add(Activation("relu"))
			model.add(MaxPooling2D(pool_size=(2,2)))


			for l in range(conv_layer-1):  #we already have the first layer above, so when conv_layer = 2, we will iterate one loop
				#second layer
				model.add(Conv2D(layer_size,(3,3)))
				model.add(Activation("relu"))
				model.add(MaxPooling2D(pool_size=(2,2)))

			model.add(Flatten())  #this converts 3D feature maps to 1D feature vectors, we need to put before the dense layer
			
			for l in range(dense_layer):
				model.add(Dense(layer_size))   #normally we need to declare different sizes for dense, however for simplicity just use layer_size
				model.add(Activation("relu"))

			model.add(Dense(1))
			model.add(Activation("sigmoid"))

			model.compile(loss="binary_crossentropy",
				optimizer="adam",
				metrics=['accuracy'])

			model.fit(X,y,batch_size = 32,epochs=10,validation_split=0.3,callbacks=[tensorboard])
