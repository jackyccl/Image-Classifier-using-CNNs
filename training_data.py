import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle

DATADIR = "C:/Users/User/Desktop/cat_dog_classifier/PetImages"
CATEGORIES = ["Dog","Cat"]   #0 is the DOG , 1 is the CAT This is the label
IMG_SIZE = 50  #be careful when dealing with smaller features in the image
training_data = []
X = []
y =[]


def create_training_data():
	#just to read through all the images inside the folder
	for category in CATEGORIES:
		path = os.path.join(DATADIR,category)  #path to cats or dogs dir
		class_number = CATEGORIES.index(category)
		for image in os.listdir(path):
			try:
				image_array = cv2.imread(os.path.join(path,image),cv2.IMREAD_GRAYSCALE)
				new_array = cv2.resize(image_array,(IMG_SIZE,IMG_SIZE))   #resize all the images
				training_data.append([new_array,class_number])
			# 	plt.imshow(new_array,cmap='gray')
			# 	plt.show()
			# 	break
			# break
			except Exception as e:
				#usually put error message here, like reading file fail
				#print(e)  #some of the images are broken so we decided to just pass some of it
				pass



create_training_data()
random.shuffle(training_data) #just to shuffle data, so that it wont be keep seeing the same image over the time and prevent memorization
# for sample in training_data[:10]:
# 	print(sample[0],sample[1])  #just for checking purposes


for features,label in training_data:
	X.append(features)
	y.append(label)

X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,1)  #1 meaning the gray scale

pickle_out = open("X.pickle","wb")
pickle.dump(X,pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y,pickle_out)
pickle_out.close()