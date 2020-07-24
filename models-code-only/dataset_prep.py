import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Input
from tensorflow.keras.layers import Reshape, UpSampling2D, InputLayer, Lambda, ZeroPadding2D
from tensorflow.keras.layers import Cropping2D, Conv2DTranspose, BatchNormalization

from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse, binary_crossentropy



dataset_train = '/home/leo/deeplearning/depth_prediction/train'
ground_truth = '/home/leo/deeplearning/depth_prediction/ground_truth'
width = 320
height = 180

width = int(width/2)
height = int(height/2)

def create_training_data():
	train_data = []
	depth_data = []

	for img in sorted(os.listdir(dataset_train)):
		img_array = cv2.imread(os.path.join(dataset_train,img))
		img_array = cv2.resize(img_array, (width, height))
		train_data.append(img_array)

		# plt.imshow(img_array, cmap='gray')  # graph it
		# plt.show()  # display!
		# break


	for img in sorted(os.listdir(ground_truth)):
		img_array = cv2.imread(os.path.join(ground_truth,img) ,cv2.IMREAD_GRAYSCALE)
		img_array = cv2.resize(img_array, (width, height))
		depth_data.append(img_array)
		# plt.imshow(img_array, cmap='gray')  # graph it
		# plt.show()  # display!
		# break
	train_data = np.array(train_data).reshape(-1, height, width, 3)
	depth_data = np.array(depth_data).reshape(-1, height, width, 1)
	train_data = np.float32(train_data / 255.)
	depth_data = np.float32(depth_data / 255.)

	# train_data, RGB_validation, depth_data, depth_validation = train_test_split(train_data, depth_data, test_size=0.15)

	pickle_out = open("train_data.pickle","wb")
	pickle.dump(train_data, pickle_out)
	pickle_out.close()

	pickle_out = open("depth_data.pickle","wb")
	pickle.dump(depth_data, pickle_out)
	pickle_out.close()

	# pickle_out = open("RGB_validation.pickle","wb")
	# pickle.dump(RGB_validation, pickle_out)
	# pickle_out.close()

	# pickle_out = open("depth_validation.pickle","wb")
	# pickle.dump(depth_validation, pickle_out)
	# pickle_out.close()


create_training_data()

def select_batch(train_data, depth_data, b_size):
	index = np.random.choice(train_data.shape[0], b_size, replace=False)
	index = sorted(index)
	params1 = tf.constant(train_data)
	params2 = tf.constant(depth_data)
	indices = tf.constant(index)
	test_sample = tf.gather(params1, indices)
	depth_sample = tf.gather(params2, indices)
	return test_sample, depth_sample


pickle_in = open("train_data.pickle","rb")
train_data = pickle.load(pickle_in)

pickle_in = open("depth_data.pickle","rb")
depth_data = pickle.load(pickle_in)

print(train_data.shape)
print(depth_data.shape)

RGB_validation, depth_validation  = select_batch(train_data, depth_data, 300)
pickle_out = open("RGB_validation.pickle","wb")
pickle.dump(RGB_validation, pickle_out)
pickle_out.close()

pickle_out = open("depth_validation.pickle","wb")
pickle.dump(depth_validation, pickle_out)
pickle_out.close()