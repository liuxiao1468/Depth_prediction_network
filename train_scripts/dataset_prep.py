import os
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle
import glob
import random



dataset_train = '/home/leo/deeplearning/depth_prediction/train'
ground_truth = '/home/leo/deeplearning/depth_prediction/ground_truth'
width = 320
height = 180

# width = int(width/2)
# height = int(height/2)

class get_dataset():

	def select_batch(self, train_data_path, depth_data_path, batch_size):
		DIR = train_data_path
		N = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
		train = sorted(glob.glob(train_data_path + "//*"))
		depth = sorted(glob.glob(depth_data_path + "//*"))
		select = random.sample(range(0, N), batch_size)
		train_data = []
		depth_data = []
		for idx in select:
			img_array = cv2.imread(train[idx])
			img_array = cv2.resize(img_array, (width, height))
			train_data.append(img_array)

			img_array = cv2.imread(depth[idx],cv2.IMREAD_GRAYSCALE)
			img_array = cv2.resize(img_array, (width, height))
			img_array = (img_array/255)*6
			img_array = np.where(img_array>1.0, 1.0, img_array)
			depth_data.append(img_array)

		train_data = np.array(train_data).reshape(-1, height, width, 3)
		depth_data = np.array(depth_data).reshape(-1, height, width, 1)
		train_data = np.float32(train_data / 255.)
		depth_data = np.float32(depth_data)

		return train_data, depth_data



	def train_generator(self, train_data_path, depth_data_path, batch_size):
		while True:
			DIR = train_data_path
			N = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
			train = sorted(glob.glob(train_data_path + "//*"))
			depth = sorted(glob.glob(depth_data_path + "//*"))
			select = random.sample(range(0, N), batch_size)
			train_data = []
			depth_data = []
			for idx in select:
				img_array = cv2.imread(train[idx])
				img_array = cv2.resize(img_array, (width, height))
				train_data.append(img_array)

				img_array = cv2.imread(depth[idx],cv2.IMREAD_GRAYSCALE)
				img_array = cv2.resize(img_array, (width, height))
				img_array = (img_array/255)*6
				img_array = np.where(img_array>1.0, 1.0, img_array)
				depth_data.append(img_array)

			train_data = np.array(train_data).reshape(-1, height, width, 3)
			depth_data = np.array(depth_data).reshape(-1, height, width, 1)
			train_data = np.float32(train_data / 255.)
			depth_data = np.float32(depth_data)

			yield train_data, depth_data

	def validation_generator(self, train_data_path, depth_data_path, batch_size):
		while True:
			DIR = train_data_path
			N = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
			train = sorted(glob.glob(train_data_path + "//*"))
			depth = sorted(glob.glob(depth_data_path + "//*"))
			select = random.sample(range(0, N), batch_size)
			train_data = []
			depth_data = []
			for idx in select:
				img_array = cv2.imread(train[idx])
				img_array = cv2.resize(img_array, (width, height))
				train_data.append(img_array)

				img_array = cv2.imread(depth[idx],cv2.IMREAD_GRAYSCALE)
				img_array = cv2.resize(img_array, (width, height))
				img_array = (img_array/255)*6
				img_array = np.where(img_array>1.0, 1.0, img_array)
				depth_data.append(img_array)

			train_data = np.array(train_data).reshape(-1, height, width, 3)
			depth_data = np.array(depth_data).reshape(-1, height, width, 1)
			train_data = np.float32(train_data / 255.)
			depth_data = np.float32(depth_data)

			yield train_data, depth_data