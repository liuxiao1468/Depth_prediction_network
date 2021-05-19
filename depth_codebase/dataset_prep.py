import pandas as pd
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.layers import Reshape, UpSampling2D, InputLayer, Lambda, ZeroPadding2D, AveragePooling2D
import os
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle
import glob
import random
from PIL import Image
import csv




width = 160
height = 90

# width = int(width/2)
# height = int(height/2)

class get_dataset():

	def transform_ground_truth(self, depth_data):
	    gt_transformed = []

	    gt_transformed.append(AveragePooling2D(pool_size=(4, 4), strides=None, padding='same')(depth_data))
	    gt_transformed.append(AveragePooling2D(pool_size=(2, 2), strides=None, padding='same')(depth_data))
	    gt_transformed.append(depth_data)

	    return gt_transformed

	def select_batch(self, batch_size):
		dataset_v1 = []
		with open(r'./Dataset_v1/files'+'.csv','rt')as f:
		    data = csv.reader(f)
		    for row in data:
		        dataset_v1.append(row)
		# dataset_v2 = dataset_v2[0:7271]
		dataset_v1 = dataset_v1[7271:]
		dataset_v2 = []
		with open(r'./Dataset_v2/files'+'.csv','rt')as f:
		    data = csv.reader(f)
		    for row in data:
		        dataset_v2.append(row)
		# dataset_v1 = dataset_v1[0:24000]
		dataset_v2 = dataset_v2[25500:]
		merge = []
		for i in range (len(dataset_v1)):
		    string_1 = ['./Dataset_v1/', dataset_v1[i][0], dataset_v1[i][1], dataset_v1[i][2] ]
		    merge.append(string_1)
		for i in range (len(dataset_v2)):
			string_1 = ['./Dataset_v2/', dataset_v2[i][0], dataset_v2[i][1], dataset_v2[i][2] ]
			merge.append(string_1)

		N = len(merge)
		select = random.sample(range(0, N), batch_size)

		train_data = []
		depth_data = []
		for idx in select:
			img_array = cv2.imread(merge[idx][0]+merge[idx][1]+'/'+merge[idx][2])
			img_array = cv2.resize(img_array, (width, height))
			train_data.append(img_array)

			img_array = cv2.imread(merge[idx][0]+merge[idx][1]+'/'+merge[idx][3],cv2.IMREAD_GRAYSCALE)
			img_array = cv2.resize(img_array, (width, height))
			img_array = (img_array/255.0)*6
			img_array = np.where(img_array>1.0, 1.0, img_array)
			depth_data.append(img_array)

		train_data = np.array(train_data).reshape(-1, height, width, 3)
		depth_data = np.array(depth_data).reshape(-1, height, width, 1)
		train_data = np.float32(train_data / 255.)
		depth_data = np.float32(depth_data)
		depth_data = self.transform_ground_truth(depth_data)

		return train_data, depth_data

	# def select_batch_mask(self):
	# 	dataset_v2 = []
	# 	with open(r'/datasets/industrious_v2/files'+'.csv','rt')as f:
	# 	    data = csv.reader(f)
	# 	    for row in data:
	# 	        dataset_v2.append(row)
	# 	# delete scene 001 and save it for testing
	# 	del dataset_v2[269:613]
	# 	merge = []
	# 	for i in range (len(dataset_v2)):
	# 	    string_1 = ['/datasets/industrious_v2/', dataset_v2[i][0], dataset_v2[i][1], dataset_v2[i][2], dataset_v2[i][3] ]
	# 	    merge.append(string_1)

	# 	N = len(merge)
	# 	idx =  random.randint(0, N-2)
	# 	while merge[idx][1]!= merge[idx+1][1]:
	# 		idx = random.randint(0, N-1)

	# 	train_data = []
	# 	depth_data = []
	# 	mask_data = []
	# 	mask_data_final = []

	# 	for idx in range (2):

	# 		img_array = cv2.imread(merge[idx][0]+merge[idx][1]+'/'+merge[idx][2].split(' ')[1])
	# 		img_array = cv2.resize(img_array, (width, height))
	# 		train_data.append(img_array)

	# 		img_array = cv2.imread(merge[idx][0]+merge[idx][1]+'/'+merge[idx][3].split(' ')[1],cv2.IMREAD_GRAYSCALE)
	# 		img_array = cv2.resize(img_array, (width, height))
	# 		img_array = (img_array/255.)

	# 		depth_data.append(img_array)

	# 		img_array = cv2.imread(merge[idx][0]+merge[idx][1]+'/'+merge[idx][4].split(' ')[1],cv2.IMREAD_GRAYSCALE)
	# 		img_array = cv2.resize(img_array, (width, height))
	# 		img_array = (img_array/255.)
	# 		mask_data.append(img_array)
	# 	M_1 = 1.0-mask_data[0]
	# 	M_2 = 1.0-mask_data[1]

	# 	M = M_1+M_2
	# 	M = np.where(M==1.0, 0.0, M)
	# 	mask_data_0 = np.where(M>=1.5, 1.0, M)

	# 	mask_data_1 = np.where(M==0.0, 1.0, M)
	# 	mask_data_1 = np.where(M==1.0, 0.0, M)
	# 	mask_data_final.append(mask_data_0)
	# 	mask_data_final.append(mask_data_1)


	# 	train_data = np.array(train_data).reshape(-1, height, width, 3)
	# 	depth_data = np.array(depth_data).reshape(-1, height, width, 1)
	# 	mask_data_final = np.array(mask_data_final).reshape(-1, height, width, 1)
	# 	train_data = np.float32(train_data / 255.)
	# 	depth_data = np.float32(depth_data)
	# 	mask_data_final = np.float32(mask_data_final)
	# 	# depth_data = self.transform_ground_truth(depth_data)

	# 	return train_data, depth_data, mask_data_final



	def train_generator(self, batch_size):
		dataset_v1 = []
		with open(r'./Dataset_v1/files'+'.csv','rt')as f:
		    data = csv.reader(f)
		    for row in data:
		        dataset_v1.append(row)
		# dataset_v2 = dataset_v2[0:7271]
		dataset_v1 = dataset_v1[7271:]
		dataset_v2 = []
		with open(r'./Dataset_v2/files'+'.csv','rt')as f:
		    data = csv.reader(f)
		    for row in data:
		        dataset_v2.append(row)
		# dataset_v1 = dataset_v1[0:24000]
		dataset_v2 = dataset_v2[25500:]
		merge = []
		for i in range (len(dataset_v1)):
		    string_1 = ['./Dataset_v1/', dataset_v1[i][0], dataset_v1[i][1], dataset_v1[i][2] ]
		    merge.append(string_1)
		for i in range (len(dataset_v2)):
			string_1 = ['./Dataset_v2/', dataset_v2[i][0], dataset_v2[i][1], dataset_v2[i][2] ]
			merge.append(string_1)

		while True:
			N = len(merge)
			select = random.sample(range(0, N), batch_size)

			train_data = []
			depth_data = []
			for idx in select:
				img_array = cv2.imread(merge[idx][0]+merge[idx][1]+'/'+merge[idx][2])
				img_array = cv2.resize(img_array, (width, height))
				train_data.append(img_array)

				img_array = cv2.imread(merge[idx][0]+merge[idx][1]+'/'+merge[idx][3],cv2.IMREAD_GRAYSCALE)
				img_array = cv2.resize(img_array, (width, height))
				img_array = (img_array/255.0)*6
				img_array = np.where(img_array>1.0, 1.0, img_array)
				depth_data.append(img_array)

			train_data = np.array(train_data).reshape(-1, height, width, 3)
			depth_data = np.array(depth_data).reshape(-1, height, width, 1)
			train_data = np.float32(train_data / 255.)
			depth_data = np.float32(depth_data)
			depth_data = self.transform_ground_truth(depth_data)

			yield (train_data, depth_data)

	def validation_generator(self, batch_size):
		dataset_v1 = []
		with open(r'./Dataset_v1/files'+'.csv','rt')as f:
		    data = csv.reader(f)
		    for row in data:
		        dataset_v1.append(row)
		# dataset_v2 = dataset_v2[0:7271]
		dataset_v1 = dataset_v1[7271:]
		dataset_v2 = []
		with open(r'./Dataset_v2/files'+'.csv','rt')as f:
		    data = csv.reader(f)
		    for row in data:
		        dataset_v2.append(row)
		# dataset_v1 = dataset_v1[0:24000]
		dataset_v2 = dataset_v2[25500:]
		merge = []
		for i in range (len(dataset_v1)):
		    string_1 = ['./Dataset_v1/', dataset_v1[i][0], dataset_v1[i][1], dataset_v1[i][2] ]
		    merge.append(string_1)
		for i in range (len(dataset_v2)):
			string_1 = ['./Dataset_v2/', dataset_v2[i][0], dataset_v2[i][1], dataset_v2[i][2] ]
			merge.append(string_1)

		while True:
			N = len(merge)
			select = random.sample(range(0, N), batch_size)

			train_data = []
			depth_data = []
			for idx in select:
				img_array = cv2.imread(merge[idx][0]+merge[idx][1]+'/'+merge[idx][2])
				img_array = cv2.resize(img_array, (width, height))
				train_data.append(img_array)

				img_array = cv2.imread(merge[idx][0]+merge[idx][1]+'/'+merge[idx][3],cv2.IMREAD_GRAYSCALE)
				img_array = cv2.resize(img_array, (width, height))
				img_array = (img_array/255.0)*6
				img_array = np.where(img_array>1.0, 1.0, img_array)
				depth_data.append(img_array)

			train_data = np.array(train_data).reshape(-1, height, width, 3)
			depth_data = np.array(depth_data).reshape(-1, height, width, 1)
			train_data = np.float32(train_data / 255.)
			depth_data = np.float32(depth_data)
			depth_data = self.transform_ground_truth(depth_data)

			yield (train_data, depth_data)

# class get_NYU_dataset():

# 	def transform_ground_truth(self, depth_data):
# 	    # 6, ?, x, x, 1
# 	    # [(?, 6, 12, 1), (?, 12, 24, 1), (?, 24, 48, 1), (?, 48, 96, 1), (?, 96, 192, 1), (?, 192, 384, 1)]
# 	    gt_transformed = []
# 	    # gt_transformed.append(AveragePooling2D(pool_size=(64, 64), strides=None, padding='same')(depth_data))
# 	    gt_transformed.append(AveragePooling2D(pool_size=(32, 32), strides=None, padding='same')(depth_data))
# 	    gt_transformed.append(AveragePooling2D(pool_size=(16, 16), strides=None, padding='same')(depth_data))
# 	    gt_transformed.append(AveragePooling2D(pool_size=(8, 8), strides=None, padding='same')(depth_data))
# 	    gt_transformed.append(AveragePooling2D(pool_size=(4, 4), strides=None, padding='same')(depth_data))
# 	    gt_transformed.append(AveragePooling2D(pool_size=(2, 2), strides=None, padding='same')(depth_data))
# 	    gt_transformed.append(depth_data)

# 	    return gt_transformed

# 	def select_batch(self, train_data_path, batch_size):
# 	    with open(f'/datasets/nyu_depth_v2/{train_data_path}', 'r') as f:
# 	        filenames = f.readlines()
# 	    N = len(filenames)
# 	    select = random.sample(range(0, N), batch_size)
# 	    train_data = []
# 	    depth_data = []
# 	    for idx in select:
# 	        img_path = '/datasets/nyu_depth_v2/sync/'+str(filenames[idx].split()[0])
# 	        img_array = Image.open(os.path.join(img_path))
# 	        img_array = img_array.crop((43, 45, 608, 472))
# 	        img_array = np.asarray(img_array, dtype=np.float32)
# 	        img_array = cv2.resize(img_array, (width, height))
# 	        train_data.append(img_array)

# 	        depth_path = '/datasets/nyu_depth_v2/sync/'+str(filenames[idx].split()[1])
# 	        depth_gt = Image.open(os.path.join(depth_path))
# 	        depth_gt = depth_gt.crop((43, 45, 608, 472))
# 	        depth_gt = np.asarray(depth_gt, dtype=np.float32) / 1000.0
# 	        depth_gt = np.clip(depth_gt, 0.1, 10)
# 	        depth_gt = depth_gt/10
# 	        depth_gt = cv2.resize(depth_gt, (width, height))
# 	        depth_data.append(depth_gt)

# 	    train_data = np.array(train_data).reshape(-1, height, width, 3)
# 	    depth_data = np.array(depth_data).reshape(-1, height, width, 1)
# 	    train_data = np.float32(train_data / 255.)
# 	    depth_data = np.float32(depth_data)
# 	    depth_data = self.transform_ground_truth(depth_data)

# 	    return (train_data, depth_data)



# 	def train_generator(self, train_data_path, batch_size):
# 	    with open(f'/datasets/nyu_depth_v2/{train_data_path}', 'r') as f:
# 	        filenames = f.readlines()
# 	    N = len(filenames)
# 	    while True:
# 	        select = random.sample(range(0, N), batch_size)
# 	        train_data = []
# 	        depth_data = []
# 	        for idx in select:
# 	            img_path = '/datasets/nyu_depth_v2/sync/'+str(filenames[idx].split()[0])
# 	            img_array = Image.open(os.path.join(img_path))
# 	            img_array = img_array.crop((43, 45, 608, 472))
# 	            img_array = np.asarray(img_array, dtype=np.float32)
# 	            img_array = cv2.resize(img_array, (width, height))
# 	            train_data.append(img_array)

# 	            depth_path = '/datasets/nyu_depth_v2/sync/'+str(filenames[idx].split()[1])
# 	            depth_gt = Image.open(os.path.join(depth_path))
# 	            depth_gt = depth_gt.crop((43, 45, 608, 472))
# 	            depth_gt = np.asarray(depth_gt, dtype=np.float32) / 1000.0
# 	            depth_gt = np.clip(depth_gt, 0.1, 10)
# 	            depth_gt = depth_gt/10
# 	            depth_gt = cv2.resize(depth_gt, (width, height))
# 	            depth_data.append(depth_gt)

# 	        train_data = np.array(train_data).reshape(-1, height, width, 3)
# 	        depth_data = np.array(depth_data).reshape(-1, height, width, 1)
# 	        train_data = np.float32(train_data / 255.)
# 	        depth_data = np.float32(depth_data)
# 	        depth_data = self.transform_ground_truth(depth_data)

# 	        yield (train_data, depth_data)

# 	def validation_generator(self, train_data_path,batch_size):
# 	    with open(f'/datasets/nyu_depth_v2/{train_data_path}', 'r') as f:
# 	        filenames = f.readlines()
# 	    N = len(filenames)
# 	    while True:
# 	        select = random.sample(range(0, N), batch_size)
# 	        train_data = []
# 	        depth_data = []
# 	        for idx in select:
# 	            img_path = '/datasets/nyu_depth_v2/official_splits/test/'+str(filenames[idx].split()[0])
# 	            img_array = Image.open(os.path.join(img_path))
# 	            img_array = img_array.crop((43, 45, 608, 472))
# 	            img_array = np.asarray(img_array, dtype=np.float32)
# 	            img_array = cv2.resize(img_array, (width, height))
# 	            train_data.append(img_array)

# 	            depth_path = '/datasets/nyu_depth_v2/official_splits/test/'+str(filenames[idx].split()[1])
# 	            depth_gt = Image.open(os.path.join(depth_path))
# 	            depth_gt = depth_gt.crop((43, 45, 608, 472))
# 	            depth_gt = np.asarray(depth_gt, dtype=np.float32) / 1000.0
# 	            depth_gt = np.clip(depth_gt, 0.1, 10)
# 	            depth_gt = depth_gt/10
# 	            depth_gt = cv2.resize(depth_gt, (width, height))
# 	            depth_data.append(depth_gt)

# 	        train_data = np.array(train_data).reshape(-1, height, width, 3)
# 	        depth_data = np.array(depth_data).reshape(-1, height, width, 1)
# 	        train_data = np.float32(train_data / 255.)
# 	        depth_data = np.float32(depth_data)
# 	        depth_data = self.transform_ground_truth(depth_data)

# 	        yield (train_data, depth_data)

# class get_dataset_v3():

# 	def transform_ground_truth(self, depth_data):
# 	    gt_transformed = []
# 	    # gt_transformed.append(AveragePooling2D(pool_size=(64, 64), strides=None, padding='same')(depth_data))
# 	    # gt_transformed.append(AveragePooling2D(pool_size=(32, 32), strides=None, padding='same')(depth_data))
# 	    gt_transformed.append(AveragePooling2D(pool_size=(16, 16), strides=None, padding='same')(depth_data))
# 	    gt_transformed.append(AveragePooling2D(pool_size=(8, 8), strides=None, padding='same')(depth_data))
# 	    gt_transformed.append(AveragePooling2D(pool_size=(4, 4), strides=None, padding='same')(depth_data))
# 	    gt_transformed.append(AveragePooling2D(pool_size=(2, 2), strides=None, padding='same')(depth_data))
# 	    gt_transformed.append(depth_data)

# 	    return gt_transformed

# 	def select_batch(self, batch_size, time_step):
# 		dataset_v2 = []
# 		with open(r'/datasets/industrious_v2/files'+'.csv','rt')as f:
# 		    data = csv.reader(f)
# 		    for row in data:
# 		        dataset_v2.append(row)
# 		# delete scene 001 and save it for testing
# 		del dataset_v2[269:613]
# 		dataset_v3 = []
# 		with open(r'/datasets/industrious_v3/files'+'.csv','rt')as f:
# 		    data = csv.reader(f)
# 		    for row in data:
# 		        dataset_v3.append(row)
# 		merge = []
# 		for i in range (len(dataset_v2)):
# 		    string_1 = ['/datasets/industrious_v2/', dataset_v2[i][0], dataset_v2[i][1], dataset_v2[i][2] ]
# 		    merge.append(string_1)
# 		for i in range (len(dataset_v3)):
# 			string_1 = ['/datasets/industrious_v3/', dataset_v3[i][0], dataset_v3[i][1], dataset_v3[i][2] ]
# 			merge.append(string_1)


# 		N = len(merge)
# 		train_data = []
# 		depth_data = []

# 		# select some video sequence and make sure they are valid 
# 		check = 1
# 		while check>=1:
# 			select = random.sample(range(0, N-time_step), batch_size)
# 			check = check-1
# 			for i in range (len(select)):
# 				try:
# 					if merge[select[i]][1] != merge[select[i]+time_step][1]:
# 						check = check +1
# 				except:
# 					check = check +1

# 		# select rgb sequence, gt, and reshape into desired tensor shape
# 		for idx in select:
# 			for i in range (time_step):
# 				idx_temp = idx + i
# 				img_array = cv2.imread(merge[idx_temp][0]+merge[idx_temp][1]+'/'+merge[idx_temp][2].split(' ')[1])
# 				img_array = cv2.resize(img_array, (width, height))
# 				train_data.append(img_array)

# 			img_array = cv2.imread(merge[idx][0]+merge[idx][1]+'/'+merge[idx][3].split(' ')[1],cv2.IMREAD_GRAYSCALE)
# 			img_array = cv2.resize(img_array, (width, height))
# 			img_array = (img_array/255.)
# 			depth_data.append(img_array)

# 		train_data = np.array(train_data).reshape(-1, height, width, 3)
# 		depth_data = np.array(depth_data).reshape(-1, height, width, 1)
# 		train_data = np.float32(train_data / 255.)
# 		depth_data = np.float32(depth_data)
# 		depth_data = self.transform_ground_truth(depth_data)

# 		train_data = np.array(train_data).reshape(-1, time_step, height, width, 3)

# 		return train_data, depth_data


# 	def train_generator(self, batch_size,time_step):
# 		dataset_v2 = []
# 		with open(r'/datasets/industrious_v2/files'+'.csv','rt')as f:
# 		    data = csv.reader(f)
# 		    for row in data:
# 		        dataset_v2.append(row)
# 		# delete scene 001 and save it for testing
# 		del dataset_v2[269:613]
# 		dataset_v3 = []
# 		with open(r'/datasets/industrious_v3/files'+'.csv','rt')as f:
# 		    data = csv.reader(f)
# 		    for row in data:
# 		        dataset_v3.append(row)
# 		merge = []
# 		for i in range (len(dataset_v2)):
# 		    string_1 = ['/datasets/industrious_v2/', dataset_v2[i][0], dataset_v2[i][1], dataset_v2[i][2] ]
# 		    merge.append(string_1)
# 		for i in range (len(dataset_v3)):
# 			string_1 = ['/datasets/industrious_v3/', dataset_v3[i][0], dataset_v3[i][1], dataset_v3[i][2] ]
# 			merge.append(string_1)

# 		while True:
# 			N = len(merge)
# 			train_data = []
# 			depth_data = []

# 			# select some video sequence and make sure they are valid 
# 			check = 1
# 			while check>=1:
# 				select = random.sample(range(0, N-time_step), batch_size)
# 				check = check-1
# 				for i in range (len(select)):
# 					try:
# 						if merge[select[i]][1] != merge[select[i]+time_step][1]:
# 							check = check +1
# 					except:
# 						check = check +1
# 			# select rgb sequence, gt, and reshape into desired tensor shape
# 			for idx in select:
# 				for i in range (time_step):
# 					idx_temp = idx + i
# 					img_array = cv2.imread(merge[idx_temp][0]+merge[idx_temp][1]+'/'+merge[idx_temp][2].split(' ')[1])
# 					img_array = cv2.resize(img_array, (width, height))
# 					train_data.append(img_array)

# 				img_array = cv2.imread(merge[idx][0]+merge[idx][1]+'/'+merge[idx][3].split(' ')[1],cv2.IMREAD_GRAYSCALE)
# 				img_array = cv2.resize(img_array, (width, height))
# 				img_array = (img_array/255.)
# 				depth_data.append(img_array)
# 			# reshape train and depth data into tensor shape
# 			train_data = np.array(train_data).reshape(-1, height, width, 3)
# 			depth_data = np.array(depth_data).reshape(-1, height, width, 1)
# 			train_data = np.float32(train_data / 255.)
# 			depth_data = np.float32(depth_data)
# 			depth_data = self.transform_ground_truth(depth_data)

# 			train_data = np.array(train_data).reshape(-1, time_step, height, width, 3)

# 			yield (train_data, depth_data)

# 	def validation_generator(self, batch_size,time_step):
# 		dataset_v2 = []
# 		with open(r'/datasets/industrious_v2/files'+'.csv','rt')as f:
# 		    data = csv.reader(f)
# 		    for row in data:
# 		        dataset_v2.append(row)
# 		# delete scene 001 and save it for testing
# 		del dataset_v2[269:613]
# 		dataset_v3 = []
# 		with open(r'/datasets/industrious_v3/files'+'.csv','rt')as f:
# 		    data = csv.reader(f)
# 		    for row in data:
# 		        dataset_v3.append(row)
# 		merge = []
# 		for i in range (len(dataset_v2)):
# 		    string_1 = ['/datasets/industrious_v2/', dataset_v2[i][0], dataset_v2[i][1], dataset_v2[i][2] ]
# 		    merge.append(string_1)
# 		for i in range (len(dataset_v3)):
# 			string_1 = ['/datasets/industrious_v3/', dataset_v3[i][0], dataset_v3[i][1], dataset_v3[i][2] ]
# 			merge.append(string_1)

# 		while True:
# 			N = len(merge)
# 			train_data = []
# 			depth_data = []

# 			# select some video sequence and make sure they are valid 
# 			check = 1
# 			while check>=1:
# 				select = random.sample(range(0, N-time_step), batch_size)
# 				check = check-1
# 				for i in range (len(select)):
# 					try:
# 						if merge[select[i]][1] != merge[select[i]+time_step][1]:
# 							check = check +1
# 					except:
# 						check = check +1
# 			# select rgb sequence, gt, and reshape into desired tensor shape
# 			for idx in select:
# 				for i in range (time_step):
# 					idx_temp = idx + i
# 					img_array = cv2.imread(merge[idx_temp][0]+merge[idx_temp][1]+'/'+merge[idx_temp][2].split(' ')[1])
# 					img_array = cv2.resize(img_array, (width, height))
# 					train_data.append(img_array)

# 				img_array = cv2.imread(merge[idx][0]+merge[idx][1]+'/'+merge[idx][3].split(' ')[1],cv2.IMREAD_GRAYSCALE)
# 				img_array = cv2.resize(img_array, (width, height))
# 				img_array = (img_array/255.)
# 				depth_data.append(img_array)
# 			# reshape train and depth data into tensor shape
# 			train_data = np.array(train_data).reshape(-1, height, width, 3)
# 			depth_data = np.array(depth_data).reshape(-1, height, width, 1)
# 			train_data = np.float32(train_data / 255.)
# 			depth_data = np.float32(depth_data)
# 			depth_data = self.transform_ground_truth(depth_data)

# 			train_data = np.array(train_data).reshape(-1, time_step, height, width, 3)

# 			yield (train_data, depth_data)