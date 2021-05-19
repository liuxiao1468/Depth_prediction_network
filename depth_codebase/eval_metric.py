import pandas as pd
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.layers import Reshape, UpSampling2D, InputLayer, Lambda, ZeroPadding2D, AveragePooling2D
from tensorflow.keras import backend as K
import dataset_prep
import depth_prediction_net
import loss
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np
import csv
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import numpy as np
from numpy import linalg as LA
from statistics import mean
import glob 
import random
from PIL import Image
import os

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


width = 160
height = 90
batch_size = 150

get_loss = loss.get_loss()

def select_batch_mask():
    dataset_v2 = []
    with open(r'./Dataset_mask_v1/files'+'.csv','rt')as f:
        data = csv.reader(f)
        for row in data:
            dataset_v2.append(row)
    dataset_v2 =  dataset_v2[0:226]
    merge = []
    for i in range (len(dataset_v2)):
        string_1 = ['./Dataset_mask_v1/', dataset_v2[i][0], dataset_v2[i][1], dataset_v2[i][2], dataset_v2[i][3] ]
        merge.append(string_1)

    N = len(merge)

    train_data = []
    depth_data = []
    mask_data = []


    for idx in range (N):

        img_array = cv2.imread(merge[idx][0]+merge[idx][1]+'/'+merge[idx][2])
        img_array = cv2.resize(img_array, (width, height))
        train_data.append(img_array)

        img_array = cv2.imread(merge[idx][0]+merge[idx][1]+'/'+merge[idx][3],cv2.IMREAD_GRAYSCALE)
        img_array = cv2.resize(img_array, (width, height))
        img_array = (img_array/255.0)*6
        img_array = np.where(img_array>1.0, 1.0, img_array)

        depth_data.append(img_array)

        img_array = cv2.imread(merge[idx][0]+merge[idx][1]+'/'+merge[idx][4],cv2.IMREAD_GRAYSCALE)
        img_array = cv2.resize(img_array, (width, height))
        img_array = np.where(img_array>0.0, 255.0, img_array)
        img_array = (img_array/255.)
        img_array = 1.0 - img_array
        mask_data.append(img_array)

    train_data = np.array(train_data).reshape(-1, height, width, 3)
    depth_data = np.array(depth_data).reshape(-1, height, width, 1)
    mask_data = np.array(mask_data).reshape(-1, height, width, 1)
    train_data = np.float32(train_data / 255.)
    depth_data = np.float32(depth_data)
    mask_data = np.float32(mask_data)
    # depth_data = self.transform_ground_truth(depth_data)

    return train_data, depth_data, mask_data

def select_batch(batch_size):
    dataset_v2 = []
    with open(r'./Dataset_v2/files'+'.csv','rt')as f:
        data = csv.reader(f)
        for row in data:
            dataset_v2.append(row)
    dataset_v2 = dataset_v2[7271:]
    dataset_v1 = []
    with open(r'./Dataset_v1/files'+'.csv','rt')as f:
        data = csv.reader(f)
        for row in data:
            dataset_v1.append(row)
    dataset_v1 = dataset_v1[24000:]
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
        array_sum = np.sum(img_array)
        depth_data.append(img_array)

    train_data = np.array(train_data).reshape(-1, height, width, 3)
    depth_data = np.array(depth_data).reshape(-1, height, width, 1)
    train_data = np.float32(train_data / 255.)
    depth_data = np.float32(depth_data)
    # depth_data = self.transform_ground_truth(depth_data)

    return train_data, depth_data


def accuracy_metric(gt, pred):
    # Relative errors
    temp = np.where(gt==0, 0, np.abs(gt - pred)/ gt)
    # abs_rel = np.mean(np.abs(gt - pred) / gt)
    abs_rel = np.mean(temp)

    temp = np.where(gt==0, 0, ((gt - pred) ** 2)/ gt)
    # sq_rel = np.mean(((gt - pred) ** 2) / gt)
    sq_rel = np.mean(temp)

    rmse = sqrt(mean_squared_error(gt, pred))
    rms_log = sqrt(mean_squared_log_error(gt, pred))


    temp_1 = np.where(pred==0, 0, gt / pred)
    temp_2 = np.where(gt==0, 0, pred / gt)

    # thresh = np.maximum((gt / pred), (pred / gt))
    thresh = np.maximum(temp_1, temp_2)
    d1_temp = (thresh < 1.25).mean()
    d2_temp = (thresh < 1.25 ** 2).mean()
    d3_temp = (thresh < 1.25 ** 3).mean()

    return abs_rel,sq_rel,rmse,rms_log, d1_temp, d2_temp, d3_temp


# train_data, depth_data = select_batch(batch_size)

train_data, depth_data, mask_data = select_batch_mask()



# model_path = './saved_model/depth_model_v4'

# model_path = sorted(glob.glob(model_path + "//*"))

# final_result = []

# for j in range (len(model_path)):

#     # # Recreate the exact same model, including its weights and the optimizer
#     model = tf.keras.models.load_model(model_path[j],custom_objects={'autoencoder_loss': get_loss.autoencoder_loss})
#     # model = tf.keras.models.load_model('/tfdepth/model_HD/saved_model/ft_parameter/few_layer_TC_'+str((j+1)*5)+'.h5')

#     abs_rel = []
#     sq_rel = []
#     rmse = []
#     rmse_log = []
#     d1 = []
#     d2 = []
#     d3 = []

#     tc = []

#     for i in range (0, int(len(train_data)/2), 2):
#         depth1 = tf.reshape(depth_data[i], [1, height, width, 1])
#         depth2 = tf.reshape(depth_data[i+1], [1, height, width, 1])
#         rgb1 = tf.reshape(train_data[i], [1, height, width, 3])
#         rgb2 = tf.reshape(train_data[i+1], [1, height, width, 3])

#         para = 1
#         ex = 0


#         gt1 = depth1[0, :, :, 0]*para+ex
#         gt2 = depth2[0, :, :, 0]*para+ex
#         _, _, prediction1 = model.predict(rgb1)
#         _, _, prediction2 = model.predict(rgb2)
#         # prediction1 = model.predict(rgb1)
#         # prediction2 = model.predict(rgb2)
#         pred1 = prediction1[0, :, :, 0]*para+ex
#         pred2 = prediction2[0, :, :, 0]*para+ex

#         # # for temporal consistency
#         # mask1 = tf.reshape(mask_data[i], [1, height, width, 1])
#         # mask2 = tf.reshape(mask_data[i+1], [1, height, width, 1])
#         # M1 = mask1[0, :, :, 0]
#         # M2 = mask2[0, :, :, 0]
#         # M = M1+M2
#         # M = np.where(M==1.0, 0.0, M)
#         # M = np.where(M>=1.5, 1.0, M)

#         # D1 = np.multiply(pred1,M)
#         # D2 = np.multiply(pred2,M)
#         # tc_temp = LA.norm(np.abs(np.subtract(D1,D2)), 2)/np.sum(M)
#         # tc.append(tc_temp)
#         # print(gt1)
#         # array_sum = np.sum(gt1)
#         # array_has_nan = np.isnan(array_sum)
#         # if array_has_nan == True:
#         #     print("here-------------------------------------")
#         # print('------------')
#         # print(pred1, pred1.dtype)

#         abs_rel_temp,sq_rel_remp,rmse_temp,rms_log_remp, d1_temp, d2_temp, d3_temp = accuracy_metric(gt1, pred1)
#         abs_rel.append(abs_rel_temp)
#         sq_rel.append(sq_rel_remp)
#         rmse.append(rmse_temp)
#         rmse_log.append(rms_log_remp)
#         d1.append(d1_temp)
#         d2.append(d2_temp)
#         d3.append(d3_temp)

#         abs_rel_temp,sq_rel_remp,rmse_temp,rms_log_remp, d1_temp, d2_temp, d3_temp = accuracy_metric(gt2, pred2)
#         abs_rel.append(abs_rel_temp)
#         sq_rel.append(sq_rel_remp)
#         rmse.append(rmse_temp)
#         rmse_log.append(rms_log_remp)
#         d1.append(d1_temp)
#         d2.append(d2_temp)

#     result = []
#     result.append(mean(abs_rel))
#     result.append(mean(sq_rel))
#     result.append(mean(rmse))
#     result.append(mean(rmse_log))
#     result.append(mean(d1))
#     result.append(mean(d2))
#     result.append(mean(d3))

#     # result.append(mean(tc))

#     print(model_path[j] , result)
#     final_result.append(result)
#     K.clear_session()

# with open('./eval_result/depth_model_v4.txt', 'w') as f:
#     for item in final_result:
#         f.write("%s\n" % item)





# # Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('/home/leo/RSS_repo/depth_net/saved_model/depth_model_v4/weights00000060.h5',custom_objects={'autoencoder_loss': get_loss.autoencoder_loss})
# model = tf.keras.models.load_model('/tfdepth/model_HD/saved_model/ft_parameter/few_layer_TC_'+str((j+1)*5)+'.h5')

abs_rel = []
sq_rel = []
rmse = []
rmse_log = []
d1 = []
d2 = []
d3 = []

tc = []

for i in range (0, int(len(train_data)/2), 2):
    depth1 = tf.reshape(depth_data[i], [1, height, width, 1])
    depth2 = tf.reshape(depth_data[i+1], [1, height, width, 1])
    rgb1 = tf.reshape(train_data[i], [1, height, width, 3])
    rgb2 = tf.reshape(train_data[i+1], [1, height, width, 3])

    para = 1
    ex = 0


    gt1 = depth1[0, :, :, 0]*para+ex
    gt2 = depth2[0, :, :, 0]*para+ex
    _, _, prediction1 = model.predict(rgb1)
    _, _, prediction2 = model.predict(rgb2)
    # prediction1 = model.predict(rgb1)
    # prediction2 = model.predict(rgb2)
    pred1 = prediction1[0, :, :, 0]*para+ex
    pred2 = prediction2[0, :, :, 0]*para+ex

    # for temporal consistency
    mask1 = tf.reshape(mask_data[i], [1, height, width, 1])
    mask2 = tf.reshape(mask_data[i+1], [1, height, width, 1])
    M1 = mask1[0, :, :, 0]
    M2 = mask2[0, :, :, 0]
    M = M1+M2
    M = np.where(M==1.0, 0.0, M)
    M = np.where(M>=1.5, 1.0, M)

    D1 = np.multiply(pred1,M)
    D2 = np.multiply(pred2,M)
    tc_temp = LA.norm(np.abs(np.subtract(D1,D2)), 2)/np.sum(M)
    tc.append(tc_temp)


    abs_rel_temp,sq_rel_remp,rmse_temp,rms_log_remp, d1_temp, d2_temp, d3_temp = accuracy_metric(gt1, pred1)
    abs_rel.append(abs_rel_temp)
    sq_rel.append(sq_rel_remp)
    rmse.append(rmse_temp)
    rmse_log.append(rms_log_remp)
    d1.append(d1_temp)
    d2.append(d2_temp)
    d3.append(d3_temp)

    abs_rel_temp,sq_rel_remp,rmse_temp,rms_log_remp, d1_temp, d2_temp, d3_temp = accuracy_metric(gt2, pred2)
    abs_rel.append(abs_rel_temp)
    sq_rel.append(sq_rel_remp)
    rmse.append(rmse_temp)
    rmse_log.append(rms_log_remp)
    d1.append(d1_temp)
    d2.append(d2_temp)

result = []
result.append(mean(abs_rel))
result.append(mean(sq_rel))
result.append(mean(rmse))
result.append(mean(rmse_log))
result.append(mean(d1))
result.append(mean(d2))
result.append(mean(d3))

result.append(mean(tc))
print(result)

