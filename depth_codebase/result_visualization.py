import pandas as pd
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras.layers import Reshape, UpSampling2D, InputLayer, Lambda, ZeroPadding2D, AveragePooling2D
import dataset_prep
import depth_prediction_net
import loss
import matplotlib.pyplot as plt
import cv2
import numpy as np

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)




dataset_train = '/home/xiaoliu/zed_camera/path_2/RGB'
ground_truth = '/home/xiaoliu/zed_camera/path_2/Depth'
width = 320
height = 180
batch_size = 16

get_dataset = dataset_prep.get_dataset()
get_depth_net = depth_prediction_net.get_depth_net()
get_loss = loss.get_loss()


def generate_and_save_images(model, test_sample):
    pre_2, pre_3, pre_4, pre_5, pre_6 , predictions = model.predict(test_sample)
    fig = plt.figure(figsize=(4, 4))
    # fig = plt.figure()

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0])
        plt.axis('off')
        # plt.savefig(str(i)+'.png')
        # plt.show()
    # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()
    return predictions



test_sample, depth_sample = get_dataset.select_batch(dataset_train, ground_truth, 16)

print("here: ", test_sample.shape)

# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('model_1_DispNet_autoencoder.h5', custom_objects={
                                   'autoencoder_loss': get_loss.autoencoder_loss})
# model = tf.keras.models.load_model('U-net_depth.h5', custom_objects={'autoencoder_loss': autoencoder_loss})
# Show the model architecture
# model.summary()

predictions = generate_and_save_images(model, test_sample)




fig = plt.figure(figsize=(4, 4))

for i in range(test_sample.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(test_sample[i, :, :, :])
    plt.axis('off')
    # plt.show()
plt.show()  # display!


fig = plt.figure(figsize=(4, 4))
# fig = plt.figure()
for j in range(depth_sample[-1].shape[0]):
    plt.subplot(4, 4, j + 1)
    # plt.imshow(depth_sample[j, :, :, 0], cmap='gray')
    plt.imshow(depth_sample[-1][j, :, :, 0])
    plt.axis('off')
    # plt.savefig(str(j)+'-D.png')
    # plt.show()
plt.show()  # display!


from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import numpy as np
from statistics import mean 

rms = []
rms_log = []
mse = []
d1 = []
d2 = []
d3 = []

for i in range(predictions.shape[0]):
    gt = depth_sample[-1][i, :, :, 0]*(255/20)
    pred = predictions[i, :, :, 0]*(255/20)
    rms_temp = sqrt(mean_squared_error(gt, pred))
    rms.append(rms_temp)
    rms_log_temp = sqrt(mean_squared_log_error(gt, pred))
    rms_log.append(rms_log_temp)
    mse_temp = mean_absolute_error(gt, pred)
    mse.append(mse_temp)

    thresh = np.maximum((gt / pred), (pred / gt))
    d1_temp = (thresh < 1.25).mean()
    d2_temp = (thresh < 1.25 ** 2).mean()
    d3_temp = (thresh < 1.25 ** 3).mean()
    d1.append(d1_temp)
    d2.append(d2_temp)
    d3.append(d3_temp)

print(mean(rms), mean(rms_log), mean(mse), mean(d1), mean(d2), mean(d3)  )







