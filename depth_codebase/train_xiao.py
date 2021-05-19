import pandas as pd
import dataset_prep
import depth_prediction_net
import loss
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Input, Activation, Add
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.callbacks import TensorBoard

TF_FORCE_GPU_ALLOW_GROWTH=True


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


width = 160
height = 90
batch_size = 16

get_dataset = dataset_prep.get_dataset()
get_depth_net = depth_prediction_net.get_depth_net()
get_loss = loss.get_loss()


train_data, depth_data = get_dataset.select_batch(batch_size)
# print(len(depth_data))
# for i in range (len(depth_data)):
# 	print(depth_data[i].shape)


train_gen = get_dataset.train_generator(batch_size)
validation_gen = get_dataset.validation_generator(batch_size)

opt = Adam(lr=1e-5)
Disp_ResNet_autoencoder = get_depth_net.DispNet_autoencoder(height, width, 3)
Disp_ResNet_autoencoder.compile(optimizer=opt, loss=get_loss.autoencoder_loss)
# fine_tune_autoencoder.compile(optimizer=opt, loss=get_loss.autoencoder_loss,loss_weights= [1/64, 1/32, 1/16, 1/8, 1/4, 1])
# print(Disp_ResNet_autoencoder.summary())

mc = tf.keras.callbacks.ModelCheckpoint('./saved_model/depth_model_v4/weights{epoch:08d}.h5', save_weights_only=False, period=3)

NAME = "depth_net_2.0"
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))


Disp_ResNet_autoencoder.fit_generator(train_gen, steps_per_epoch =1875, validation_data = validation_gen, epochs=60, validation_steps= 100, callbacks=[mc])

# # Disp_ResNet_autoencoder.save('/tfdepth/model_HD/NYU_'+str(i)+'_DispNet_autoencoder.h5')


