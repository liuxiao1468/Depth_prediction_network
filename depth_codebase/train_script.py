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

TF_FORCE_GPU_ALLOW_GROWTH=True


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



dataset_train = 'nyudepthv2_train_files_with_gt.txt'
width = 640
height = 480
batch_size = 16

get_dataset = dataset_prep.get_NYU_dataset()
get_depth_net = depth_prediction_net.get_depth_net()
get_loss = loss.get_loss()


train_data, depth_data = get_dataset.select_batch(dataset_train, batch_size)
print(len(depth_data))
for i in range (len(depth_data)):
	print(depth_data[i].shape)


train_gen = get_dataset.train_generator(dataset_train,batch_size)
validation_gen = get_dataset.validation_generator(dataset_train,batch_size)

opt = Adam(lr=1e-3)

Disp_ResNet_autoencoder = get_depth_net.DispResNet_autoencoder(height, width, 3)
Disp_ResNet_autoencoder.compile(optimizer=opt, loss=get_loss.autoencoder_loss,loss_weights= [1/64, 1/32, 1/16, 1/8, 1/4, 1/2])
# print(DispNet_autoencoder.summary())



# # Create a MirroredStrategy.
# strategy = tf.distribute.MirroredStrategy()
# print("Number of devices: {}".format(strategy.num_replicas_in_sync))

# # Open a strategy scope.
# with strategy.scope():
#     # Everything that creates variables should be under the strategy scope.
#     # In general this is only model construction & `compile()`.
#     Disp_ResNet_autoencoder = get_depth_net.DispResNet_autoencoder(height, width, 3)
#     Disp_ResNet_autoencoder.compile(optimizer=opt, loss=get_loss.autoencoder_loss,loss_weights= [1/64, 1/32, 1/16, 1/8, 1/4, 1/2])
#     i=2
#     Disp_ResNet_autoencoder.fit(x = train_gen, steps_per_epoch = 150, validation_data = validation_gen, epochs=150, validation_steps= 20)
#     Disp_ResNet_autoencoder.save('model_'+str(i)+'_DispNet_autoencoder.h5')

i=2
Disp_ResNet_autoencoder.fit(x = train_gen, steps_per_epoch = 1500, validation_data = validation_gen, epochs=100, validation_steps= 20)
Disp_ResNet_autoencoder.save('model_'+str(i)+'_DispNet_autoencoder.h5')


