import pandas as pd
import dataset_prep
import depth_prediction_net
import loss
from tensorflow.keras.optimizers import Adam
import numpy as np




dataset_train = '/home/leo/deeplearning/depth_prediction/train'
ground_truth = '/home/leo/deeplearning/depth_prediction/ground_truth'
width = 320
height = 180
batch_size = 32

get_dataset = dataset_prep.get_dataset()
get_depth_net = depth_prediction_net.get_depth_net()
get_loss = loss.get_loss()


# train_data, depth_data = get_dataset.select_batch(dataset_train,ground_truth,batch_size)
# print(depth_data.shape)


train_gen = get_dataset.train_generator(dataset_train,ground_truth,batch_size)
validation_gen = get_dataset.validation_generator(dataset_train,ground_truth,batch_size)

opt = Adam(lr=1e-3)
autoencoder = get_depth_net.ResNet_autoencoder(height, width, 3, batch_size)
autoencoder.compile(optimizer=opt, loss=get_loss.autoencoder_loss)
# print(autoencoder.summary())

i=1
autoencoder.fit_generator(train_gen, steps_per_epoch = 180, validation_data = validation_gen, epochs=1, validation_steps= 20)
autoencoder.save('model_'+str(i)+'_ResNet_autoencoder.h5')



