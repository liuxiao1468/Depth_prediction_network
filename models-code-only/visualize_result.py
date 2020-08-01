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
import math

from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dropout, Flatten, Dense, Conv2D, MaxPooling2D, Input, Activation, Add
from tensorflow.keras.layers import Reshape, UpSampling2D, InputLayer, Lambda, ZeroPadding2D, AveragePooling2D
from tensorflow.keras.layers import Cropping2D, Conv2DTranspose, BatchNormalization, Concatenate

from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model


def select_batch(train_data, depth_data, b_size):
    index = np.random.choice(train_data.shape[0], b_size, replace=False)
    index = sorted(index)
    params1 = tf.constant(train_data)
    params2 = tf.constant(depth_data)
    indices = tf.constant(index)
    test_sample = tf.gather(params1, indices)
    depth_sample = tf.gather(params2, indices)
    return test_sample, depth_sample


def autoencoder_loss(depth_img, output):
    # Compute error in reconstruction
    reconstruction_loss = mse(K.flatten(depth_img) , K.flatten(output))

    dy_true, dx_true = tf.image.image_gradients(depth_img)
    dy_pred, dx_pred = tf.image.image_gradients(output)
    term3 = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

    tv = (1e-8)*tf.reduce_sum(tf.image.total_variation(output))

    total_loss = 100*reconstruction_loss + term3 + tv
    # total_loss = tv
    return total_loss


def generate_and_save_images(model, epoch, test_sample):
    predictions = model.predict(test_sample)
    fig = plt.figure(figsize=(4, 4))
    # fig = plt.figure()

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')
        # plt.savefig(str(i)+'.png')
        # plt.show()
    # plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()


def save_RGB_images(test_sample, depth_sample):
    # fig = plt.figure(figsize=(4, 4))
    fig = plt.figure()

    for i in range(test_sample.shape[0]):
        plt.imshow(test_sample[i, :, :, :])
        plt.axis('off')
        plt.savefig(str(i)+'-RGB.png')



def save_depth_images(test_sample, depth_sample):
    fig = plt.figure()

    for j in range(depth_sample.shape[0]):
        plt.imshow(depth_sample[j, :, :, :], cmap='gray')
        plt.axis('off')
        plt.savefig(str(i)+'-D.png')

## Get a test sample
# pickle_in = open("train_data.pickle", "rb")
# train_data = pickle.load(pickle_in)
# pickle_in = open("depth_data.pickle", "rb")
# depth_data = pickle.load(pickle_in)

# num_examples_to_generate = 16
# test_sample, depth_sample = select_batch(train_data, depth_data, num_examples_to_generate)

# pickle_out = open("test_sample.pickle","wb")
# pickle.dump(test_sample, pickle_out)
# pickle_out.close()

# pickle_out = open("depth_sample.pickle","wb")
# pickle.dump(depth_sample, pickle_out)
# pickle_out.close()


pickle_in = open("test_sample.pickle", "rb")
test_sample = pickle.load(pickle_in)
pickle_in = open("depth_sample.pickle", "rb")
depth_sample = pickle.load(pickle_in)


EPOCHS = 10

# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('model_1_ResNet_autoencoder.h5', custom_objects={
                                   'autoencoder_loss': autoencoder_loss})
# model = tf.keras.models.load_model('U-net_depth.h5', custom_objects={'autoencoder_loss': autoencoder_loss})
# Show the model architecture
model.summary()

generate_and_save_images(model, EPOCHS, test_sample)
# save_depth_images(test_sample, depth_sample)

fig = plt.figure(figsize=(4, 4))

for i in range(test_sample.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(test_sample[i, :, :, :], cmap='gray')
    plt.axis('off')
    # plt.show()
plt.show()  # display!


fig = plt.figure(figsize=(4, 4))
# fig = plt.figure()
for j in range(depth_sample.shape[0]):
    plt.subplot(4, 4, j + 1)
    plt.imshow(depth_sample[j, :, :, 0], cmap='gray')
    plt.axis('off')
    # plt.savefig(str(j)+'-D.png')
    # plt.show()
plt.show()  # display!
