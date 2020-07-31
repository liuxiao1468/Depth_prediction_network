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
    # total_loss = reconstruction_loss
    # total_variation_weight = 20
    total_loss = 100*reconstruction_loss + tf.image.total_variation(output)
    # total_loss =tf.image.total_variation(output)
    return total_loss

def generate_and_save_images(model, epoch, test_sample):
    predictions = model.predict(test_sample)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0], cmap='gray')
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()



pickle_in = open("train_data.pickle", "rb")
train_data = pickle.load(pickle_in)
pickle_in = open("depth_data.pickle", "rb")
depth_data = pickle.load(pickle_in)


num_examples_to_generate = 16
test_sample, depth_sample = select_batch(train_data, depth_data, num_examples_to_generate)

EPOCHS = 10


# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('model_1_ResNet_autoencoder.h5', custom_objects={'autoencoder_loss': autoencoder_loss})

# Show the model architecture
# model.summary()

generate_and_save_images(model, EPOCHS, test_sample)

fig = plt.figure(figsize=(4, 4))

for i in range(test_sample.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(test_sample[i, :, :, :], cmap='gray')
    plt.axis('off')
plt.show()  # display!


fig = plt.figure(figsize=(4, 4))
for j in range(depth_sample.shape[0]):
    plt.subplot(4, 4, j + 1)
    plt.imshow(depth_sample[j, :, :, 0], cmap='gray')
    plt.axis('off')

# plt.savefig('try.png')
# tight_layout minimizes the overlap between 2 sub-plots
plt.show()  # display!
