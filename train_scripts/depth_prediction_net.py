import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import matplotlib.pyplot as plt
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
import time
# import matplotlib.pyplot as plt
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


class get_depth_net():
    # @staticmethod
    def identity_block(self, X, f, filters, stage, block):
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value. We'll need this later to add back to the main path.
        X_shortcut = X

        # First component of main path
        X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                   name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path
        X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
                   name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path
        X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                   name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X = Add()([X, X_shortcut])
        out_shortcut = X
        X = Activation('relu')(X)

        return X, out_shortcut

    def identity_block_transpose(self, X, f, filters, stage, block):
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value. We'll need this later to add back to the main path.
        X_shortcut = X

        # First component of main path
        X = Conv2DTranspose(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                            name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path
        X = Conv2DTranspose(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
                            name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path
        X = Conv2DTranspose(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                            name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X = Add()([X, X_shortcut])
        out_shortcut = X
        X = Activation('relu')(X)

        return X, out_shortcut

    def convolutional_block(self, X, f, filters, stage, block, s=2):
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value
        X_shortcut = X

        ##### MAIN PATH #####
        # First component of main path
        X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base +
                   '2a', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path
        X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
                   name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path
        X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                   name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

        ##### SHORTCUT PATH ####
        X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), name=conv_name_base +
                            '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(
            axis=3, name=bn_name_base + '1')(X_shortcut)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X = Add()([X, X_shortcut])
        out_shortcut = X
        X = Activation('relu')(X)

        return X, out_shortcut

    def convolutional_block_transpose(self, X, f, filters, stage, block, s=2):
        # defining name basis
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        # Retrieve Filters
        F1, F2, F3 = filters

        # Save the input value
        X_shortcut = X

        ##### MAIN PATH #####
        # First component of main path
        X = Conv2DTranspose(F1, (1, 1), strides=(s, s), name=conv_name_base +
                            '2a', padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
        X = Activation('relu')(X)

        # Second component of main path
        X = Conv2DTranspose(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same',
                            name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
        X = Activation('relu')(X)

        # Third component of main path
        X = Conv2DTranspose(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid',
                            name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

        ##### SHORTCUT PATH ####
        X_shortcut = Conv2DTranspose(F3, (1, 1), strides=(s, s), name=conv_name_base +
                                     '1', padding='valid', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
        X_shortcut = BatchNormalization(
            axis=3, name=bn_name_base + '1')(X_shortcut)

        # Final step: Add shortcut value to main path, and pass it through a RELU activation
        X = Add()([X, X_shortcut])
        out_shortcut = X
        X = Activation('relu')(X)

        return X, out_shortcut

    def ResNet_autoencoder(self, height, width, depth, latentDim=64):
        X_input = Input(shape=(height, width, depth))

        X = X_input
        # encoder Stage 1
        X = Conv2D(32, (3, 3), strides=(2, 2), name='conv1-1',
                   padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name='bn_conv1-1')(X)
        X = Activation('relu')(X)
        X = Conv2D(32, (1, 1), strides=(1, 1), name='conv1-2',
                   padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name='bn_conv1-2')(X)

        skip_connect_1 = X
        X = Activation('relu')(X)

        # encoder Stage 2
        X = Conv2D(64, (3, 3), strides=(2, 2), name='conv2-1',
                   padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name='bn_conv2-1')(X)
        X = Activation('relu')(X)
        X = Conv2D(64, (1, 1), strides=(1, 1), name='conv2-2',
                   padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name='bn_conv2-2')(X)

        skip_connect_2 = X
        X = Activation('relu')(X)

        # encoder Stage 3
        X, _ = self.convolutional_block(
            X, f=3, filters=[64, 64, 128], stage=3, block='a', s=2)
        X, skip_connect_3 = self.identity_block(
            X, 3, [64, 64, 128], stage=3, block='b')

        # encoder Stage 4
        X, _ = self.convolutional_block(
            X, f=3, filters=[128, 128, 256], stage=4, block='a', s=2)
        X, skip_connect_4 = self.identity_block(
            X, 3, [128, 128, 256], stage=4, block='b')

        # encoder Stage 5
        X, _ = self.convolutional_block(
            X, f=3, filters=[256, 256, 512], stage=5, block='a', s=2)
        X, skip_connect_5 = self.identity_block(
            X, 3, [256, 256, 512], stage=5, block='b')

        # latent-space representation
        volumeSize = K.int_shape(X)
        X = Flatten()(X)
        latent = Dense(latentDim)(X)

        # encoder = Model(X_input, latent, name="encoder")

        # latentInputs = Input(shape=(latentDim,))
        X = Dense(np.prod(volumeSize[1:]))(latent)
        X = Reshape((volumeSize[1], volumeSize[2], volumeSize[3]))(X)

        # # # decoder Stage 1
        X = Concatenate()([X, skip_connect_5])

        X, _ = self.identity_block_transpose(
            X, 3, [1024, 1024, 1024], stage=6, block='b')
        X = Conv2DTranspose(512, (1, 1), strides=(
            1, 1), padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
        X, _ = self.convolutional_block_transpose(
            X, f=3, filters=[512, 256, 256], stage=6, block='a', s=2)

        # # # decoder Stage 2
        X = Concatenate()([X, skip_connect_4])

        X, _ = self.identity_block_transpose(
            X, 3, [512, 512, 512], stage=7, block='b')
        X = Conv2DTranspose(256, (1, 1), strides=(
            1, 1), padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
        X, _ = self.convolutional_block_transpose(
            X, f=3, filters=[256, 128, 128], stage=7, block='a', s=2)
        X = Cropping2D(cropping=((1, 0), (0, 0)), data_format=None)(X)

        # # decoder Stage 3

        X = Concatenate()([X, skip_connect_3])

        X, _ = self.identity_block_transpose(
            X, 3, [256, 256, 256], stage=8, block='b')
        X = Conv2DTranspose(256, (1, 1), strides=(
            1, 1), padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
        X, _ = self.convolutional_block_transpose(
            X, f=3, filters=[128, 64, 64], stage=8, block='a', s=2)
        X = Cropping2D(cropping=((1, 0), (0, 0)), data_format=None)(X)

        # # # decoder Stage 4
        X = Concatenate()([X, skip_connect_2])
        X = Conv2DTranspose(128, (1, 1), strides=(
            1, 1), name='conv9-1', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name='bn_conv9-1')(X)
        X = Activation('relu')(X)
        X = Conv2DTranspose(64, (3, 3), strides=(2, 2), name='conv9-2',
                            padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name='bn_conv9-2')(X)
        X = Activation('relu')(X)

        # # decoder Stage 5
        X = Concatenate()([X, skip_connect_1])
        X = Conv2DTranspose(64, (1, 1), strides=(
            1, 1), name='conv10-1', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name='bn_conv10-1')(X)
        X = Activation('relu')(X)
        X = Conv2DTranspose(32, (1, 1), strides=(
            1, 1), name='conv10-2', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name='bn_conv10-2')(X)
        X = Activation('relu')(X)
        X = Conv2DTranspose(16, (1, 1), strides=(
            1, 1), name='conv10-3', kernel_initializer=glorot_uniform(seed=0))(X)
        X = BatchNormalization(axis=3, name='bn_conv10-3')(X)
        X = Activation('relu')(X)

        X = Conv2DTranspose(1, (3, 3), strides=(2, 2), padding="same")(X)
        outputs = Activation('sigmoid')(X)

        autoencoder = Model(inputs=X_input, outputs=outputs,
                            name='ResNet_autoencoder')
        # print(autoencoder.summary())
        return autoencoder
