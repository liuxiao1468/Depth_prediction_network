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

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions


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
        # X = Cropping2D(cropping=((1, 0), (0, 0)), data_format=None)(X)

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
        X = Cropping2D(cropping=((1, 0), (0, 0)), data_format=None)(X)

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

    def DispNet_encoder(self, height, width, depth):
        # Conv 1 (batch, 192, 384, 64)
        inputs = Input(shape=(height, width, depth))
        print("in ----- ",inputs.shape)

        conv_1 = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(inputs)
        conv_1 = Activation('relu')(conv_1)
        print("in_1----- ",conv_1.shape)
        # self.conv_1 = conv_1
        # Conv 2 (batch, 96, 192, 128)
        conv_2 = Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same')(conv_1)
        conv_2 = Activation('relu')(conv_2)
        print("in_2----- ",conv_2.shape)
        # self.conv_2 = conv_2
        # Conv 3a (batch, 48, 96, 256)
        conv_3a = Conv2D(256, kernel_size=(5, 5), strides=(2, 2), padding='same')(conv_2)
        conv_3a = Activation('relu')(conv_3a)
        # Conv 3b (batch, 48, 96, 256)
        conv_3b = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_3a)
        conv_3b = Activation('relu')(conv_3b)
        print("in_3----- ",conv_3b.shape)
        # self.conv_3b = conv_3b
        # Conv 4a (batch, 24, 48, 512)
        conv_4a = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv_3b)
        conv_4a = Activation('relu')(conv_4a)
        # Conv 4b (batch, 24, 48, 512)
        conv_4b = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_4a)
        conv_4b = Activation('relu')(conv_4b)
        print("in_4----- ",conv_4b.shape)
        # self.conv_4b = conv_4b
        # Conv 5a (batch, 12, 24, 512)
        conv_5a = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv_4b)
        conv_5a = Activation('relu')(conv_5a)
        # Conv 5b (batch, 12, 24, 512)
        conv_5b = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_5a)
        conv_5b = Activation('relu')(conv_5b)
        print("in_5----- ",conv_5b.shape)
        # self.conv_5b = conv_5b
        # Conv 6a (batch, 6, 12, 1024)
        conv_6a = Conv2D(1024, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv_5b)
        conv_6a = Activation('relu')(conv_6a)
        # Conv 6b (batch, 6, 12, 1024)
        conv_6b = Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_6a)
        conv_6b = Activation('relu')(conv_6b)
        print("in_6----- ",conv_6b.shape)
        # self.conv_6b = conv_6b
        # Prediction_Loss 6 (batch, 6, 12, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_6b)
        prediction = Activation('relu', name='6x12')(prediction)
        # print("not 6---- ",prediction.shape)
        pre_1 = prediction

        encoder = Model(inputs=inputs, outputs=[conv_1, conv_2,  conv_3b,  conv_4b, conv_5b, conv_6b, pre_1],
                            name='DispNet_encoder')

        # print(encoder.summary())
        return conv_1, conv_2,  conv_3b,  conv_4b, conv_5b, conv_6b, pre_1


    def DispNet_decoder(self, conv_1, conv_2, conv_3b, conv_4b, conv_5b, inputs):
        # Upconv 6 (batch, 12, 24, 512)

        upconv_5 = Conv2DTranspose(512, kernel_size=(4, 4), strides=(2, 2), padding='same')(inputs)
        upconv_5 = BatchNormalization(axis=-1)(upconv_5)
        upconv_5 = Activation('relu')(upconv_5)
        # Iconv 5 (batch, 12, 24, 512)
        c = Concatenate(axis=-1)([upconv_5, conv_5b])
        iconv_5 = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
        iconv_5 = Activation('relu')(iconv_5)
        # Prediction_Loss 5 (batch, 12, 24, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_5)
        prediction = Activation('relu', name='12x24')(prediction)
        print("5---- ",prediction.shape)
        pre_2 = prediction
        

        # Upconv 4 (batch, 24, 48, 256)
        upconv_4 = Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), padding='same')(iconv_5)
        upconv_4 = BatchNormalization(axis=-1)(upconv_4)
        upconv_4 = Activation('relu')(upconv_4)
        # Iconv 4 (batch, 24, 48, 256)
        c = Concatenate(axis=-1)([upconv_4, conv_4b])
        iconv_4 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
        iconv_4 = Activation('relu')(iconv_4)
        # Prediction_Loss 4 (batch, 24, 48, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_4)
        prediction = Activation('relu', name='24x48')(prediction)
        print("4---- ",prediction.shape)
        pre_3 = prediction
        

        # Upconv 3 (batch, 48, 96, 128)
        upconv_3 = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same')(iconv_4)
        upconv_3 = BatchNormalization(axis=-1)(upconv_3)
        upconv_3 = Activation('relu')(upconv_3)
        # Iconv 3 (batch, 48, 96, 128)
        upconv_3 = Cropping2D(cropping=((1, 0), (0, 0)), data_format=None)(upconv_3)
        c = Concatenate(axis=-1)([upconv_3, conv_3b])
        iconv_3 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
        iconv_3 = Activation('relu')(iconv_3)
        # Prediction_Loss 3 (batch, 48, 96, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_3)
        prediction = Activation('relu', name='48x96')(prediction)
        print("3---- ",prediction.shape)
        pre_4 = prediction
        

        # Upconv 2 (batch, 96, 192, 64)
        upconv_2 = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='same')(iconv_3)
        upconv_2 = BatchNormalization(axis=-1)(upconv_2)
        upconv_2 = Activation('relu')(upconv_2)
        # Iconv 2 (batch, 96, 192, 64)
        upconv_2 = Cropping2D(cropping=((1, 0), (0, 0)), data_format=None)(upconv_2)
        c = Concatenate(axis=-1)([upconv_2, conv_2])
        iconv_2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
        iconv_2 = Activation('relu')(iconv_2)
        # Prediction_Loss 2 (batch, 96, 192, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_2)
        prediction = Activation('relu', name='96x192')(prediction)
        print("2---- ",prediction.shape)
        pre_5 = prediction
        

        # Upconv 1 (batch, 192, 384, 32)
        upconv_1 = Conv2DTranspose(32, kernel_size=(4, 4), strides=(2, 2), padding='same')(iconv_2)
        upconv_1 = BatchNormalization(axis=-1)(upconv_1)
        upconv_1 = Activation('relu')(upconv_1)
        # Iconv 1 (batch, 192, 384, 32)
        c = Concatenate(axis=-1)([upconv_1, conv_1])
        iconv_1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
        iconv_1 = Activation('relu')(iconv_1)
        # Prediction_Loss 1 (batch, 192, 384, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_1)
        prediction = Activation('relu', name='192x384')(prediction)
        print("1---- ",prediction.shape)
        pre_6 = prediction

        # Upconv 1 (batch, 192, 384, 32)
        upconv_0 = Conv2DTranspose(16, kernel_size=(4, 4), strides=(2, 2), padding='same')(iconv_1)
        upconv_0 = BatchNormalization(axis=-1)(upconv_0)
        upconv_0 = Activation('relu')(upconv_0)
        # Iconv 1 (batch, 192, 384, 32)
        iconv_0 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(upconv_0)
        iconv_0 = Activation('relu')(iconv_0)
        # Prediction_Loss 1 (batch, 192, 384, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_0)
        prediction = Activation('relu', name='192x384')(prediction)
        print("0---- ",prediction.shape)
        pre_final = prediction

        decoder = Model(inputs=[conv_1, conv_2, conv_3b, conv_4b, conv_5b, inputs], outputs=[pre_2, pre_3, pre_4, pre_5, pre_6, pre_final], name='DispNet_decoder')
        print(decoder.summary())

        return  pre_2, pre_3, pre_4, pre_5, pre_6, pre_final



    def DispNet_autoencoder(self, height, width, depth):
        # Conv 1 (batch, 192, 384, 64)
        inputs = Input(shape=(height, width, depth))
        print("in ----- ",inputs.shape)

        conv_1 = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(inputs)
        conv_1 = Activation('relu')(conv_1)
        print("in_1----- ",conv_1.shape)
        # self.conv_1 = conv_1
        # Conv 2 (batch, 96, 192, 128)
        conv_2 = Conv2D(128, kernel_size=(5, 5), strides=(2, 2), padding='same')(conv_1)
        conv_2 = Activation('relu')(conv_2)
        print("in_2----- ",conv_2.shape)
        # self.conv_2 = conv_2
        # Conv 3a (batch, 48, 96, 256)
        conv_3a = Conv2D(256, kernel_size=(5, 5), strides=(2, 2), padding='same')(conv_2)
        conv_3a = Activation('relu')(conv_3a)
        # Conv 3b (batch, 48, 96, 256)
        conv_3b = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_3a)
        conv_3b = Activation('relu')(conv_3b)
        print("in_3----- ",conv_3b.shape)
        # self.conv_3b = conv_3b
        # Conv 4a (batch, 24, 48, 512)
        conv_4a = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv_3b)
        conv_4a = Activation('relu')(conv_4a)
        # Conv 4b (batch, 24, 48, 512)
        conv_4b = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_4a)
        conv_4b = Activation('relu')(conv_4b)
        print("in_4----- ",conv_4b.shape)
        # self.conv_4b = conv_4b
        # Conv 5a (batch, 12, 24, 512)
        conv_5a = Conv2D(512, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv_4b)
        conv_5a = Activation('relu')(conv_5a)
        # Conv 5b (batch, 12, 24, 512)
        conv_5b = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_5a)
        conv_5b = Activation('relu')(conv_5b)
        print("in_5----- ",conv_5b.shape)
        # self.conv_5b = conv_5b
        # Conv 6a (batch, 6, 12, 1024)
        conv_6a = Conv2D(1024, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv_5b)
        conv_6a = Activation('relu')(conv_6a)
        # Conv 6b (batch, 6, 12, 1024)
        conv_6b = Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_6a)
        conv_6b = Activation('relu')(conv_6b)
        print("in_6----- ",conv_6b.shape)
        # self.conv_6b = conv_6b
        # Prediction_Loss 6 (batch, 6, 12, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_6b)
        prediction = Activation('relu')(prediction)
        # print("not 6---- ",prediction.shape)
        pre_1 = prediction

        upconv_5 = Conv2DTranspose(512, kernel_size=(4, 4), strides=(2, 2), padding='same')(conv_6b)
        upconv_5 = BatchNormalization(axis=-1)(upconv_5)
        upconv_5 = Activation('relu')(upconv_5)
        upconv_5 = Cropping2D(cropping=((1, 0), (1, 0)), data_format=None)(upconv_5)
        # Iconv 5 (batch, 12, 24, 512)
        c = Concatenate(axis=-1)([upconv_5, conv_5b])
        iconv_5 = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
        iconv_5 = Activation('relu')(iconv_5)
        # Prediction_Loss 5 (batch, 12, 24, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_5)
        prediction = Activation('relu')(prediction)
        print("5---- ",prediction.shape)
        pre_2 = prediction
        

        # Upconv 4 (batch, 24, 48, 256)
        upconv_4 = Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), padding='same')(iconv_5)
        upconv_4 = BatchNormalization(axis=-1)(upconv_4)
        upconv_4 = Activation('relu')(upconv_4)
        # Iconv 4 (batch, 24, 48, 256)
        c = Concatenate(axis=-1)([upconv_4, conv_4b])
        iconv_4 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
        iconv_4 = Activation('relu')(iconv_4)
        # Prediction_Loss 4 (batch, 24, 48, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_4)
        prediction = Activation('relu', name='24x48')(prediction)
        print("4---- ",prediction.shape)
        pre_3 = prediction
        

        # Upconv 3 (batch, 48, 96, 128)
        upconv_3 = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same')(iconv_4)
        upconv_3 = BatchNormalization(axis=-1)(upconv_3)
        upconv_3 = Activation('relu')(upconv_3)
        # Iconv 3 (batch, 48, 96, 128)
        # upconv_3 = Cropping2D(cropping=((1, 0), (0, 0)), data_format=None)(upconv_3)
        c = Concatenate(axis=-1)([upconv_3, conv_3b])
        iconv_3 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
        iconv_3 = Activation('relu')(iconv_3)
        # Prediction_Loss 3 (batch, 48, 96, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_3)
        prediction = Activation('relu', name='48x96')(prediction)
        print("3---- ",prediction.shape)
        pre_4 = prediction
        

        # Upconv 2 (batch, 96, 192, 64)
        upconv_2 = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='same')(iconv_3)
        upconv_2 = BatchNormalization(axis=-1)(upconv_2)
        upconv_2 = Activation('relu')(upconv_2)
        # Iconv 2 (batch, 96, 192, 64)
        upconv_2 = Cropping2D(cropping=((1, 0), (0, 0)), data_format=None)(upconv_2)
        c = Concatenate(axis=-1)([upconv_2, conv_2])
        iconv_2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
        iconv_2 = Activation('relu')(iconv_2)
        # Prediction_Loss 2 (batch, 96, 192, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_2)
        prediction = Activation('relu', name='96x192')(prediction)
        print("2---- ",prediction.shape)
        pre_5 = prediction
        

        # Upconv 1 (batch, 192, 384, 32)
        upconv_1 = Conv2DTranspose(32, kernel_size=(4, 4), strides=(2, 2), padding='same')(iconv_2)
        upconv_1 = BatchNormalization(axis=-1)(upconv_1)
        upconv_1 = Activation('relu')(upconv_1)
        # Iconv 1 (batch, 192, 384, 32)
        upconv_1 = Cropping2D(cropping=((1, 0), (0, 0)), data_format=None)(upconv_1)
        c = Concatenate(axis=-1)([upconv_1, conv_1])
        iconv_1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
        iconv_1 = Activation('relu')(iconv_1)
        # Prediction_Loss 1 (batch, 192, 384, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_1)
        prediction = Activation('relu', name='192x384')(prediction)
        print("1---- ",prediction.shape)
        pre_6 = prediction

        # Upconv 1 (batch, 192, 384, 32)
        upconv_0 = Conv2DTranspose(16, kernel_size=(4, 4), strides=(2, 2), padding='same')(iconv_1)
        upconv_0 = BatchNormalization(axis=-1)(upconv_0)
        upconv_0 = Activation('relu')(upconv_0)
        # Iconv 1 (batch, 192, 384, 32)
        iconv_0 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(upconv_0)
        iconv_0 = Activation('relu')(iconv_0)
        # Prediction_Loss 1 (batch, 192, 384, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_0)
        prediction = Activation('relu', name='final_layer')(prediction)
        print("0---- ",prediction.shape)
        pre_final = prediction

        DispNet_autoencoder = Model(inputs=inputs, outputs=[pre_5, pre_6 , pre_final], name='Disp_ResNet_autoencoder')
        print(DispNet_autoencoder.summary())

        return  DispNet_autoencoder

    def DispResNet_autoencoder(self, height, width, depth):
        # Conv 1 (batch, 192, 384, 64)
        inputs = Input(shape=(height, width, depth))
        print("in ----- ",inputs.shape)

        conv_1 = Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(inputs)
        conv_1 = Activation('relu')(conv_1)
        print("in_1----- ",conv_1.shape)
        # self.conv_1 = conv_1
        # Conv 2 (batch, 96, 192, 128)

        # encoder Stage 2
        conv_2a, _ = self.convolutional_block(conv_1, f=3, filters=[64, 64, 128], stage=2, block='a', s=2)
        conv_2b, skip_connect_3 = self.identity_block(conv_2a, 3, [64, 64, 128], stage=2, block='b')
        print("in_2----- ",conv_2b.shape)

        # encoder Stage 3
        conv_3a, _ = self.convolutional_block(conv_2b, f=3, filters=[128, 128, 256], stage=3, block='a', s=2)
        conv_3b, _ = self.identity_block(conv_3a, 3, [128, 128, 256], stage=3, block='b')
        print("in_3----- ",conv_3b.shape)

        # encoder Stage 4
        conv_4a, _ = self.convolutional_block(conv_3b, f=3, filters=[256, 256, 512], stage=4, block='a', s=2)
        conv_4b, _ = self.identity_block(conv_4a, 3, [256, 256, 512], stage=4, block='b')
        print("in_4----- ",conv_4b.shape)

        # encoder Stage 5
        conv_5a, _ = self.convolutional_block(conv_4b, f=3, filters=[256, 256, 512], stage=5, block='a', s=2)
        conv_5b, _ = self.identity_block(conv_5a, 3, [256, 256, 512], stage=5, block='b')
        print("in_5----- ",conv_5b.shape)


        conv_6a = Conv2D(1024, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv_5b)
        conv_6a = Activation('relu')(conv_6a)
        # Conv 6b (batch, 6, 12, 1024)
        conv_6b = Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_6a)
        conv_6b = Activation('relu')(conv_6b)
        print("in_6----- ",conv_6b.shape)
        # self.conv_6b = conv_6b
        # Prediction_Loss 6 (batch, 6, 12, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_6b)
        prediction = Activation('relu')(prediction)
        # print("not 6---- ",prediction.shape)
        pre_1 = prediction

        upconv_5 = Conv2DTranspose(512, kernel_size=(4, 4), strides=(2, 2), padding='same')(conv_6b)
        upconv_5 = BatchNormalization(axis=-1)(upconv_5)
        upconv_5 = Activation('relu')(upconv_5)
        upconv_5 = Cropping2D(cropping=((1, 0), (1, 0)), data_format=None)(upconv_5)
        # Iconv 5 (batch, 12, 24, 512)
        c = Concatenate(axis=-1)([upconv_5, conv_5b])
        iconv_5 = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
        iconv_5 = Activation('relu')(iconv_5)
        # Prediction_Loss 5 (batch, 12, 24, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_5)
        prediction = Activation('relu')(prediction)
        print("5---- ",prediction.shape)
        pre_2 = prediction
        

        # Upconv 4 (batch, 24, 48, 256)
        upconv_4 = Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), padding='same')(iconv_5)
        upconv_4 = BatchNormalization(axis=-1)(upconv_4)
        upconv_4 = Activation('relu')(upconv_4)
        # Iconv 4 (batch, 24, 48, 256)
        c = Concatenate(axis=-1)([upconv_4, conv_4b])
        iconv_4 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
        iconv_4 = Activation('relu')(iconv_4)
        # Prediction_Loss 4 (batch, 24, 48, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_4)
        prediction = Activation('relu', name='24x48')(prediction)
        print("4---- ",prediction.shape)
        pre_3 = prediction
        

        # Upconv 3 (batch, 48, 96, 128)
        upconv_3 = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same')(iconv_4)
        upconv_3 = BatchNormalization(axis=-1)(upconv_3)
        upconv_3 = Activation('relu')(upconv_3)
        # Iconv 3 (batch, 48, 96, 128)
        # upconv_3 = Cropping2D(cropping=((1, 0), (0, 0)), data_format=None)(upconv_3)
        c = Concatenate(axis=-1)([upconv_3, conv_3b])
        iconv_3 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
        iconv_3 = Activation('relu')(iconv_3)
        # Prediction_Loss 3 (batch, 48, 96, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_3)
        prediction = Activation('relu', name='48x96')(prediction)
        print("3---- ",prediction.shape)
        pre_4 = prediction
        

        # Upconv 2 (batch, 96, 192, 64)
        upconv_2 = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='same')(iconv_3)
        upconv_2 = BatchNormalization(axis=-1)(upconv_2)
        upconv_2 = Activation('relu')(upconv_2)
        # Iconv 2 (batch, 96, 192, 64)
        upconv_2 = Cropping2D(cropping=((1, 0), (0, 0)), data_format=None)(upconv_2)
        c = Concatenate(axis=-1)([upconv_2, conv_2b])
        iconv_2 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
        iconv_2 = Activation('relu')(iconv_2)
        # Prediction_Loss 2 (batch, 96, 192, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_2)
        prediction = Activation('relu', name='96x192')(prediction)
        print("2---- ",prediction.shape)
        pre_5 = prediction
        

        # Upconv 1 (batch, 192, 384, 32)
        upconv_1 = Conv2DTranspose(32, kernel_size=(4, 4), strides=(2, 2), padding='same')(iconv_2)
        upconv_1 = BatchNormalization(axis=-1)(upconv_1)
        upconv_1 = Activation('relu')(upconv_1)
        # Iconv 1 (batch, 192, 384, 32)
        upconv_1 = Cropping2D(cropping=((1, 0), (0, 0)), data_format=None)(upconv_1)
        c = Concatenate(axis=-1)([upconv_1, conv_1])
        iconv_1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
        iconv_1 = Activation('relu')(iconv_1)
        # Prediction_Loss 1 (batch, 192, 384, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_1)
        prediction = Activation('relu', name='192x384')(prediction)
        print("1---- ",prediction.shape)
        pre_6 = prediction

        # Upconv 1 (batch, 192, 384, 32)
        upconv_0 = Conv2DTranspose(16, kernel_size=(4, 4), strides=(2, 2), padding='same')(iconv_1)
        upconv_0 = BatchNormalization(axis=-1)(upconv_0)
        upconv_0 = Activation('relu')(upconv_0)
        # Iconv 1 (batch, 192, 384, 32)
        iconv_0 = Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same')(upconv_0)
        iconv_0 = Activation('relu')(iconv_0)
        # Prediction_Loss 1 (batch, 192, 384, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_0)
        prediction = Activation('relu', name='final_layer')(prediction)
        print("0---- ",prediction.shape)
        pre_final = prediction

        Disp_ResNet_autoencoder = Model(inputs=inputs, outputs=[pre_5, pre_6 , pre_final], name='Disp_ResNet_autoencoder')
        # print(DispNet_autoencoder.summary())

        return  Disp_ResNet_autoencoder

    def res_50_disp_autoencoder(self, height, width, depth):
        inputs = Input(shape=(height, width, depth))
        model = ResNet50(weights='imagenet',include_top=False,input_shape=(height, width,3))
        # X = model(inputs, training=True)
        # model.summary()
        skip_1 = model.layers[4].output
        skip_2 = model.layers[38].output
        skip_3 = model.layers[80].output
        skip_4 = model.layers[142].output
        skip_5 = model.layers[174].output
        X = model.layers[-1].output


        conv_6a = Conv2D(1024, kernel_size=(3, 3), strides=(2, 2), padding='same')(X)
        conv_6a = Activation('relu')(conv_6a)
        # Conv 6b (batch, 6, 12, 1024)
        conv_6b = Conv2D(1024, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_6a)
        conv_6b = Activation('relu')(conv_6b)

        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(conv_6b)
        prediction = Activation('sigmoid')(prediction)
        # print("not 6---- ",prediction.shape)
        pre_1 = prediction
        print("6---- ",prediction.shape)

        upconv_5 = Conv2DTranspose(512, kernel_size=(4, 4), strides=(2, 2), padding='same')(conv_6b)
        upconv_5 = BatchNormalization(axis=-1)(upconv_5)
        upconv_5 = Activation('relu')(upconv_5)
        upconv_5 = Cropping2D(cropping=((1, 0), (1, 0)), data_format=None)(upconv_5)
        # Iconv 5 (batch, 12, 24, 512)

        c = Concatenate(axis=-1)([upconv_5, skip_5])
        iconv_5 = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
        iconv_5 = Activation('relu')(iconv_5)
        # Prediction_Loss 5 (batch, 12, 24, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_5)
        prediction = Activation('sigmoid')(prediction)
        print("5---- ",prediction.shape)
        pre_2 = prediction
        

        # Upconv 4 (batch, 24, 48, 256)
        upconv_4 = Conv2DTranspose(512, kernel_size=(4, 4), strides=(2, 2), padding='same')(iconv_5)
        upconv_4 = BatchNormalization(axis=-1)(upconv_4)
        upconv_4 = Activation('relu')(upconv_4)
        # Iconv 4 (batch, 24, 48, 256)
        c = Concatenate(axis=-1)([upconv_4, skip_4])
        iconv_4 = Conv2D(512, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
        iconv_4 = Activation('relu')(iconv_4)
        # Prediction_Loss 4 (batch, 24, 48, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_4)
        prediction = Activation('sigmoid')(prediction)
        print("4---- ",prediction.shape)
        pre_3 = prediction
        

        # Upconv 3 (batch, 48, 96, 128)
        upconv_3 = Conv2DTranspose(256, kernel_size=(4, 4), strides=(2, 2), padding='same')(iconv_4)
        upconv_3 = BatchNormalization(axis=-1)(upconv_3)
        upconv_3 = Activation('relu')(upconv_3)
        # Iconv 3 (batch, 48, 96, 128)
        # upconv_3 = Cropping2D(cropping=((1, 0), (0, 0)), data_format=None)(upconv_3)
        c = Concatenate(axis=-1)([upconv_3, skip_3])
        iconv_3 = Conv2D(256, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
        iconv_3 = Activation('relu')(iconv_3)
        # Prediction_Loss 3 (batch, 48, 96, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_3)
        prediction = Activation('sigmoid')(prediction)
        print("3---- ",prediction.shape)
        pre_4 = prediction
        

        # Upconv 2 (batch, 96, 192, 64)
        upconv_2 = Conv2DTranspose(128, kernel_size=(4, 4), strides=(2, 2), padding='same')(iconv_3)
        upconv_2 = BatchNormalization(axis=-1)(upconv_2)
        upconv_2 = Activation('relu')(upconv_2)
        # Iconv 2 (batch, 96, 192, 64)
        upconv_2 = Cropping2D(cropping=((1, 0), (0, 0)), data_format=None)(upconv_2)
        c = Concatenate(axis=-1)([upconv_2, skip_2])
        iconv_2 = Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
        iconv_2 = Activation('relu')(iconv_2)
        # Prediction_Loss 2 (batch, 96, 192, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_2)
        prediction = Activation('sigmoid')(prediction)
        print("2---- ",prediction.shape)
        pre_5 = prediction
        

        # Upconv 1 (batch, 192, 384, 32)
        upconv_1 = Conv2DTranspose(64, kernel_size=(4, 4), strides=(2, 2), padding='same')(iconv_2)
        upconv_1 = BatchNormalization(axis=-1)(upconv_1)
        upconv_1 = Activation('relu')(upconv_1)
        # Iconv 1 (batch, 192, 384, 32)
        upconv_1 = Cropping2D(cropping=((1, 0), (0, 0)), data_format=None)(upconv_1)
        c = Concatenate(axis=-1)([upconv_1, skip_1])
        iconv_1 = Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same')(c)
        iconv_1 = Activation('relu')(iconv_1)
        # Prediction_Loss 1 (batch, 192, 384, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_1)
        prediction = Activation('sigmoid')(prediction)
        print("1---- ",prediction.shape)
        pre_6 = prediction

        # Upconv 1 (batch, 192, 384, 32)
        upconv_0 = Conv2DTranspose(32, kernel_size=(4, 4), strides=(2, 2), padding='same')(iconv_1)
        upconv_0 = BatchNormalization(axis=-1)(upconv_0)
        upconv_0 = Activation('relu')(upconv_0)
        # Iconv 1 (batch, 192, 384, 32)
        iconv_0 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same')(upconv_0)
        iconv_0 = Activation('relu')(iconv_0)
        # Prediction_Loss 1 (batch, 192, 384, 1)
        prediction = Conv2D(1, kernel_size=(3, 3), strides=(1, 1), padding='same')(iconv_0)
        prediction = Activation('sigmoid', name='final_layer')(prediction)
        print("0---- ",prediction.shape)
        pre_final = prediction


        new_model = Model(inputs=model.inputs, outputs= [pre_5, pre_6 , pre_final], name='Disp_ResNet_autoencoder')
        # print(new_model.summary())
        return new_model
        


