import pandas as pd
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
import dataset_prep
import depth_prediction_net
import loss
import matplotlib.pyplot as plt
import cv2
import numpy as np
import time


width = 320
height = 180
batch_size = 32
save_width = width*4

name_1 = 'training-RGB-1596064718'
name_2 = 'training-depth-1596064718'

vidcap = cv2.VideoCapture(name_1+'.avi')
vidcap1 = cv2.VideoCapture(name_2+'.avi')
success,image = vidcap.read()
success1,image1 = vidcap1.read()


get_depth_net = depth_prediction_net.get_depth_net()
get_loss = loss.get_loss()

NAME = "result-{}".format(int(time.time()))
out0 = cv2.VideoWriter(NAME+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (save_width,height))

# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('model_1_ResNet_autoencoder.h5', custom_objects={
                                   'autoencoder_loss': get_loss.autoencoder_loss})


while (vidcap.isOpened() and vidcap1.isOpened()):
    success,image = vidcap.read()
    success1,image1 = vidcap1.read()
    if (success and success1):
        # RGB-image operation
        image = cv2.resize(image, (int(image.shape[1]/2), int(image.shape[0]/2)))
        # Depth-image operation
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image1 = cv2.resize(image1, (int(image1.shape[1]/2), int(image1.shape[0]/2)))
        distance_img =(image1/255)*6
        distance_img = np.where(distance_img>1.0, 1.0, distance_img)
        # Prediction
        img_array = np.array(image).reshape(-1, height, width, 3)
        img_array = np.float32(img_array / 255.)
        prediction = model.predict(img_array)
        prediction = prediction[0, :, :, 0]
        # Disp map
        disp = abs(distance_img - prediction)

        # cvt images to display
        distance_img = np.uint8(255 * distance_img)
        prediction = np.uint8(255 * prediction)
        disp = np.uint8(255 * disp)
        distance_img = cv2.cvtColor(distance_img, cv2.COLOR_GRAY2BGR)
        prediction = cv2.cvtColor(prediction, cv2.COLOR_GRAY2BGR)
        disp = cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)


        images = np.hstack((image, distance_img))
        images = np.hstack((images, prediction))
        images = np.hstack((images, disp))

        cv2.namedWindow('Align video', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Align video', images)

        out0.write(images)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

vidcap.release()
vidcap1.release()
out0.release()
cv2.destroyAllWindows()



# img = cv2.imread('RGB-3.jpg')
# img_array = cv2.resize(img, (width, height))
# img_array = np.array(img_array).reshape(-1, height, width, 3)
# img_array = np.float32(img_array / 255.)
# print(img_array.shape)

# # Recreate the exact same model, including its weights and the optimizer
# model = tf.keras.models.load_model('model_1_ResNet_autoencoder.h5', custom_objects={
#                                    'autoencoder_loss': get_loss.autoencoder_loss})
# prediction = model.predict(img_array)
# print(prediction[0, :, :, 0])
# print(prediction.shape)
# prediction = prediction[0, :, :, 0]

# img0 = cv2.imread('frame-3.jpg')

# # for depth image
# img0 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)
# img0 =(img0/255)*6
# img0 = np.where(img0>1.0, 1.0, img0)

# disp = abs(img0 - prediction)

# prediction = cv2.applyColorMap(cv2.convertScaleAbs(prediction, alpha=0.03), cv2.COLORMAP_JET)
# img0 = cv2.applyColorMap(cv2.convertScaleAbs(img0, alpha=0.03), cv2.COLORMAP_JET)
# disp = cv2.applyColorMap(cv2.convertScaleAbs(disp, alpha=0.03), cv2.COLORMAP_JET)
# window_name = 'predict'
# cv2.imshow(window_name, prediction)
# cv2.imshow('truth', img0)
# cv2.imshow('disp', disp)


# cv2.waitKey(0)
# cv2.destroyAllWindows()

