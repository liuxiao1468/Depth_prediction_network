import pandas as pd
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
import dataset_prep
import depth_prediction_net
import loss
import matplotlib.pyplot as plt
import cv2


dataset_train = '/home/leo/deeplearning/depth_prediction/train'
ground_truth = '/home/leo/deeplearning/depth_prediction/ground_truth'
width = 320
height = 180
batch_size = 32

get_dataset = dataset_prep.get_dataset()
get_depth_net = depth_prediction_net.get_depth_net()
get_loss = loss.get_loss()


def generate_and_save_images(model, test_sample):
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



test_sample, depth_sample = get_dataset.select_batch(dataset_train, ground_truth, 16)


# Recreate the exact same model, including its weights and the optimizer
model = tf.keras.models.load_model('model_1_ResNet_autoencoder.h5', custom_objects={
                                   'autoencoder_loss': get_loss.autoencoder_loss})
# model = tf.keras.models.load_model('U-net_depth.h5', custom_objects={'autoencoder_loss': autoencoder_loss})
# Show the model architecture
# model.summary()

generate_and_save_images(model, test_sample)


fig = plt.figure(figsize=(4, 4))

for i in range(test_sample.shape[0]):
    plt.subplot(4, 4, i + 1)
    plt.imshow(test_sample[i, :, :, :])
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
