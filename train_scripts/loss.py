import pandas as pd
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras import backend as K
from tensorflow.keras.losses import mse


class get_loss():
    def autoencoder_loss(self, depth_img, output):
        # Compute error in reconstruction
        reconstruction_loss = mse(K.flatten(depth_img), K.flatten(output))

        dy_true, dx_true = tf.image.image_gradients(depth_img)
        dy_pred, dx_pred = tf.image.image_gradients(output)
        term3 = K.mean(K.abs(dy_pred - dy_true) +
                       K.abs(dx_pred - dx_true), axis=-1)

        tv = (1e-8)*tf.reduce_sum(tf.image.total_variation(output))

        total_loss = 100*reconstruction_loss + term3 + tv
        # total_loss = tv
        return total_loss
