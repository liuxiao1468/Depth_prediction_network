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

        im1 = tf.image.convert_image_dtype(depth_img, tf.float32)
        im2 = tf.image.convert_image_dtype(output, tf.float32)

        l_ssim = K.clip((1 - tf.image.ssim(im1, im2, max_val=1.0, 
            filter_size=11,filter_sigma=1.5, k1=0.01, k2=0.03)) * 0.5, 0, 1)

        dy_true, dx_true = tf.image.image_gradients(depth_img)
        dy_pred, dx_pred = tf.image.image_gradients(output)
        term3 = K.mean(K.abs(dy_pred - dy_true) +
                       K.abs(dx_pred - dx_true), axis=-1)

        tv = (1e-7)*tf.reduce_sum(tf.image.total_variation(output))

        total_loss = 100*reconstruction_loss + l_ssim + K.mean(term3) + tv
        # total_loss = tv
        return total_loss

class get_finetune_loss():
    def autoencoder_loss(self, depth_img_1, depth_img_2, output_1, output_2, mask_1, mask_2):
        # pred_1 = tf.multiply(depth_img_1, mask_1)
        # pred_2 = tf.multiply(depth_img_2, mask_1)
        # out_1 = tf.multiply(output_1, mask_1)
        # out_2 = tf.multiply(output_2, mask_1)
        pred_1 = depth_img_1
        pred_2 = depth_img_2
        out_1 = output_1
        out_2 = output_2


        # Compute error in reconstruction
        reconstruction_loss = mse(K.flatten(pred_1), K.flatten(out_1))

        dy_true, dx_true = tf.image.image_gradients(pred_1)
        dy_pred, dx_pred = tf.image.image_gradients(out_1)
        term3 = K.mean(K.abs(dy_pred - dy_true) +
                       K.abs(dx_pred - dx_true), axis=-1)
        im1 = tf.image.convert_image_dtype(pred_1, tf.float32)
        im2 = tf.image.convert_image_dtype(out_1, tf.float32)

        l_ssim = K.clip((1 - tf.image.ssim(im1, im2, max_val=1.0, 
            filter_size=11,filter_sigma=1.5, k1=0.01, k2=0.03)) * 0.5, 0, 1)
        tv = (1e-8)*tf.reduce_sum(tf.image.total_variation(depth_img_1))
        loss_base_1 = 10*reconstruction_loss + l_ssim + K.mean(term3) +tv



        # Compute error in reconstruction
        reconstruction_loss = mse(K.flatten(pred_2), K.flatten(out_2))

        dy_true, dx_true = tf.image.image_gradients(pred_2)
        dy_pred, dx_pred = tf.image.image_gradients(out_2)
        term3 = K.mean(K.abs(dy_pred - dy_true) +
                       K.abs(dx_pred - dx_true), axis=-1)
        im1 = tf.image.convert_image_dtype(pred_2, tf.float32)
        im2 = tf.image.convert_image_dtype(out_2, tf.float32)

        l_ssim = K.clip((1 - tf.image.ssim(im1, im2, max_val=1.0, 
            filter_size=11,filter_sigma=1.5, k1=0.01, k2=0.03)) * 0.5, 0, 1)
        tv = (1e-8)*tf.reduce_sum(tf.image.total_variation(depth_img_2))
        loss_base_2 = 10*reconstruction_loss + l_ssim + K.mean(term3) +tv




        # loss with mask
        mask1 = tf.multiply(depth_img_1, mask_1)
        mask2 = tf.multiply(depth_img_2, mask_1)

        im1 = tf.image.convert_image_dtype(mask1, tf.float32)
        im2 = tf.image.convert_image_dtype(mask2, tf.float32)
        l_ssim = K.clip((1 - tf.image.ssim(im1, im2, max_val=1.0, 
            filter_size=11,filter_sigma=1.5, k1=0.01, k2=0.03)) * 0.5, 0, 1)
        # ssim2 = K.mean(ssim2,axis=-1)

        L_mse = mse(K.flatten(mask1), K.flatten(mask2))
        alpha = 0.7
        loss_consistency = alpha * l_ssim + (1- alpha) * L_mse 

        total_loss = 0.1* loss_consistency + (loss_base_1+loss_base_2)

        # total_loss = term3

        return total_loss