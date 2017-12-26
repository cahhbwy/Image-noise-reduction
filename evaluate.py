# coding:utf-8
import tensorflow as tf
from PIL import Image
import numpy as np

sess = tf.Session()
sess.run(tf.global_variables_initializer())


def judge(img_image, img_origin):
    global sess
    image_input = tf.placeholder(tf.uint8, [None, None, None])
    origin_input = tf.placeholder(tf.uint8, [None, None, None])
    image = tf.cast(image_input, tf.float32)
    origin = tf.cast(origin_input, tf.float32)
    s = tf.reduce_sum(tf.square(image))
    n = tf.reduce_sum(tf.square(image - origin))
    mse = tf.reduce_mean(tf.square(image - origin))
    snr = 10.0 * tf.log(s / n) / tf.log(10.0)
    psnr = 10.0 * tf.log(255.0 * 255.0 / mse) / tf.log(10.0)
    image_data = np.array(img_image)
    if len(image_data.shape) == 2:
        image_data = image_data[..., np.newaxis]
    image_origin_data = np.array(img_origin)
    if len(image_origin_data.shape) == 2:
        image_origin_data = image_origin_data[..., np.newaxis]
    v_snr, v_psnr, v_mse = sess.run([snr, psnr, mse], feed_dict={image_input: image_data, origin_input: image_origin_data})
    return v_snr, v_psnr, v_mse


if __name__ == '__main__':
    origin = Image.open("test/test.jpg")
    noisy_image = Image.open("test/noisy_image.jpg")
    print("noisy_image, SNR = %f, PSNR = %f, MSE = %f" % judge(noisy_image, origin))

    fcn = Image.open("test/output_fcn.jpg")
    bm3d = Image.open("test/output_bm3d.jpg")
    blur = Image.open("test/output_blur.jpg")
    gaussian = Image.open("test/output_gaussian.jpg")

    print("fcv     , SNR = %f, PSNR = %f, MSE = %f" % judge(fcn, origin))
    print("bm3d    , SNR = %f, PSNR = %f, MSE = %f" % judge(bm3d, origin))
    print("blur    , SNR = %f, PSNR = %f, MSE = %f" % judge(blur, origin))
    print("gaussian, SNR = %f, PSNR = %f, MSE = %f" % judge(gaussian, origin))
