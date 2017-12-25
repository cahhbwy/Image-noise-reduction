# coding:utf-8
# 模型: 0

from utils import *
from data_IO import *
import tensorflow as tf


def model(_image, _batch_size, _block_size, _channel):
    noise_stddev = tf.random_uniform([_batch_size, 1, 1, 1], 0.0, 0.3, name="noise_stddev")
    influence_range = tf.random_uniform([_batch_size, 1, 1, 1], 0.0, 1.0, name="influence_range")
    influence = tf.cast(tf.greater(tf.random_uniform([_batch_size, _block_size, _block_size, _channel], 0.0, 1.0), influence_range), tf.float32)
    noise = tf.random_normal([_batch_size, _block_size, _block_size, _channel], 0.0, noise_stddev, name="noise")
    noise = tf.multiply(noise, influence)
    clear_image = tf.divide(tf.cast(_image, tf.float32), 255.0, "clear_image")
    noisy_image = tf.clip_by_value(clear_image + noise, 0.0, 1.0, "noisy_image")
    h_0 = conv2d(noisy_image, 16, 5, 5, (1, 1, 1, 1), "conv2d_0")
    h_1 = lrelu(tf.contrib.layers.layer_norm(conv2d(h_0, 48, 5, 5, (1, 1, 1, 1), "conv2d_1"), scope="ln_1"))
    h_2 = lrelu(tf.contrib.layers.layer_norm(conv2d(h_1, 96, 5, 5, (1, 1, 1, 1), "conv2d_2"), scope="ln_2"))
    h_3 = lrelu(tf.contrib.layers.layer_norm(conv2d(h_2, 36, 5, 5, (1, 1, 1, 1), "conv2d_3"), scope="ln_3"))
    output_noise = tf.contrib.layers.layer_norm(conv2d(h_3, _channel, 5, 5, (1, 1, 1, 1), "conv2d_4"), scope="ln_4")
    output_clear_image = noisy_image - output_noise
    diff = output_clear_image - clear_image
    loss = tf.reduce_mean(tf.pow(diff, 2) + tf.abs(diff) + tf.multiply(tf.multiply(tf.abs(diff), tf.cast(tf.less(influence, 0.5), tf.float32)), 2.0))
    return loss, noise, noisy_image, output_noise, clear_image, output_clear_image


if __name__ == '__main__':
    batch_size = 64
    block_size = 128
    channel = 3
    image = read_tfrecords("data", block_size, channel)
    image = tf.train.shuffle_batch([image], batch_size, capacity=2000, min_after_dequeue=1000)
    m_loss, m_noise, m_noisy_image, m_output_noise, m_clear_image, m_output_clear_image = model(image, batch_size, block_size, channel)
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.0001, global_step=global_step, decay_steps=100, decay_rate=0.90)
    op = tf.train.AdamOptimizer(learning_rate).minimize(m_loss)
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)
    saver = tf.train.Saver(max_to_keep=10)
    for step in range(100000):
        _, v_loss = sess.run([op, m_loss], feed_dict={global_step: step})
        if step % 10 == 0:
            print("step %6d: loss = %f" % (step, v_loss))
        if step % 100 == 0:
            v_noise, v_noisy_image, v_clear_image, v_output_clear_image = sess.run([m_noise, m_noisy_image, m_clear_image, m_output_clear_image])
            sample_save(v_noise[0], v_noisy_image[0], v_clear_image[0], v_output_clear_image[0], "sample/%04d.jpg" % step, block_size)
            saver.save(sess, "./model/model.ckpt", global_step=step)
