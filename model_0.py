# coding:utf-8
# 模型: 0

from data_IO import *
import tensorflow as tf
from tensorflow.contrib import layers
import numpy as np
import time


def model(_image, _batch_size, _block_size):
    noise_stddev = tf.random_uniform([_batch_size, 1, 1, 1], 0.015, 0.1, name="noise_stddev")
    noise = tf.random_normal([_batch_size, _block_size, _block_size, 1], 0.0, noise_stddev, name="noise")
    clear_image = tf.divide(tf.cast(_image, tf.float32), 255.0, "clear_image")
    noisy_image = tf.clip_by_value(clear_image + noise, 0.0, 1.0, "noisy_image")
    regularizer = tf.contrib.layers.l2_regularizer(0.001)
    h_0 = layers.conv2d(noisy_image, 24, 5, activation_fn=None, weights_regularizer=regularizer, biases_regularizer=regularizer)
    h_1 = layers.conv2d(h_0, 48, 5, activation_fn=tf.nn.leaky_relu, normalizer_fn=tf.contrib.layers.layer_norm, weights_regularizer=regularizer, biases_regularizer=regularizer)
    h_2 = layers.conv2d(h_1, 24, 5, activation_fn=tf.nn.leaky_relu, normalizer_fn=tf.contrib.layers.layer_norm, weights_regularizer=regularizer, biases_regularizer=regularizer)
    output_noise = layers.conv2d(h_2, 1, 5, activation_fn=None, weights_regularizer=regularizer, biases_regularizer=regularizer)
    output_clear_image = noisy_image - output_noise
    loss = tf.losses.mean_squared_error(output_clear_image, clear_image) + tf.contrib.layers.apply_regularization(regularizer, tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    return loss, noise, noisy_image, output_noise, clear_image, tf.clip_by_value(output_clear_image, 0.0, 1.0)


def train(_batch_size, _block_size, _channel, step_start=0, restore=False):
    image = read_tfrecords("data", _block_size, _channel)
    image = tf.train.shuffle_batch([image], _batch_size, capacity=2000, min_after_dequeue=1000)
    m_loss, m_noise, m_noisy_image, m_output_noise, m_clear_image, m_output_clear_image = model(image, _batch_size, _block_size)
    tf.summary.scalar("loss", m_loss)
    merged_summary_op = tf.summary.merge_all()
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(learning_rate=0.0002, global_step=global_step, decay_steps=100, decay_rate=0.90)
    op = tf.train.AdamOptimizer(learning_rate).minimize(m_loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners(sess=sess)
    summary_writer = tf.summary.FileWriter("./log", sess.graph)
    saver = tf.train.Saver(max_to_keep=10)
    if restore:
        saver.restore(sess, "./model/model.ckpt-%d" % step_start)
    for step in range(step_start, 100000):
        _, v_loss = sess.run([op, m_loss], feed_dict={global_step: step})
        if step % 10 == 0:
            print("step %6d: loss = %f" % (step, v_loss))
            summary_writer.add_summary(sess.run(merged_summary_op), step)
        if step % 100 == 0:
            v_noise, v_noisy_image, v_clear_image, v_output_clear_image = sess.run([m_noise, m_noisy_image, m_clear_image, m_output_clear_image])
            sample_save(v_noise[0], v_noisy_image[0], v_clear_image[0], v_output_clear_image[0], "sample/%04d.jpg" % step, _block_size, 1)
            saver.save(sess, "./model/model.ckpt", global_step=step)
    summary_writer.close()
    sess.close()


def test(_image, step):
    def test_model():
        noisy_image_input = tf.placeholder(tf.uint8, [None, None, None, 1])
        noisy_image = tf.divide(tf.cast(noisy_image_input, tf.float32), 255.)
        h_0 = layers.conv2d(noisy_image, 24, 5, activation_fn=None)
        h_1 = layers.conv2d(h_0, 48, 5, activation_fn=tf.nn.leaky_relu, normalizer_fn=tf.contrib.layers.layer_norm)
        h_2 = layers.conv2d(h_1, 24, 5, activation_fn=tf.nn.leaky_relu, normalizer_fn=tf.contrib.layers.layer_norm)
        output_noise = layers.conv2d(h_2, 1, 5, activation_fn=None)
        output_clear_image = tf.clip_by_value(noisy_image - output_noise, 0.0, 1.0)
        return noisy_image_input, output_clear_image

    if _image.mode not in ["RGB", "L"]:
        _image = _image.convert("RGB")
    img_data = np.array(_image).transpose((2, 0, 1))[..., np.newaxis]
    m_noisy_image_input, m_output_clear_image = test_model()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, "./model/model.ckpt-%d" % step)
    start_time = time.time()
    v_output_clear_image = sess.run(m_output_clear_image, feed_dict={m_noisy_image_input: img_data})
    stop_time = time.time()
    print("time_cost: %f" % (stop_time - start_time))
    out_image = v_output_clear_image.reshape(v_output_clear_image.shape[0:3]).transpose((1, 2, 0))
    if out_image.shape[2] == 1:
        out_image = out_image.reshape(out_image.shape[0:2])
    out_image = np.multiply(out_image, 255.).astype(np.uint8)
    sess.close()
    return Image.fromarray(out_image)


def add_noise(_image, noise_stddev=None):
    img_data = np.divide(np.array(_image).astype(np.float32), 255.)
    if noise_stddev is None:
        noise_stddev = np.random.uniform(0.015, 0.1)
    print("noise stddev is %d" % np.round(noise_stddev * 255.).astype(np.int))
    noise = np.random.normal(0.0, noise_stddev, img_data.shape)
    noise_image = np.clip(np.add(img_data, noise), 0.0, 1.0)
    return Image.fromarray(np.multiply(noise_image, 255.).astype(np.uint8))


if __name__ == '__main__':
    sel = 2
    if sel == 1:
        batch_size = 64
        block_size = 128
        channel = 1
        train(batch_size, block_size, channel)
    else:
        image = Image.open("test/00000014.jpg")
        noisy_image = add_noise(image, 25. / 255.)
        output = test(noisy_image, 1900)
        noisy_image.save("test/noisy_image.jpg")
        output.save("test/output_fcn.jpg")
        print("finished")

