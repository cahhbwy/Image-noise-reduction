# coding:utf-8
# 二次封装

import tensorflow as tf


def bias(name, shape, bias_start=0.0, trainable=True):
    return tf.get_variable(name, shape, tf.float32, trainable=trainable, initializer=tf.constant_initializer(bias_start, dtype=tf.float32))


def weight(name, shape, stddev=0.5, trainble=True):
    return tf.get_variable(name, shape, tf.float32, trainable=trainble, initializer=tf.random_normal_initializer(stddev=stddev, dtype=tf.float32))


def fully_connected(value, output_shape, name="fully_connected", with_w=False):
    shape = value.get_shape().as_list()
    with tf.variable_scope(name):
        weights = weight('weights', [shape[1], output_shape], 0.02)
        biases = bias('biases', [output_shape], 0.0)
    if with_w:
        return tf.matmul(value, weights) + biases, weights, biases
    else:
        return tf.matmul(value, weights) + biases


def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        return tf.maximum(x, leak * x, name=name)


def deconv2d(value, output_shape, k_h=5, k_w=5, strides=(1, 2, 2, 1), name='deconv2d', with_w=False):
    with tf.variable_scope(name):
        weights = weight('weights', [k_h, k_w, output_shape[-1], value.get_shape()[-1]])
        deconv = tf.nn.conv2d_transpose(value, weights, output_shape, strides=strides)
        biases = bias('biases', [output_shape[-1]])
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, weights, biases
        else:
            return deconv


def conv2d(value, output_dim, k_h=5, k_w=5, strides=(1, 2, 2, 1), name='conv2d'):
    with tf.variable_scope(name):
        weights = weight('weights', [k_h, k_w, value.get_shape()[-1], output_dim])
        conv = tf.nn.conv2d(value, weights, strides=strides, padding='SAME')
        biases = bias('biases', [output_dim])
        conv_shape = conv.get_shape().as_list()
        conv_shape[0] = -1
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv_shape)
        return conv


def conv_cond_concat(value, cond, name='concat'):
    value_shapes = value.get_shape().as_list()
    cond_shapes = cond.get_shape().as_list()
    with tf.variable_scope(name):
        return tf.concat(axis=3, values=[value, cond * tf.ones(value_shapes[0:3] + cond_shapes[3:])])


def batch_norm(value, is_train=True, name='batch_norm', epsilon=1e-5, momentum=0.9):
    with tf.variable_scope(name):
        ema = tf.train.ExponentialMovingAverage(decay=momentum)
        shape = value.get_shape().as_list()[-1]
        beta = bias('beta', [shape], bias_start=0.0)
        gamma = bias('gamma', [shape], bias_start=1.0)
        if is_train:
            batch_mean, batch_variance = tf.nn.moments(value, list(range(len(value.get_shape().as_list()) - 1)), name='moments')
            moving_mean = bias('moving_mean', [shape], 0.0, False)
            moving_variance = bias('moving_variance', [shape], 1.0, False)
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                ema_apply_op = ema.apply([batch_mean, batch_variance])
            assign_mean = moving_mean.assign(ema.average(batch_mean))
            assign_variance = moving_variance.assign(ema.average(batch_variance))
            with tf.control_dependencies([ema_apply_op]):
                mean, variance = tf.identity(batch_mean), tf.identity(batch_variance)
            with tf.control_dependencies([assign_mean, assign_variance]):
                return tf.nn.batch_normalization(value, mean, variance, beta, gamma, 1e-5)
        else:
            mean = bias('moving_mean', [shape], 0.0, False)
            variance = bias('moving_variance', [shape], 1.0, False)
            return tf.nn.batch_normalization(value, mean, variance, beta, gamma, epsilon)
