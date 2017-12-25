# coding:utf-8
# 读取图像并制作适合tensorflow的TFRecords数据

import tensorflow as tf
import os
from PIL import Image
import random
import numpy
import multiprocessing


def create_tfrecord(_filename, _block_size, _data_block, _from_path="image", _save_path="data", _mode="RGB"):
    image_list = os.listdir(_from_path)
    len_image_list = len(image_list)
    if _mode is "RGB":
        channel = 3
    elif _mode is "L":
        channel = 1
    else:
        return
    writer = tf.python_io.TFRecordWriter(os.path.join(_save_path, _filename))
    count = 0
    while count < _data_block:
        image_index = random.randint(0, len_image_list - 1)
        try:
            image = Image.open(os.path.join(_from_path, image_list[image_index]))
        except OSError:
            continue
        if image.size[0] < _block_size or image.size[1] < _block_size:
            continue
        if image.mode is not _mode:
            image = image.convert(_mode)
        x = random.randint(0, image.size[0] - _block_size)
        y = random.randint(0, image.size[1] - _block_size)
        image_bytes = image.crop((x, y, x + _block_size, y + _block_size)).tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'block_size': tf.train.Feature(int64_list=tf.train.Int64List(value=[_block_size])),
            'channel': tf.train.Feature(int64_list=tf.train.Int64List(value=[channel])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
        }))
        writer.write(example.SerializeToString())
        count += 1
    print("create {} finished.".format(_filename))
    writer.close()


def create_tfrecords(block_size=128, total=65536, mode='RGB', from_path="image", save_path="data"):
    image_list = os.listdir(from_path)
    len_image_list = len(image_list)
    count = 0
    if mode is "RGB":
        channel = 3
    elif mode is "L":
        channel = 1
    else:
        return
    data_block = (192 * 1024 * 1024) // (block_size * block_size * channel)
    print("create data_%04d.tfrecords" % 0)
    writer = tf.python_io.TFRecordWriter(os.path.join(save_path, "%04d.tfrecords" % 0))
    while count < total:
        image_index = random.randint(0, len_image_list - 1)
        try:
            image = Image.open(os.path.join(from_path, image_list[image_index]))
        except OSError:
            continue
        if image.size[0] < block_size or image.size[1] < block_size:
            continue
        if image.mode is not mode:
            image = image.convert(mode)
        x = random.randint(0, image.size[0] - block_size)
        y = random.randint(0, image.size[1] - block_size)
        image_bytes = image.crop((x, y, x + block_size, y + block_size)).tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'block_size': tf.train.Feature(int64_list=tf.train.Int64List(value=[block_size])),
            'channel': tf.train.Feature(int64_list=tf.train.Int64List(value=[channel])),
            'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes]))
        }))
        if count > 0 and count % data_block == 0:
            writer.close()
            print("create data_%04d.tfrecords" % (count // data_block))
            writer = tf.python_io.TFRecordWriter(os.path.join(save_path, "%04d.tfrecords" % (count // data_block)))
        writer.write(example.SerializeToString())
        count += 1
    writer.close()


def create_tfrecords_multi_threads(block_size=128, total=65536, file_size=192, mode='RGB', from_path="image", save_path="data", limit_threads_num=None):
    if limit_threads_num is None:
        pool = multiprocessing.Pool()
    else:
        pool = multiprocessing.Pool(limit_threads_num)
    if mode is "RGB":
        channel = 3
    elif mode is "L":
        channel = 1
    else:
        return
    data_block = (file_size * 1024 * 1024) // (block_size * block_size * channel)
    count = total // data_block
    params = []
    for i in range(count):
        params.append(("%04d.tfrecords" % i, block_size, data_block, from_path, save_path, mode))
    pool.starmap(create_tfrecord, params)


def read_tfrecords(dir_path="data", block_size=128, channel=3):
    filename_list = [os.path.join(dir_path, filename) for filename in os.listdir(dir_path)]
    filename_queue = tf.train.string_input_producer(filename_list)
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(serialized_example, features={
        'block_size': tf.FixedLenFeature([], tf.int64),
        'channel': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)
    })
    data = tf.decode_raw(features['image_raw'], tf.uint8)
    return tf.reshape(data, (block_size, block_size, channel))


def sample_save(noise, noisy_image, image, output_clear_image, save_path, block_size=128, channel=3):
    if channel == 1:
        noise = numpy.reshape(noise, (block_size, block_size))
        noisy_image = numpy.reshape(noisy_image, (block_size, block_size))
        image = numpy.reshape(image, (block_size, block_size))
        output_clear_image = numpy.reshape(output_clear_image, (block_size, block_size))
        output = Image.new("L", (block_size * 2, block_size * 2))
    else:
        output = Image.new("RGB", (block_size * 2, block_size * 2))
    img_noise = Image.fromarray(numpy.clip(numpy.add(numpy.multiply(noise, 255), 127).astype(numpy.uint8), 0, 255))
    img_noisy_image = Image.fromarray(numpy.multiply(noisy_image, 255).astype(numpy.uint8))
    img_image = Image.fromarray(numpy.multiply(image, 255).astype(numpy.uint8))
    img_output_clear_image = Image.fromarray(numpy.multiply(output_clear_image, 255).astype(numpy.uint8))
    output.paste(img_noise, (0, 0, block_size, block_size))
    output.paste(img_noisy_image, (block_size, 0, block_size * 2, block_size))
    output.paste(img_image, (0, block_size, block_size, block_size * 2))
    output.paste(img_output_clear_image, (block_size, block_size, block_size * 2, block_size * 2))
    output.save(save_path, quality=100)


if __name__ == '__main__':
    # create_tfrecords(128, 65536, "L", "image", "data")
    create_tfrecords_multi_threads(128, 262144, 256, 'L', "image", "data")
