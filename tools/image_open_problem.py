import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

"""
When testing trained model using opencv io functions, the test accuracy is lower than 
validation accuracy by a large margin, this script is used to find the reason for that.

==== tf.image.decode_images return RGB format image array 
==== cv2.imread open images to BGR format by default.

"""


def open_image_tf(image_path):
    image_mean = tf.constant([121.55213, 113.84197, 99.5037], dtype=tf.float32)
    path = tf.placeholder(tf.string)
    img_string = tf.read_file(path)
    img_decode = tf.image.decode_jpeg(img_string, channels=3)
    img_resize = tf.image.resize_images(img_decode, (224, 224))
    img_centered = tf.subtract(img_resize, image_mean)
    # img_centered = img_resize

    with tf.Session() as sess:
        img = sess.run(img_centered, feed_dict={path: image_path})
    return img


def open_image_tfcv(image_path):
    path = tf.placeholder(tf.string)
    img_string = tf.read_file(path)

    with tf.Session() as sess:
        buffer = sess.run(img_string, feed_dict={path: image_path})

    image_mean = np.array([121.55213, 113.84197, 99.5037])
    buffer = np.asarray(bytearray(buffer), dtype='uint8')
    img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), cv2.INTER_LINEAR)
    img = img - image_mean
    return img


def open_image_cv2(image_path):
    image_mean = np.array([121.55213, 113.84197, 99.5037])
    # image_mean = np.array([99.5037, 113.84197, 121.55213])
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224), cv2.INTER_LINEAR)
    img = img - image_mean
    return img


def open_image_pil(image_path):
    img = Image.open(image_path, mode='r')
    return img


if __name__ == '__main__':
    img_path = r'D:\normal_004837.jpg'
    img_tf = open_image_tf(img_path)
    img_cv = open_image_cv2(img_path)
    img_cvtf = open_image_tfcv(img_path)
    img_pil = open_image_pil(img_path)
    print(np.max(img_tf))
    print(np.min(img_tf))
    print(img_tf[0, 0, 0])
    print('*' * 40)
    print(np.max(img_cv))
    print(np.min(img_cv))
    print(img_cv[0, 0, 0])
    print('*' * 40)
    print(np.max(img_cvtf))
    print(np.min(img_cvtf))
    print('*' * 40)
    print(np.max(img_pil))
    print(np.min(img_pil))
