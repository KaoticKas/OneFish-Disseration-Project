import cv2
import numpy as np
import tensorflow as tf
import os
from matplotlib import pyplot as plt

if __name__ == "__main__":

    tf.compat.v1.disable_eager_execution()
    with tf.device('/gpu:0'):
        a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[2, 3], name='a')
        b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape=[3, 2], name='b')
        c = tf.matmul(a, b)
    with tf.compat.v1.Session() as sess:
        print (sess.run(c))