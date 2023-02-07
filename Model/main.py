import cv2
import numpy as np
import tensorflow as tf
import os
from matplotlib import pyplot as plt

if __name__ == "__main__":

    data = tf.keras.utils.image_dataset_from_directory("img")
    data_iterator = data.as_numpy_iterator()
    batch = data_iterator.next()

    fig, ax = plt.subplots(ncols=4, figsize=(20,20))
    for idx, img in enumerate(batch[0][:4]):
        ax[idx].imshow(img.astype(int))
        ax[idx].title.set_text(batch[1][idx])
    
    plt.show()