import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2  # Can be download in pycham by adding "opencv-python" library

from time import time
from tensorflow.python.keras.callbacks import TensorBoard

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

load_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
load_model.compile(optimizer='adam',
                   loss=loss_fn,
                   metrics=['accuracy'])

checkpoint_path = "traning_1/cp.ckpt"
load_model.load_weights(checkpoint_path)

load_model.evaluate(x_test, y_test, verbose=2)

# Issue of loading full model most probabaly the input shape is the issue

new_model = tf.keras.models.load_model('my_mnist_model.h5')
new_model.summary()

print('restore model accuracy ')
new_model.evaluate(x_train, y_train)

