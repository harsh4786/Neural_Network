import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2  # Can be download in pycham by adding "opencv-python" library

from time import time
from tensorflow.python.keras.callbacks import TensorBoard

# The line must seen like this in terminal
# (Neural_Network_Env) C:\Projects\Pycharm\Neural_Network\Neural_Network\Neural_Network_Models>tensorboard --logdir=logs/MnistExample


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

img1 = np.array(x_train[2])

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
predictions

tf.nn.softmax(predictions).numpy()

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

# Implement tensorboard
tensorboard = TensorBoard(log_dir="logs/MnistExample/{}".format(time()))

# Saving the model using callbacks
# Link of tutorial
# https://www.youtube.com/watch?v=HxtBIwfy0kM&feature=emb_logo

checkpoint_path = "traning_1/cp.ckpt"
#checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True, verbose=1)
# Save on every 5 epoch
# cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, period=5)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=50, callbacks=[tensorboard, cp_callback])

model.evaluate(x_test, y_test, verbose=2)

probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

# Loading the Model

Unload_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

Unload_model.compile(optimizer='adam',
                     loss=loss_fn,
                     metrics=['accuracy'])

Unload_model.evaluate(x_test, y_test, verbose=2)

# Loading the Model and geeting weights

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

#  Manually without callbacks weights

model.save_weights("./checkpoints/my_checkpoint")

# restore the weihts

restore_model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
restore_model.compile(optimizer='adam',
                      loss=loss_fn,
                      metrics=['accuracy'])
print('manual restore ')
restore_model.evaluate(x_test, y_test)

# Save the entire Model
model.save('my_mnist_model.h5')

# Recreate the model
new_model = tf.keras.models.load_model('my_mnist_model.h5')
new_model.summary()

print('restore model accuracy ')
new_model.evaluate(x_test, y_test)
