import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Activation, Flatten, MaxPool2D
def import_model(shape):
    model = Sequential()
    model.add(Conv2D(786), (3,3), shape ) #shape = (240, 240, 3)
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(786), (3, 3))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Conv2D(512), (3, 3))
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(512))

    model.add(Dense(120))
    model.add(Activation('sigmoid'))


    return model
