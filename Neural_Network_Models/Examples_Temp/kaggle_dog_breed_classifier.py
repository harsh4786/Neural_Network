import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input, EfficientNetB3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense, Conv2D

from sklearn.model_selection import train_test_split

# fpath = 'C:/Projects/pycharm projects/Neural_Network/Neural_Network/DATASETS/images/Images'
fpath = 'C:/Projects/PycharmProjects/Neural_Network/Dataset/Dog_Breeds/images/Images'
categories = os.listdir(fpath)
imgsize = 150
X = []
Y = []

# he kernel would not allow me to run the full 120 class dataset. So I had to compromise with 40
# I think the accuracy for the full dataset would fall somewhere around the 80% mark
for index, folder in enumerate(categories[:40]):
    path = os.path.join(fpath, folder)
    for img in tqdm(os.listdir(path)):
        Ipath = os.path.join(path, img)
        img = cv2.imread(Ipath, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (imgsize, imgsize))
        X.append(np.array(img))
        Y.append(index)

X = np.array(X)
Y = np.array(Y)
print(X.shape)
print(Y.shape)
labels = np_utils.to_categorical(Y, 40)

trainImg, testImg, trainLabel, testLabel = train_test_split(X, labels, test_size=0.3, random_state=69)
labels = None
X = None

augs_gen = ImageDataGenerator(preprocessing_function=preprocess_input, featurewise_center=False,
                              samplewise_center=False,
                              featurewise_std_normalization=False,
                              samplewise_std_normalization=False,
                              zca_whitening=False,
                              rotation_range=10,
                              zoom_range=0.1,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              horizontal_flip=True,
                              vertical_flip=False)
augs_gen.fit(trainImg)

base_model = EfficientNetB3(include_top=False,
                            input_shape=(imgsize, imgsize, 3),
                            weights='imagenet')

base_model.trainable = False

model = Sequential([base_model])
# model.add(Conv2D(10, (3,3) , activation = 'relu'))
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.5))
model.add(Dense(40, activation='softmax'))
model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
h = model.fit(augs_gen.flow(trainImg, trainLabel), epochs=20, verbose=1, batch_size=250,
              validation_data=(preprocess_input(testImg), testLabel))

'''
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(h.history['accuracy'], label='Train')
plt.plot(h.history['val_accuracy'], label='Validation')
plt.title('Accuracy')
plt.legend(loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(h.history['loss'], label='Train')
plt.plot(h.history['val_loss'], label='Validation')
plt.title('Loss')
plt.legend(loc='upper left')
plt.show()

print(100 * np.max(h.history['val_accuracy']))
model.save('SavedEfficientNetB3.h5')
'''
