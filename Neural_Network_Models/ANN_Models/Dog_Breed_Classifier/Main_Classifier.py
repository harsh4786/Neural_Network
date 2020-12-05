# from ANN_Models.Dog_Breed_Classifier import Image_Perprocessing as ipp # Harsh
# from ANN_Models.Dog_Breed_Classifier import functions  # Harsh
from Neural_Network_Models.ANN_Models.Dog_Breed_Classifier import Image_Perprocessing as ipp  # Anil
from Neural_Network_Models.ANN_Models.Dog_Breed_Classifier import functions  # Anil

# dataset_path = "C:/Projects/pycharm projects/Neural_Network/Neural_Network/DATASETS/images/Images" # Harsh
dataset_path = "C:/Projects/PycharmProjects/Neural_Network/Dataset/Dog_Breeds/images/Images"  # Anil

Dog_dic = ipp.prepare_dic(dataset_path)  # Return dictionary in which keys = 'Breed_Name' values = 'all images of dog'
breed_index_dic, image_list, image_breed_index_list = ipp.pre_process_load_data(Dog_dic)

print(breed_index_dic)

# -----------------------Start Building the Neural Network Model---------------------------- #

import tensorflow as tf
from time import time
from tensorflow.python.keras.callbacks import TensorBoard
import cv2
import random
import numpy as np

# See Random Image
'''
while 'q' != str(input('for exit enter == q else for next image enter key')):
    n = random.randint(0, (len(image_list) - 1))
    cv2.imshow(str(n) + '-' + str(image_breed_index_list[n]), image_list[n])
    cv2.waitKey()
    cv2.destroyAllWindows()

'''

train_images, train_labels, test_images, test_labels = ipp.split_train_test_data(image_list, image_breed_index_list,
                                                                                 0.7)

print(train_images[1])
print(train_labels[1])
print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)
print(test_labels.shape)

model = ipp.import_model()

model.compile(optimizer='RMSprop',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/DogBreed/{}".format(time()))

model.fit(train_images, train_labels, epochs=25, callbacks=tensorboard)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nTest accuracy:', str(int(test_acc * 100)) + '%')

prediction = model.predict(np.array([image_list[52], ]))
print(prediction)
print(np.argmax(prediction))

model.save("Dog_breed_classifier_v4.h5")
