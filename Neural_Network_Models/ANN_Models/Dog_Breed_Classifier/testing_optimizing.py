import pickle

import tensorflow as tf
import cv2
import numpy as np
import os

import random

from Neural_Network_Models.ANN_Models.Dog_Breed_Classifier import Image_Perprocessing as ipp

dataset_path = "C:/Projects/PycharmProjects/Neural_Network/Dataset/Dog_Breeds/images/Images"
model = tf.keras.models.load_model("Dog_breed_classifier_v4.h5")

breed_index_dic = pickle.load(open("breed_index_dic.p", "rb"))
image_list = pickle.load(open("image_list.p", "rb"))
image_breed_index_list = pickle.load(open("image_breed_index_list.p", "rb"))

train_images, train_labels, test_images, test_labels = ipp.split_train_test_data(image_list, image_breed_index_list,
                                                                                 0.70)
# random testing
Right = 0
for test in range(0, 100):

    # img_path = 'C:/Projects/PycharmProjects/Neural_Network/Dataset/Dog_Breeds/images/Images'  #/n02087046-toy_terrier/n02087046_2279.jpg'
    folder_list = os.listdir(dataset_path)
    random_breed = random.randint(0, len(folder_list) - 1)
    folder_list[random_breed]
    breed_image_list = os.listdir(dataset_path + '/' + folder_list[random_breed])
    random_image_no = random.randint(0, len(breed_image_list) - 1)
    img_path = dataset_path + '/' + folder_list[random_breed] + '/' + breed_image_list[random_image_no]

    img = cv2.imread(img_path)
    img = cv2.resize(img, (120, 120))
    img = np.array(img)
    # model.summary()
    prediction = model.predict(np.array([img, ]))
    # print(prediction)
    # print(np.argmax(prediction))

    dog_index, DB = ipp.get_breeds_info(dataset_path)

    if np.argmax(prediction) == random_breed:
        print('true')
        Right += 1
    else:
        print('false')
        print('predicted = ', DB[np.argmax(prediction)])
        print('suppose to predict = ', DB[random_breed])

print('Accuracy = ' + str(Right) + "%")

