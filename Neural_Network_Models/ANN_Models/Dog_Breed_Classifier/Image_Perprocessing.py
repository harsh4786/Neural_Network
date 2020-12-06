import os
import cv2
import numpy as np

dataset_path = "C:/Projects/PycharmProjects/Neural_Network/Dataset/Dog_Breeds/images/Images"


def get_breeds_info(dataset_path):
    Dogindex = []
    DogBreed = []
    Breed_Folder_names = os.listdir(dataset_path)
    for x in Breed_Folder_names:
        y = x.split('-')
        Dogindex.append(y[0])
        DogBreed.append(y[1])
    return Dogindex, DogBreed


def prepare_dic(path):
    print("Warning Code consume too much RAM ...  ")
    print('----------- (⊙_⊙;)  (⊙_⊙;)  (⊙_⊙;) ------------')
    print('---------Started-----------')
    dogs_dic = {}
    breed_folder_names = os.listdir(path)
    loading = 0

    for x in breed_folder_names:
        folder_name = x.split('-')
        dog_breed_name = folder_name[1]
        img_path = os.listdir(path + '/' + x)
        images_list = []
        for y in img_path:
            img = cv2.imread(path + '/' + x + '/' + y)
            pre_process_img = pre_processing(img)
            images_list.append(pre_process_img)

        print('loadingdata' + '......' + str(
            int(((loading + 1) / len(breed_folder_names)) * 100)) + '%......' + dog_breed_name)
        loading += 1
        dogs_dic[dog_breed_name] = images_list
    return dogs_dic


def pre_processing(image):
    size = (60, 60)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pre_process_img = cv2.resize(image, size)
    pre_process_img = np.array(pre_process_img)

    '''
    ##----------To Look the preprocess Images----------##
    
    cv2.imshow('dog_pre_process_img', pre_process_img)
    cv2.waitKey()
    cv2.destroyAllWindows()   
    '''

    return pre_process_img


import random


def pre_process_load_data(dog_dic):
    # asssign values to dog breed
    i = 0
    breed_index_dic = {}
    image_list = []
    image_breed_index_list = []
    for breed_name in dog_dic.keys():
        breed_index_dic[i] = breed_name

        for images in dog_dic[breed_name]:
            image_list.append(images)
            image_breed_index_list.append(i)
        i += 1

    # suffleing data

    temp = list(zip(image_list, image_breed_index_list))
    random.shuffle(temp)
    image_list, image_breed_index_list = zip(*temp)

    return breed_index_dic, image_list, image_breed_index_list


def split_train_test_data(image_list, image_label, train_size_ratio):
    list_len = len(image_list)
    train_size = int(train_size_ratio * list_len)

    train_images = np.array(image_list[0:train_size])
    test_images = np.array(image_list[train_size + 1:(list_len - 1)])
    train_label = np.array(image_label[0:train_size])
    test_label = np.array(image_label[train_size + 1:(list_len - 1)])
    '''
    train_labels = np.zeros([train_size, 120])
    test_labels = np.zeros([list_len - train_size - 1, 120])
    i = 0
    for x in train_label:
        train_labels[i][x] = 1
        i += 1
    i = 0
    for x in test_label:
        test_labels[i][x] = 1
        i += 1
    '''
    return train_images, train_label, test_images, test_label


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Activation, Flatten, MaxPool2D

'''
def import_model():
    model = Sequential()
    model.add(Conv2D(12, 4, (3, 3), input_shape=(60, 60,1)))  # shape = (240, 240, 3)
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Activation('softmax'))
    return model
import_model()
'''


def import_model():
    model = Sequential()
    model.add(Conv2D(12, 4, (3, 3), input_shape=(60, 60, 1)))  # shape = (240, 240, 3)
    model.add(Activation("relu"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(972))
    model.add(Activation('relu'))
    model.add(Dense(120))
    model.add(Activation('softmax'))
    return model


# import_model()
'''
Dic = PrepareDic(dataset_path)
Di, DB = Get_Breeds_info(dataset_path)
Image = Dic[DB[1]]

cv2.imshow(DB[1], Image[0])
cv2.waitKey()
cv2.destroyAllWindows()

'''
