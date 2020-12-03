import os
import cv2

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
    size = (240, 240)
    pre_process_img = cv2.resize(image, size)

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

    train_images = image_list[0:train_size]
    test_images = image_list[train_size + 1:(list_len - 1)]
    train_labels = image_label[0:train_size]
    test_labels = image_label[train_size + 1:(list_len - 1)]

    return train_images, train_labels, test_images, test_labels


'''
Dic = PrepareDic(dataset_path)
Di, DB = Get_Breeds_info(dataset_path)
Image = Dic[DB[1]]

cv2.imshow(DB[1], Image[0])
cv2.waitKey()
cv2.destroyAllWindows()

'''
