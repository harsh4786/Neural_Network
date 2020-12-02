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


prepare_dic(dataset_path)

'''
Dic = PrepareDic(dataset_path)
Di, DB = Get_Breeds_info(dataset_path)
Image = Dic[DB[1]]

cv2.imshow(DB[1], Image[0])
cv2.waitKey()
cv2.destroyAllWindows()

'''
