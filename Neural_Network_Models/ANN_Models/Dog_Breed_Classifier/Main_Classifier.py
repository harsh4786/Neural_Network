from Neural_Network_Models.ANN_Models.Dog_Breed_Classifier import Image_Perprocessing as ipp

dataset_path = "C:/Projects/PycharmProjects/Neural_Network/Dataset/Dog_Breeds/images/Images"

Dog_dic = ipp.prepare_dic(dataset_path)  # Return dictionary in which keys = 'Breed_Name' values = 'all images of dog'
breed_index_dic, image_list, image_breed_index_list = ipp.pre_process_load_data(Dog_dic)

print(breed_index_dic)
# -----------------------Start Building the Neural Network Model---------------------------- #

import tensorflow as tf
import cv2
import random
import numpy

# See Random Image
'''
while 'q' != str(input('for exit enter == q else for next image enter key')):
    n = random.randint(0, (len(image_list) - 1))
    cv2.imshow(str(n) + '-' + str(image_breed_index_list[n]), image_list[n])
    cv2.waitKey()
    cv2.destroyAllWindows()
'''

print(image_breed_index_list)
print(image_list[1].shape)

train_images, train_labels, test_images, test_labels = ipp.split_train_test_data(image_list, image_breed_index_list,
                                                                                 0.6)
