import os

dataset_path = "C:/Projects/PycharmProjects/Neural_Network/Dataset/Dog_Breeds/images/Images"


def Get_Breeds(dataset_path):
    Breed_Folder_names = os.listdir(dataset_path)
    for x in Breed_Folder_names:
        print(x)


Get_Breeds(dataset_path)
