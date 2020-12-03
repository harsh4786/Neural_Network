import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
DATADIR = "C:/Projects/pycharm projects/Neural_Network/Neural_Network/DATASETS/images/Images"
CATEGORIES = ["n02085620-Chihuahua", "n02085782-Japanese_spaniel", "n02085936-Maltese_dog", "n02086079-Pekinese", "n02086240-Shih-Tzu","n02086646-Blenheim_spaniel", "n02086910-papillon", "n02087046-toy_terrier", "n02087394-Rhodesian_ridgeback", "n02088094-Afghan_hound",
              "n02088238-basset", "n02088364-beagle", "n02088466-bloodhound", "n02088632-bluetick", "n02089078-black-and-tan_coonhound", "n02089867-Walker_hound", "n02089973-English_foxhound", "n02090379-redbone", "n02090622-borzoi", "n02090721-Irish_wolfhound",
              "n02091032-Italian_greyhound", "n02091134-whippet", "n02091244-Ibizan_hound", "n02091467-Norwegian_elkhound", "n02091635-otterhound", "n02091831-Saluki", "n02092002-Scottish_deerhound", "n02092339-Weimaraner", "n02093256-Staffordshire_bullterrier",
              "n02093428-American_Staffordshire_terrier", "n02093647-Bedlington_terrier", "n02093754-Border_terrier", "n02093859-Kerry_blue_terrier", "n02093991-Irish_terrier", "n02094114-Norfolk_terrier", "n02094258-Norwich_terrier",
              "n02094433-Yorkshire_terrier", "n02095314-wire-haired_fox_terrier", "n02095570-Lakeland_terrier", "n02095889-Sealyham_terrier", "n02096051-Airedale", "n02096177-cairn", "n02096294-Australian_terrier", "n02096437-Dandie_dinmont",
              "n02096585-Boston_bull", "n02097047-miniature_schnauzer", "n02097130-giant_schnauzer", "n02097209-Standard_schnauzer", "n02097298-Scotch_terrier", "n02097474-Tibetan_terrier", "n02097658-silky_terrier", "n02098105-soft_coated_wheaten_terrier",
              "n02098286-West_Highland_white_terrier", "n02098413-Lhasa","n02099267-FLat_coated_retriever", "n02099429-curly_coated_retriever", "n02099601-Golden_retriever", "n02099712-Labrador_retriever", "n02099849-Chesapeake_bay_retriever",
              "n02100236-German_short_haired_pointer", "n02100583-vizsla", "n02100735-English_setter", "n02100877-Irish_setter", "n02101006-Gordon_setter", "n02101388-Brittany_spaniel", "n02101556-clumber", "n02102040-English_springer",
              "n02102177-Welsh_springer_spaniel", "n02102318-cocker_spaniel", "n02102480-Sussex_spaniel", "n02102973-Irish_water_spaniel", "n02104029-kuvasz","n02104365-schipperke","n02105056-groenendael", "n02105162-malinois", "n02105251-briard", "n02105412-kelpie",
              "n02105505-komondor","n02105641-Old_English_sheepdog", "n02105855-Shetland_sheepdog", "n02106030-collie", "n02106166-border_collie", "n02106382-Bouvier_des_Flandres", "n02106550-Rottweiler", "n02106662-German_shepherd", "n02107142-Doberman",
              "n02107312-miniature_pinscher", "n02107574-Great_Swiss_mountain_dog", "n02107683-Bernese_mountain_dog","n02107908-Appenzeller","n02108000-Entlebucher", "n02108089-boxer","n02108422-bull_mastiff","n02108551-Tibetan_mastiff",
              "n02108915-French_bulldog", "n02109047-Great_dane", "n02109525-Saint_Bernard", "n02109961-Eskimo_dog", "n02110063-malamute", "n02110185-Siberian_husky", "n02110627-affenpinscher", "n02110806-basenji", "n02110958-pug", "n02111129-Leonberg", "n02111277-Newfoundland",
              "n02111500-Great_pyrenees", "n02111889-Samoyed", "n02112018-pomeranian", "n02112137-chow", "n02112350-keeshond", "n02112706-Brabancon_griffon", "n02113023-Pembroke", "n02113186-Cardigan", "n02113624-toy_poodle", "n02113712-miniature_poodle",
              "n02113799-standard_poodle", "n02113978-Mexican_hairless", "n02115641-dingo", "n02115913-dhole", "n02116738-African_hunting_dog"]

for category in CATEGORIES:
    path =  os.path.join(DATADIR, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_COLOR)
        plt.imshow(img_array, cmap="gray")
        plt.show()
        break
    break

IMG_SIZE = 150

training_data = []
def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
create_training_data()
random.shuffle(training_data)


x = []
y = []
for features, label in training_data:
    x.append(features)
    y.append(label)

x = np.array(x).reshape(-1,IMG_SIZE, IMG_SIZE, 3)

pickle_out = open("x.pickle", "wb")
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()


