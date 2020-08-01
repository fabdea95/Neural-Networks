import numpy as np
import os
from matplotlib import pyplot as plt
import cv2
import random
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

file_list = []
class_list = []

DATADIR = r"C:\Users\fabde\OneDrive\Desktop\Universita\imgclassCNN\dataset"
#All categories:
CATEGORIES = ["car", "motocycle", "other" ]

#size of the images
IMG_SIZE = 100

#Checking all images in the data folder
for category in CATEGORIES :
    path =os.path.join(DATADIR, category)
    print("", path)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)

training_data = []


def create_training_data():
    for category in CATEGORIES:
        path =os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)
        i = 0
        for img in os.listdir(path):
            try :
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                training_data.append([new_array, class_num])
                i = i+1
                #print("\n", i)
            except Exception as e:
                pass

create_training_data()
random.shuffle(training_data)

def data_augmentation():
            image_gen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.5
                    )
            #ImageDataGenerator(rescale=1./255, horizontal_flip=True)
            for category in CATEGORIES:
                path =os.path.join(DATADIR, category)
                class_num = CATEGORIES.index(category)
                try :
                    train_data_gen = image_gen.flow_from_directory(batch_size=32,
                                                                    directory=path,
                                                                    shuffle=True,
                                                                    target_size=(IMG_SIZE, IMG_SIZE))
                except Exception as e:
                    pass
                try:
                    imgaug_array = cv2.imread(train_data_gen, cv2.IMREAD_GRAYSCALE)
                    training_data.append([imgaug_array, class_num])
                except Exception as e:
                    pass


data_augmentation()

X = []  #features
y = []  #labels


for features, label in training_data:
    X.append(features)
    y.append(label)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y)
print(X[1])
#Creating files contaianing all the info about the model
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)
