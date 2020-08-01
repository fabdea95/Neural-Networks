import cv2
import tensorflow as tf
import os
import matplotlib.pyplot as plt

CATEGORIES = ["car", "motocycle", "other"]
testset = []

def prepare(file):
    IMG_SIZE = 100
    img_array = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    img_array = img_array/255.0
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    cv2.imshow('image', new_array)
    cv2.waitKey()
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

model = tf.keras.models.load_model("CNN.model")
testdir = r"C:\Users\fabde\OneDrive\Desktop\Universita\testset2" #image path
#newimg = prepare(image)
for img in os.listdir(testdir):
    file = os.path.join(testdir, img)
    new_array = prepare(file)
    #testset.append([new_array])
    #print("prepare OK")

    prediction = model.predict([new_array])
    prediction = list(prediction[0])
    maxpred = CATEGORIES[prediction.index(max(prediction))]
    #cv2.imshow('maxpred', new_array)
    #cv2.waitKey()
    correct = 0
    print("FILE: ", img, "\tPREDICTION: ", CATEGORIES[prediction.index(max(prediction))], "\tscore: ", max(prediction) )
    #scelta = input("Correct? (s|n)\t")
    #if scelta == 's' :
    #    correct = correct + 1
