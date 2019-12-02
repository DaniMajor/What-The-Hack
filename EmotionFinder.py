from __future__ import division
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import time
import cv2
import matplotlib
from tkinter.filedialog import askdirectory
import shutil
import os

emotionfile = open('Emotion.json', 'r')
emotionloaded = emotionfile.read()
emotionfile.close()
emotionModel = model_from_json(emotionloaded) #formats emotion detection training

emotionModel.load_weights("fer.h5") #emotion detection


def emotionfind():
    inputPictureTaken = "myPicture.jpg"  #picture from webpage
    image = cv2.imread(inputPictureTaken) #uses opencv on photo
    mood = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

    faceCapture = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") #xml that we compare face against
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    findface = faceCapture.detectMultiScale(image, scaleFactor=1.1, minNeighbors=1, minSize=(48, 48),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in findface:
        raw_image = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(raw_image, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        # cv2.rectangle(inputPictureTaken, (x, y), ((x+w), (y+h)), (200, 125, 0), 2)

        this = emotionModel.predict(cropped_img) #makes prediction of users mood
        print(mood[int(np.argmax(this))])
        return (mood[int(np.argmax(this))])


def main():
    mood = emotionfind()
    return mood


if __name__ == "__main__":
    x = main()
