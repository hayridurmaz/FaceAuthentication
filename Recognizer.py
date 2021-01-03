import sys
import cv2
import os
import time
import numpy as np
import argparse as arg
import matplotlib.pyplot as plt

from Utilities import create_if_not_exist


class Recognizer:
    def __init__(self, face_cascade, recognizer):
        self._Face_Cascade = cv2.CascadeClassifier(face_cascade)
        self.recognizer = recognizer
        create_if_not_exist("dataset/")

    # def Add_User(self):
    #     Name = input('\n[INFO] Please Enter a user name and press <return> ==> ')
    #     Info = open("users_name.txt", "a+")
    #     ID = len(open("users_name.txt").readlines()) + 1
    #     Info.write(str(ID) + "," + Name + "\n")
    #     print("\n[INFO] This Person has ID = " + str(ID))
    #     Info.close()
    #     return ID

    def getImagesAndLabels(self, path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []
        for imagePath in imagePaths:
            img = cv2.imread(imagePath, 0)
            img_numpy = np.array(img, 'uint8')
            id = int(os.path.split(imagePath)[- 1].split(".")[1])
            faceSamples.append(img_numpy)
            ids.append(id)
        return faceSamples, ids

    def train(self, path, file_name):
        print("\n[INFO] Face training has been started, please wait a moment...")
        # slight delay
        time.sleep(1)
        faces, ids = self.getImagesAndLabels(path)
        self.recognizer.update(faces, np.array(ids))
        # Saving the model
        self.recognizer.write(file_name)
        print("\n[INFO] {0} persons trained successfully.".format(len(np.unique(ids))))
        print("\n[INFO] Quitting the program")

    @property
    def Face_Cascade(self):
        return self._Face_Cascade
