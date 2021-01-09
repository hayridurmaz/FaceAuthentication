import sys
import cv2
import os
import time
import numpy as np
import argparse as arg
import matplotlib.pyplot as plt
import logging

import config
from Utilities import create_file_if_not_exist, create_folder_if_not_exist, create_dataset_for_user

numberOfsamples = config.recognizer_options['number_of_samples']
dataset_name = config.recognizer_options['dataset_name']
file_name = config.recognizer_options['file_name']

dataset_path = config.recognizer_options['user_dataset']


class Recognizer:
    def __init__(self, recognizer):
        self._Face_Cascade = cv2.CascadeClassifier(config.cascade_files['face_cascade_path'])
        self._Right_Eye_Cascade = cv2.CascadeClassifier(config.cascade_files['right_eye_cascade_path'])
        self._Left_Eye_Cascade = cv2.CascadeClassifier(config.cascade_files['left_eye_cascade_path'])
        self.recognizer = recognizer
        create_folder_if_not_exist("dataset/")

    # def Add_User(self):
    #     Name = input('\n[INFO] Please Enter a user name and press <return> ==> ')
    #     Info = open("users_name.txt", "a+")
    #     ID = len(open("users_name.txt").readlines()) + 1
    #     Info.write(str(ID) + "," + Name + "\n")
    #     print("\n[INFO] This Person has ID = " + str(ID))
    #     Info.close()
    #     return ID

    def getImagesAndLabels(self):
        imagePaths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
        faceSamples = []
        ids = []
        for imagePath in imagePaths:
            img = cv2.imread(imagePath, 0)
            img_numpy = np.array(img, 'uint8')
            id = int(os.path.split(imagePath)[- 1].split(".")[1])
            faceSamples.append(img_numpy)
            ids.append(id)
        return faceSamples, ids

    def train(self):
        logging.info("Training...")
        # slight delay
        time.sleep(1)
        faces, ids = self.getImagesAndLabels()
        self.recognizer.update(faces, np.array(ids))
        # Saving the model
        self.recognizer.write(file_name)
        logging.info("trained with {0} images successfully.".format(len(np.unique(ids))))

    def addNewFace(self, input, isVideo, user):
        if isVideo:
            video = cv2.VideoCapture(input)
        else:
            video = cv2.VideoCapture(config.recognizer_options['camera_id'])
            video.set(3, 640)
            video.set(4, 480)
        # create a dataset for further model training
        create_dataset_for_user(video, user, numberOfsamples, self)
        # Training the model
        self.train()

    @property
    def Face_Cascade(self):
        return self._Face_Cascade
