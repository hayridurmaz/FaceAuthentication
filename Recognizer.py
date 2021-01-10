import cv2
import logging
import os
import time

import cv2
import numpy as np

import config
from User import getUserById
from Utilities import create_folder_if_not_exist, create_dataset_for_user, Draw_Rect, DispID, getImagesAndLabels

numberOfSamples = config.recognizer_options['number_of_samples']
dataset_name = config.recognizer_options['dataset_name']
recognizer_file_name = config.recognizer_options['file_name']
dataset_path = config.recognizer_options['user_dataset']


class Recognizer:
    def __init__(self, recognizer):
        self._Face_Cascade = cv2.CascadeClassifier(config.cascade_files['face_cascade_path'])
        self._Right_Eye_Cascade = cv2.CascadeClassifier(config.cascade_files['right_eye_cascade_path'])
        self._Left_Eye_Cascade = cv2.CascadeClassifier(config.cascade_files['left_eye_cascade_path'])
        self.recognizer = recognizer
        create_folder_if_not_exist("dataset/")

    def train(self):
        logging.info("Training...")
        # slight delay
        time.sleep(1)
        faces, ids = getImagesAndLabels()
        self.recognizer.update(faces, np.array(ids))
        # Saving the model
        self.recognizer.write(recognizer_file_name)
        logging.info("trained with {0} images successfully.".format(len(np.unique(ids))))

    def addNewFace(self, input_video, isVideo, user):
        if isVideo:
            video = cv2.VideoCapture(input_video)
        else:
            video = cv2.VideoCapture(config.recognizer_options['camera_id'])
            video.set(3, 640)
            video.set(4, 480)
        # create a dataset for further model training
        create_dataset_for_user(video, user, numberOfSamples, self)
        # Training the model
        self.train()

    def predict(self, img, user_id):
        authorized = False
        if img is None:
            logging.info("Reaching the end of the video, exiting..")
            return
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray1 = gray.copy()
        # gray = cv2.equalizeHist(gray)
        gray = cv2.resize(gray, (0, 0), fx=1 / 3, fy=1 / 3)
        faces = self._Face_Cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30))
        for _, face in enumerate(faces):
            Draw_Rect(img, face * 3, [0, 255, 0])
            x, y, w, h = face * 3
            recognized_id, conf = self.recognizer.predict(gray1[y:y + h, x:x + w])
            # Check that the face is recognized
            if conf > int(config.recognizer_options['confident_threshold']):
                DispID(face * 3, "CANNOT RECOGNIZE ({}) ({})".format(conf, recognized_id), img)
            else:
                if getUserById(recognized_id) is not None and recognized_id != user_id:
                    DispID(face * 3, getUserById(recognized_id).name, img)
                    name = getUserById(recognized_id).name
                    logging.info("{0} found with conf {1}".format(name, conf))
                    authorized = True
                else:
                    DispID(face * 3, getUserById(recognized_id).name, img)
                    name = getUserById(recognized_id).name
                    logging.info("{0} found with conf {1}".format(name, conf))
                    authorized = False
        return img, authorized

    def readInputAndPredict(self, input_, user_id):
        # frame_width = int(input_.get(3))
        # frame_height = int(input_.get(4))
        # size = (frame_width, frame_height)
        # result = cv2.VideoWriter('filename.avi',
        #                          cv2.VideoWriter_fourcc(*'MJPG'),
        #                          10, size)
        count = 0
        start = time.time()
        while True:
            curr = time.time()
            if curr - start > int(config.recognizer_options['timeout']):
                logging.error("Couldnt recognize")
                return False
            ret, img = input_.read()
            predicted, authorized = self.predict(img, user_id)
            if authorized:
                count = count + 1
            else:
                count = 0

            if count > int(config.recognizer_options['number_of_recognizing_threshold']):
                return True
            # result.write(predicted)
            cv2.imshow('video', predicted)
            k = cv2.waitKey(10) & 0xff  # 'ESC' for Exit
            if k == 27 or predicted is None:
                break
        cv2.destroyAllWindows()
        input_.release()

    def queryFace(self, input_video, user):
        if not (os.path.isfile(recognizer_file_name)):
            raise RuntimeError("file: %s not found" % recognizer_file_name)
        self.recognizer.read(recognizer_file_name)

        if input_video is not None:
            video = cv2.VideoCapture(input_video)
            return self.readInputAndPredict(video, user.id)
        else:
            camera = cv2.VideoCapture(config.recognizer_options['camera_id'])
            return self.readInputAndPredict(camera, user.id)

    @property
    def Face_Cascade(self):
        return self._Face_Cascade

    @property
    def Right_Eye_Cascade(self):
        return self._Right_Eye_Cascade

    @property
    def Left_Eye_Cascade(self):
        return self._Left_Eye_Cascade
