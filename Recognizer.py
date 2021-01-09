import cv2
import logging
import os
import time

import cv2
import numpy as np

import config
from User import getUserById
from Utilities import create_folder_if_not_exist, create_dataset_for_user, Draw_Rect

numberOfsamples = config.recognizer_options['number_of_samples']
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

    def getImagesAndLabels(self):
        imagePaths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
        faceSamples = []
        ids = []
        for imagePath in imagePaths:
            img = cv2.imread(imagePath, 0)
            img_numpy = np.array(img, 'uint8')
            id = int(os.path.split(imagePath)[- 1].split(".")[0])
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
        self.recognizer.write(recognizer_file_name)
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

    def DispID(self, face, NAME, Image):
        x, y, w, h = face
        pt1 = (int(x + w / 2.0 - 50), int(y + h + 40))
        pt2 = (int(x + w / 2.0 + 50), int(y + h + 65))
        pt3 = (int(x + w / 2.0 - 46), int(y + h + (-int(y + h) + int(y + h + 25)) / 2 + 48))
        triangle_cnt = np.array([(int(x + w / 2.0), int(y + h + 10)),
                                 (int(x + w / 2.0 - 20), int(y + h + 35)),
                                 (int(x + w / 2.0 + 20), int(y + h + 35))])
        cv2.drawContours(Image, [triangle_cnt], 0, (255, 255, 255), -1)
        cv2.rectangle(Image, pt1, pt2, (255, 255, 255), -1)
        cv2.rectangle(Image, pt1, pt2, (0, 0, 255), 1)
        cv2.putText(Image, NAME, pt3, cv2.FONT_HERSHEY_PLAIN, 1.1, (0, 0, 255))

    def Get_UserName(self, ID, conf):
        print("[INFO] Confidence: " + "{:.2f} ".format(conf))
        if not ID > 0:
            return " Unknown "
        return "sa"

    def predict(self, img, size1, size2):
        if img is None:
            logging.info("Reaching the end of the video, exiting..")
            return
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray1 = gray.copy()
        # gray = cv2.equalizeHist(gray)
        gray = cv2.resize(gray, (0, 0), fx=1 / 3, fy=1 / 3)
        faces = self._Face_Cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30))
        if len(faces) == 0:
            img1 = cv2.resize(img, (0, 0), fx=1 / 3, fy=1 / 3)
            # faces = skin_face_detector.Detect_Face_Img(img1, size1, size2)
        for _, face in enumerate(faces):
            Draw_Rect(img, face * 3, [0, 255, 0])
            x, y, w, h = face * 3
            recognized_id, conf = self.recognizer.predict(gray1[y:y + h, x:x + w])
            # Check that the face is recognized
            if (conf > 75):
                self.DispID(face * 3, "CANNOT RECOGNIZE", img)
            else:
                if getUserById(recognized_id) is not None:
                    self.DispID(face * 3, getUserById(recognized_id).name, img)
                    name = getUserById(recognized_id).name
                    logging.info("{0} found with conf {1}".format(name, conf))

        return img

    def readInputAndPredict(self, input_):
        # What are they?
        size1 = (30, 30)
        size2 = (80, 110)

        frame_width = int(input_.get(3))
        frame_height = int(input_.get(4))
        size = (frame_width, frame_height)
        # result = cv2.VideoWriter('filename.avi',
        #                          cv2.VideoWriter_fourcc(*'MJPG'),
        #                          10, size)
        while True:
            ret, img = input_.read()
            predicted = self.predict(img, size1, size2)
            # result.write(predicted)
            cv2.imshow('video', predicted)
            k = cv2.waitKey(10) & 0xff  # 'ESC' for Exit
            if k == 27 or predicted is None:
                break
        cv2.destroyAllWindows()
        input_.release()

    def queryFace(self, input, user):
        if not (os.path.isfile(recognizer_file_name)):
            raise RuntimeError("file: %s not found" % recognizer_file_name)
        self.recognizer.read(recognizer_file_name)

        if input is not None:
            video = cv2.VideoCapture(input)
            self.readInputAndPredict(video)
        else:
            camera = cv2.VideoCapture(config.recognizer_options['camera_id'])
            self.readInputAndPredict(camera)

    @property
    def Face_Cascade(self):
        return self._Face_Cascade
