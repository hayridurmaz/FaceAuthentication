import cv2
import logging
import os
import time
import pickle
import cv2
import numpy as np
from deepface import DeepFace

import config
from User import getUserById
from Utilities import create_folder_if_not_exist, create_dataset_for_user, Draw_Rect, DispID, getImagesAndLabels, \
    img_to_encoding, getImagesAndLabelsForUser, showImage, get_embedding, detect_face
from mtcnn.mtcnn import MTCNN

numberOfSamples = config.recognizer_options['number_of_samples']
dataset_name = config.recognizer_options['dataset_name']
recognizer_file_name = config.recognizer_options['file_name']
model_file_name = config.recognizer_options['model_file']


# detector = MTCNN()

# models = MTCNN(weights_file='weights/mtcnn_weights.npy')


class Recognizer:
    def __init__(self, recognizer):
        self._Face_Cascade = cv2.CascadeClassifier(config.cascade_files['face_cascade_path'])
        self._Right_Eye_Cascade = cv2.CascadeClassifier(config.cascade_files['right_eye_cascade_path'])
        self._Left_Eye_Cascade = cv2.CascadeClassifier(config.cascade_files['left_eye_cascade_path'])
        self._Both_Eye_Cascade = cv2.CascadeClassifier(config.cascade_files['both_eye_cascade_path'])
        create_folder_if_not_exist("dataset/")
        self.ids_and_representations = []
        self.readModel()

    def saveModel(self):
        with open(model_file_name, 'wb') as f:
            pickle.dump(self.ids_and_representations, f)

    def readModel(self):
        try:
            with open(model_file_name, 'rb') as fp:
                self.ids_and_representations = pickle.load(fp)

        except Exception as e:
            logging.error(e)
            self.ids_and_representations = []

    def add_faces_of_one_user(self, user):
        images = getImagesAndLabelsForUser(user)
        for img in images:
            try:
                representation = get_embedding(img)
                self.ids_and_representations.append((user.id, representation))
            except Exception as e:
                logging.error(e)
                continue
        self.saveModel()

    def train(self):
        logging.info("Training...")
        # slight delay
        time.sleep(1)
        faces, ids = getImagesAndLabels()
        logging.info("trained with {0} images successfully.".format(len(np.unique(ids))))

    def addNewFace(self, input_video_path, input_video_folder_path, user):
        if input_video_path is not None:
            video = cv2.VideoCapture(input_video_path)
        # elif input_video_folder_path is not None:
        # TODO:
        #     # input_video = cv2.VideoWriter('{}video.avi'.format(input_video_folder_path), -1, 1, (600, 480))
        #     # print(os.listdir(input_video_folder_path))
        #     # imagePaths = [os.path.join(input_video_folder_path, f) for f in os.listdir(input_video_folder_path)]
        #     # print(imagePaths)
        #     # for image in imagePaths:
        #     #     input_video.write(cv2.imread(image))
        #     # input_video.release()
        #     video = cv2.VideoCapture('{}%04d.jpg'.format(input_video_folder_path), cv2.CAP_IMAGES)
        #     video.release()
        #     while True:
        #         ret, image = video.read()
        #         # Convert to gray-scale image
        #         if image is None:
        #             logging.error("NONE")
        #             continue
        #         showImage(image)
        else:
            video = cv2.VideoCapture(config.recognizer_options['camera_id'])
            video.set(3, 640)
            video.set(4, 480)
        # create a dataset for further models training
        create_dataset_for_user(video, user, numberOfSamples, self)
        self.add_faces_of_one_user(user)
        # Training the models
        # self.train()

    def predict(self, img, user_id, detectBlink):
        # df = DeepFace.find(img_path=img, db_path=config.recognizer_options['user_database'])
        # return img, df, None, 0, True

        start_func = time.time()
        authorized = False
        eyeDetected = False
        if img is None:
            logging.info("Reaching the end of the video, exiting..")
            return None, False, None, time.time() - start_func, False

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray1 = gray.copy()
        org_image = img.copy()
        # gray = cv2.equalizeHist(gray)
        # gray = cv2.resize(gray, (0, 0), fx=1 / 3, fy=1 / 3)
        # faces = self._Face_Cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=8, minSize=(30, 30))
        recognized_id = None
        faces = detect_face(img, showImage=False)
        f = None
        embedding = get_embedding(faces)
        min_distance = 10000
        min_id = -1
        start_for_loop = time.time()
        for tuple in self.ids_and_representations:
            start = time.time()
            end = time.time()
            logging.debug("embedding FUNC took {} seconds. ".format(end - start))
            start = time.time()
            dist = np.linalg.norm(embedding - tuple[1])
            end = time.time()
            logging.debug("norm FUNC took {} seconds. ".format(end - start))
            start = time.time()
            if dist < min_distance:
                min_distance = dist
                recognized_id = tuple[0]
        end_for_loop = time.time()
        logging.info("For loop took {}".format((end_for_loop - start_for_loop)))
        if detectBlink:
            # eye detection for blink detection
            eyes = self._Both_Eye_Cascade.detectMultiScale(gray1, 1.3, 5, minSize=(10, 10))
            # Examining the length of eyes object for eyes
            if len(eyes) >= 2:
                # Check if program is running for detection
                eyeDetected = True
            else:
                eyeDetected = False
        # Check that the face is recognized
        if min_distance > int(config.recognizer_options['confident_threshold']):
            # DispID(face['box'], "NOT AUTHENTICATED", img)
            logging.info("Cannot found; conf= {0}".format(min_distance))
        else:
            if getUserById(recognized_id) is not None and str(recognized_id) == user_id:
                # DispID(face['box'], getUserById(recognized_id).name, img)
                name = getUserById(recognized_id).name
                logging.info("{0} found with conf {1}".format(name, min_distance))
                # f = face
                authorized = True
            else:
                # DispID(face['box'], getUserById(recognized_id).name, img)
                name = getUserById(recognized_id).name
                logging.info("{0} found with conf {1}".format(name, min_distance))
                authorized = False
        end = time.time()
        logging.warning("PREDICTION FUNC took {} seconds. ".format(end - start_func))
        return img, authorized, f, end - start_func, eyeDetected

    def readInputAndPredict(self, input_, user_id, blink_detection, is_camera):
        # frame_width = int(input_.get(3))
        # frame_height = int(input_.get(4))
        # size = (frame_width, frame_height)
        # result = cv2.VideoWriter('filename.avi',
        #                          cv2.VideoWriter_fourcc(*'MJPG'),
        #                          10, size)
        count = 0
        start = time.time()
        oldEye = False
        blinkDetected = False
        eyeReturnValue = False
        isStart = True
        while True:
            try:
                curr = time.time()
                if curr - start > int(config.recognizer_options['timeout']):
                    logging.error("Couldnt recognize")
                    return False

                ret, img = input_.read()
                showImage(img)
                predicted, authorized, face, time_taken, eyeReturnValue = self.predict(img, user_id, blink_detection)
                # Blink detection
                if isStart:
                    oldEye = eyeReturnValue
                    isStart = False
                else:
                    if eyeReturnValue != oldEye:
                        blinkDetected = True
                        logging.info("Blink detected")
                    else:
                        oldEye = eyeReturnValue
                if is_camera:
                    for i in range(int(time_taken * 25 / 2)):
                        input_.read()
                ret, img = input_.read()
                if authorized:
                    count = count + 1

                if count > int(config.recognizer_options['number_of_recognizing_threshold']):
                    if blinkDetected or not blink_detection:
                        # DispID(face, 'AUTHENTICATED', img)
                        time.sleep(2)
                        return True
                    else:
                        logging.error("Authentication is successful but no blink detected")
                # result.write(predicted)
                if predicted is not None:
                    showImage(predicted)
            except Exception as e:
                logging.error(e)
                logging.error("Something went wrong!!!")
                continue
            k = cv2.waitKey(10) & 0xff  # 'ESC' for Exit

            if k == 27 or predicted is None:
                break

        cv2.destroyAllWindows()
        input_.release()
        return False

    def queryFace(self, input_video, user):
        self.readModel()

        # Disable blink detection when video input
        if input_video is not None:
            video = cv2.VideoCapture(input_video)
            return self.readInputAndPredict(video, user.id, False, False)
        else:
            camera = cv2.VideoCapture(config.recognizer_options['camera_id'])
            blink_detection = not config.recognizer_options['disable_blink_detection']
            return self.readInputAndPredict(camera, user.id, blink_detection, True)

    @property
    def Face_Cascade(self):
        return self._Face_Cascade

    @property
    def Right_Eye_Cascade(self):
        return self._Right_Eye_Cascade

    @property
    def Left_Eye_Cascade(self):
        return self._Left_Eye_Cascade
