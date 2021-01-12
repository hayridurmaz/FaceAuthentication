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


class Recognizer:
    def __init__(self, recognizer):
        self._Face_Cascade = cv2.CascadeClassifier(config.cascade_files['face_cascade_path'])
        self._Right_Eye_Cascade = cv2.CascadeClassifier(config.cascade_files['right_eye_cascade_path'])
        self._Left_Eye_Cascade = cv2.CascadeClassifier(config.cascade_files['left_eye_cascade_path'])
        self._Both_Eye_Cascade = cv2.CascadeClassifier(config.cascade_files['both_eye_cascade_path'])
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
        #         cv2.imshow('Video', image)
        else:
            video = cv2.VideoCapture(config.recognizer_options['camera_id'])
            video.set(3, 640)
            video.set(4, 480)
        # create a dataset for further model training
        create_dataset_for_user(video, user, numberOfSamples, self)
        # Training the model
        self.train()

    def predict(self, img, user_id, detectBlink):
        start = time.time()
        authorized = False
        eyeDetected = False
        if img is None:
            logging.info("Reaching the end of the video, exiting..")
            return None, False, None, time.time() - start, False
        # cv2.imshow('Video', img)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray1 = gray.copy()
        #
        # # Search for faces in the gray-scale image
        # # faces is an array of coordinates of the rectangles where faces exists
        # faces = recognizer.Face_Cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=8, minSize=(10, 10))
        # # check if there are only 1 face in the photo
        # try:
        #     for _, face in enumerate(faces):
        #         # Images with face coordinates
        #         # For gray_chunck, the coordinates are used for further transformation
        #         x, y, w, h = face
        #         gray_chunk = gray[y - 30: y + h + 30, x - 30: x + w + 30]
        #         image_chunk = image[y: y + h, x: x + w]
        #         # Search for the right eye
        #         Right_Eye = recognizer.Right_Eye_Cascade.detectMultiScale(
        #             gray[y: y + int(h / 2), x: x + int(w / 2)],
        #             scaleFactor=1.05, minNeighbors=6,
        #             minSize=(10, 10))
        #         # check if there only one right eye
        #         if len(Right_Eye) > 1:
        #             logging.warning("Right Eye detection is not successful")
        #             raise Exception
        #         elif len(Right_Eye) == 1:
        #             for _, eye1 in enumerate(Right_Eye):
        #                 rx, ry, rw, rh = eye1
        #                 # Search for the left eye
        #                 Left_Eye = recognizer.Left_Eye_Cascade.detectMultiScale(
        #                     gray[y: y + int(h / 2), x + int(w / 2): x + w],
        #                     scaleFactor=1.05, minNeighbors=6, minSize=(10, 10))
        #                 # check if there only one left eye
        #                 if len(Left_Eye) > 1:
        #                     logging.warning("Left Eye detection is not successful")
        #                     raise Exception
        #                 for _, eye2 in enumerate(Left_Eye):
        #                     lx, ly, lw, lh = eye2
        #                     # Calculation of the angle between the eyes
        #                     eyeXdis = (lx + w / 2 + lw / 2) - (rx + rw / 2)
        #                     eyeYdis = (ly + lh / 2) - (ry + rh / 2)
        #                     angle_rad = np.arctan(eyeYdis / eyeXdis)
        #                     # convert degree to rad
        #                     angle_degree = angle_rad * 180 / np.pi
        #                     logging.info("Rotation angle : {:.2f} degree".format(angle_degree))
        #                     # draw rectangles
        #                     Draw_Rect(image, face, [0, 255, 0])
        #
        #                     cv2.imshow('Video', img)
        #                     # Image rotation
        #                     # Find the center of the image
        #                     image_center = tuple(np.array(gray_chunk.shape) / 2)
        #                     rot_mat = cv2.getRotationMatrix2D(image_center, angle_degree, 1.0)
        #                     rotated_image = cv2.warpAffine(gray_chunk, rot_mat, gray_chunk.shape,
        #                                                    flags=cv2.INTER_LINEAR)
        #                     # print("\n[INFO] Adding image number {} to the dataset".format(count))
        #                     # Save the correct inverted image
        #
        #                     Draw_Rect(img, face * 3, [0, 0, 255])
        #                     x, y, w, h = face * 3
        #                     recognized_id, conf = self.recognizer.predict(gray1[y:y + h, x:x + w])
        #
        #
        #                     axs[int(count / 5)][count % 5].imshow(rotated_image, cmap='gray', vmin=0, vmax=255)
        #                     axs[int(count / 5)][count % 5].set_title(
        #                         str(user.id) + '.' + str(count) + ".jpg ",
        #                         fontdict={'fontsize': 15, 'fontweight': 'medium'})
        #                     axs[int(count / 5)][count % 5].axis('off')
        #                     count += 1
        #         else:
        #             # convert degree to rad
        #             # draw rectangles
        #             Draw_Rect(image, face, [0, 255, 0])
        #             cv2.imshow('Video', image)
        #             # Image rotation
        #             # Find the center of the image
        #             # print("\n[INFO] Adding image number {} to the dataset".format(count))
        #             # Save the correct inverted image
        #             if abs(gray_chunk.shape[0] - gray_chunk.shape[1]) > 20:
        #                 continue
        #
        #             axs[int(count / 5)][count % 5].imshow(gray_chunk, cmap='gray', vmin=0, vmax=255)
        #             axs[int(count / 5)][count % 5].set_title(
        #                 str(user.id) + '.' + str(count) + ".jpg ",
        #                 fontdict={'fontsize': 15, 'fontweight': 'medium'})
        #             axs[int(count / 5)][count % 5].axis('off')
        #             count += 1
        # except Exception as e:
        #     logging.error(e)
        #     logging.error("[Warning] Something went wrong!!!")
        #
        # logging.info("Dataset has been successfully created for this person...")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray1 = gray.copy()
        # gray = cv2.equalizeHist(gray)
        # gray = cv2.resize(gray, (0, 0), fx=1 / 3, fy=1 / 3)
        faces = self._Face_Cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=8, minSize=(10, 10))
        f = None
        for _, face in enumerate(faces):
            Draw_Rect(img, face, [0, 0, 255])
            x, y, w, h = face
            recognized_id, conf = self.recognizer.predict(gray1[y:y + h, x:x + w])
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
            if conf > int(config.recognizer_options['confident_threshold']):
                DispID(face, "NOT AUTHENTICATED", img)
                logging.info("Cannot found; conf= {0}".format(conf))
            else:
                if getUserById(recognized_id) is not None and str(recognized_id) == user_id:
                    DispID(face, getUserById(recognized_id).name, img)
                    name = getUserById(recognized_id).name
                    logging.info("{0} found with conf {1}".format(name, conf))
                    f = face
                    authorized = True
                else:
                    DispID(face, getUserById(recognized_id).name, img)
                    name = getUserById(recognized_id).name
                    logging.info("{0} found with conf {1}".format(name, conf))
                    authorized = False
        end = time.time()
        logging.warning("PREDICTION FUNC took {} seconds. ".format(end - start))
        return img, authorized, f, end - start, eyeDetected

    def readInputAndPredict(self, input_, user_id, blink_detection):
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

                for i in range(int(time_taken * 25)):
                    input_.read()
                ret, img = input_.read()
                if authorized:
                    count = count + 1
                else:
                    count = 0

                if count > int(config.recognizer_options['number_of_recognizing_threshold']):
                    if blinkDetected or not blink_detection:
                        DispID(face, 'AUTHENTICATED', img)
                        time.sleep(2)
                        return True
                    else:
                        logging.error("Authentication is successful but no blink detected")
                # result.write(predicted)
                if predicted is not None:
                    cv2.imshow('video', predicted)
            except Exception as e:
                logging.error(e)
                logging.error("[Warning] Something went wrong!!!")
                continue
            k = cv2.waitKey(10) & 0xff  # 'ESC' for Exit

            if k == 27 or predicted is None:
                break

        cv2.destroyAllWindows()
        input_.release()
        return False

    def queryFace(self, input_video, user):
        if not (os.path.isfile(recognizer_file_name)):
            raise RuntimeError("file: %s not found" % recognizer_file_name)
        self.recognizer.read(recognizer_file_name)

        # Disable blink detection when video input
        if input_video is not None:
            video = cv2.VideoCapture(input_video)
            return self.readInputAndPredict(video, user.id, False)
        else:
            camera = cv2.VideoCapture(config.recognizer_options['camera_id'])
            blink_detection = not config.recognizer_options['disable_blink_detection']
            return self.readInputAndPredict(camera, user.id, blink_detection)

    @property
    def Face_Cascade(self):
        return self._Face_Cascade

    @property
    def Right_Eye_Cascade(self):
        return self._Right_Eye_Cascade

    @property
    def Left_Eye_Cascade(self):
        return self._Left_Eye_Cascade
