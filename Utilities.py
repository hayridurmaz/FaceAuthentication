import logging
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from deepface import DeepFace
from keras_vggface.vggface import VGGFace

import config

dataset_path = config.recognizer_options['user_dataset']
database_path = config.recognizer_options['user_database']

resnet50_features = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3),
                            pooling='avg')  # pooling: None, avg or max


def resize(img):
    img = cv2.resize(img, (224, 224))  # resize image to match model's expected sizing
    img = img.reshape(1, 224, 224, 3)  # return the image with shaping that TF wants.
    return img


# get the face embedding for one face
def get_embedding(face_pixels):
    return resnet50_features.predict(resize(face_pixels))


def create_file_if_not_exist(file_name):
    if not os.path.exists(file_name):
        os.mknod(file_name)


def create_folder_if_not_exist(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)


def FileRead(file_path="users_name.txt"):
    NAME = []
    with open(file_path, "r") as f:
        for line in f:
            NAME.append(line.split(",")[1].rstrip())
    return NAME


def Draw_Rect(Image, face, color):
    x, y, w, h = face
    cv2.line(Image, (x, y), (int(x + (w / 5)), y), color, 2)
    cv2.line(Image, (int(x + ((w / 5) * 4)), y), (x + w, y), color, 2)
    cv2.line(Image, (x, y), (x, int(y + (h / 5))), color, 2)
    cv2.line(Image, (x + w, y), (x + w, int(y + (h / 5))), color, 2)
    cv2.line(Image, (x, int(y + (h / 5 * 4))), (x, y + h), color, 2)
    cv2.line(Image, (x, int(y + h)), (x + int(w / 5), y + h), color, 2)
    cv2.line(Image, (x + int((w / 5) * 4), y + h), (x + w, y + h), color, 2)
    cv2.line(Image, (x + w, int(y + (h / 5 * 4))), (x + w, y + h), color, 2)


def getImagesAndLabels():
    imagePaths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        img = cv2.imread(imagePath, 0)
        img_numpy = np.array(img, 'uint8')
        user_id = int(os.path.split(imagePath)[- 1].split(".")[0])  # Burada sorun olabilir.
        faceSamples.append(img_numpy)
        ids.append(user_id)
    return faceSamples, ids


def getImagesAndLabelsForUser(user):
    imagePaths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        img = cv2.imread(imagePath)
        img_numpy = np.array(img, 'uint8')
        user_id = int(os.path.split(imagePath)[- 1].split(".")[0])
        if str(user_id) == user.id:
            faceSamples.append(img_numpy)
            ids.append(user_id)
    return faceSamples


def showImage(image, title='Video'):
    cv2.imshow(title, image)
    cv2.waitKey(delay=1)


def detect_face(image, isShowImage=False):
    faces = DeepFace.detectFace(image, detector_backend="ssd")
    faces = cv2.cvtColor(faces, cv2.COLOR_RGB2BGR)
    if isShowImage:
        showImage(faces)
    faces = 255 * faces
    faces = np.asarray(faces, dtype=int)
    return faces


def create_dataset_for_user(cam, user, numberOfsamples, recognizer):
    fig, axs = plt.subplots(10, 5, figsize=(20, 20), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5, wspace=.001)
    count = 0  # Variable for counting the number of captured face photos
    logging.info("Please look into the camera and wait ...")
    start_processing_video = time.time()
    while True:
        # Capture, decode and return the next frame of the video
        ret, image = cam.read()
        if image is None:
            break
        start_reading_image = time.time()
        try:
            faces = detect_face(image, isShowImage=True)
            logging.info("Face detector took {} sec".format(time.time() - start_reading_image))
            cv2.imwrite(
                dataset_path + str(user.id) + '.' + str(
                    count) + ".jpg ",
                faces)
            logging.info("Saved one photo in {} sec".format(time.time() - start_reading_image))
            axs[int(count / 5)][count % 5].imshow(faces, vmin=0, vmax=255)
            axs[int(count / 5)][count % 5].set_title(
                str(user.id) + '.' + str(count) + ".jpg ",
                fontdict={'fontsize': 15, 'fontweight': 'medium'})
            axs[int(count / 5)][count % 5].axis('off')
            count += 1
        except Exception as e:
            logging.error("There are either no face or more than one face found")
            continue
        if cv2.waitKey(1) & 0xff == 27:  # To exit the program, press "Esc", wait 100 ms,
            break
        elif count >= numberOfsamples:  # taking pic_num photos
            break
    logging.info("Dataset has been successfully created for this person... in {} secs".format(
        time.time() - start_processing_video))
    cam.release()
    cv2.destroyAllWindows()
    plt.show()
