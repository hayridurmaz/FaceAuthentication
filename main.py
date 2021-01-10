import coloredlogs
import cv2

import config
import os
import logging
from Recognizer import Recognizer
from User import getAllUsers, addUser, User
from Utilities import create_file_if_not_exist, create_folder_if_not_exist

radius = config.lbp_params["radius"]
neighbour = config.lbp_params["neighbour"]
grid_x = config.lbp_params["grid_x"]
grid_y = config.lbp_params["grid_y"]

TP = 0
TN = 0
FP = 0
FN = 0


def initilization():
    coloredlogs.install()
    # logging.basicConfig(format='[%(levelname)s]: %(message)s', level=logging.DEBUG)
    create_file_if_not_exist(config.user_file)
    create_folder_if_not_exist(config.recognizer_options['user_dataset'])


def addUsers():
    for f in os.listdir("test_data/train"):
        addUser(User(None, f[:f.index(".")], f[:f.index(".")]))


def train(model_recognizer):
    user_list = getAllUsers()
    for u in user_list:
        model_recognizer.addNewFace("test_data/train/{}.avi".format(u.username), True, u)


def authenticate(model):
    global FN, TP
    user_list = getAllUsers()
    for u in user_list:
        res = model.queryFace("test_data/query/{}.avi".format(u.username), u)
        if res:
            TP = TP + 1
        else:
            FN = FN + 1


def testCase(model):
    # Comment unneeded lines!
    # addUsers()
    train(model)
    authenticate(model)


if __name__ == '__main__':
    initilization()
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius, neighbour, grid_x, grid_y)
    model = Recognizer(recognizer)

    testCase(model)

    # hayri = User(None, "hayri", "hdurmaz")
    # addUser(User(None, "hayri", "hayri"))
    # addUser(User(None, "akadir", "akadir"))
    users = getAllUsers()

    # model.addNewFace(None, False, users[0])
    # model.addNewFace(None, False, user=users[1])

    result = model.queryFace(None, users[0])

    if result:
        logging.info("YES...")
    else:
        logging.error("NO...")
    # image_path = "images/hayri.png"

    # logging.info(image_path)
    # img = cv2.imread(image_path)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # faces = model.Face_Cascade.detectMultiScale(gray, 1.3, 5)
    # for (x, y, w, h) in faces:
    #     img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #     roi_gray = gray[y:y + h, x:x + w]
    #     roi_color = img[y:y + h, x:x + w]
    # # According to Google[21], the face alignment increases the accuracy of its
    # # face recognition model FaceNet from 98.87 to 99.63.
    # camera = cv2.VideoCapture(0)
    # camera.set(3, 640)
    # camera.set(4, 480)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
