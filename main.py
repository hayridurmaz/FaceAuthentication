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
    # train(model)
    authenticate(model)


if __name__ == '__main__':
    initilization()
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius, neighbour, grid_x, grid_y)
    model = Recognizer(recognizer)

    testCase(model)
