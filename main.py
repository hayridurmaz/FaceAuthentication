import coloredlogs
import cv2

import config
import os
import logging
from Recognizer import Recognizer
from User import getAllUsers, addUser, User, getUserByUsername
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


def addUsers(model_recognizer):
    for f in os.listdir("test_data/train"):
        if getUserByUsername(f[:f.index(".")]) is None:
            addUser(User(None, f[:f.index(".")], f[:f.index(".")]))
            u = getUserByUsername(f[:f.index(".")])
            model_recognizer.addNewFace("test_data/train/{}.avi".format(u.username), None, u)


# def train(model_recognizer):
#     user_list = getAllUsers()
#     for u in user_list:
#         model_recognizer.addNewFace("test_data/train/{}.avi".format(u.username), None, u)


def authenticate(model):
    global FN, TP
    user_list = getAllUsers()
    for u in user_list:
        res = model.queryFace("test_data/query/{}.avi".format(u.username), u)
        if res:
            TP = TP + 1
            logging.info("Able to authenticate {}".format(u.username))
        else:
            FN = FN + 1
            logging.error("NOT ABLE to authenticate {}".format(u.username))
    print("TP= {0},\nFN={1}".format(TP, FN))


def testCase(model):
    # Comment unneeded lines!
    # for i in range(10):
    # addUsers(model)
    authenticate(model)


def testCase_2(model):
    # addUser(User(None, "hayri", "hayri"))
    # model.addNewFace(None, None, getUserByUsername("hayri"))
    res = model.queryFace(None, getUserByUsername("hayri"))
    print(res)


def testCase_4(model):
    u_name = "bush"
    addUser(User(None, u_name, u_name))
    u = getUserByUsername(u_name)
    model.addNewFace("test_data/train/{}.mp4".format(u.username), None, u)
    res = model.queryFace("test_data/query/{}.mp4".format(u.username), u)
    print(res)


def testCase_3(model):
    while True:
        username = input("please enter username\n")
        if username == "q":
            break
        addUser(User(None, username, username))
        model.addNewFace(None, None, getUserByUsername(username))

    logging.info("querying;")
    while True:
        username = input("please enter username\n")
        if username == "q":
            break
        user = getUserByUsername(username)
        if user is None:
            logging.error("User not found..")
            continue
        res = model.queryFace(None, user)
        print(res)


if __name__ == '__main__':
    initilization()
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius, neighbour, grid_x, grid_y)
    model = Recognizer(recognizer)
    print(model)
    # testCase_2(model)


    testCase(model)


    # u = getUserByUsername("hayri")
    # res = model.queryFace("test_data/query/{}.avi".format(u.username), u)
    # print(res)
