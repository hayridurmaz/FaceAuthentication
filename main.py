import logging

import cv2

from Recognizer import Recognizer

# TODO: can be configured from a file
# variables for LBPH algorithm,
from User import getUsers, User, addUser
from Utilities import create_if_not_exist

radius = 1
neighbour = 8
grid_x = 8
grid_y = 8
face_cascade_path = ''.join([cv2.data.haarcascades, 'haarcascade_frontalface_default.xml'])


def initilization():
    logging.basicConfig(format='[%(levelname)s]: %(message)s', level=logging.DEBUG)
    create_if_not_exist("text_data/", "users.csv")


if __name__ == '__main__':
    initilization()

    recognizer = cv2.face.LBPHFaceRecognizer_create(radius, neighbour, grid_x, grid_y)
    model = Recognizer(face_cascade_path, recognizer)

    # hayri = User(None, "hayri", "hdurmaz")
    addUser(User(None, "akadir", "akadirdurmaz"))
    addUser(User(None, "akadir", "akadirdurmaz"))
    users = getUsers()
    logging.info(users)

    image_path = "images/hayri.png"

    logging.info(image_path)
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = model.Face_Cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
    # According to Google[21], the face alignment increases the accuracy of its
    # face recognition model FaceNet from 98.87 to 99.63.
    camera = cv2.VideoCapture(0)
    camera.set(3, 640)
    camera.set(4, 480)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
