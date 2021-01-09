import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

import config


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


# def log(log_level, log_str):
#     print("[{}] : {} ".format(log_level, log_str))


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
        user_id = int(os.path.split(imagePath)[- 1].split(".")[0])
        faceSamples.append(img_numpy)
        ids.append(user_id)
    return faceSamples, ids


def DispID(face, NAME, Image):
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


# def Get_UserName(ID, conf):
#     print("[INFO] Confidence: " + "{:.2f} ".format(conf))
#     if not ID > 0:
#         return " Unknown "
#     return "sa"


def create_dataset_for_user(cam, user, numberOfsamples, recognizer):
    fig, axs = plt.subplots(10, 5, figsize=(20, 20), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace=.5, wspace=.001)
    count = 0  # Variable for counting the number of captured face photos
    logging.info("Please look into the camera and wait ...")
    while True:
        # Capture, decode and return the next frame of the video
        ret, image = cam.read()
        cv2.imshow('Video', image)
        # Convert to gray-scale image
        if image is None:
            break
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Search for faces in the gray-scale image
        # faces is an array of coordinates of the rectangles where faces exists
        faces = recognizer.Face_Cascade.detectMultiScale(gray, scaleFactor=1.098, minNeighbors=6, minSize=(50, 50))
        # check if there are only 1 face in the photo
        if len(faces) > 1:
            logging.error("There are more than one face")
            continue
        try:
            for _, face in enumerate(faces):
                # Images with face coordinates
                # For gray_chunck, the coordinates are used for further transformation
                x, y, w, h = face
                gray_chunk = gray[y - 30: y + h + 30, x - 30: x + w + 30]
                image_chunk = image[y: y + h, x: x + w]
                # Search for the right eye
                Right_Eye = recognizer.Right_Eye_Cascade.detectMultiScale(gray[y: y + int(h / 2), x: x + int(w / 2)],
                                                                          scaleFactor=1.05, minNeighbors=6,
                                                                          minSize=(10, 10))
                # check if there only one right eye
                if len(Right_Eye) > 1:
                    logging.warning("Right Eye detection is not successful")
                    raise Exception
                for _, eye1 in enumerate(Right_Eye):
                    rx, ry, rw, rh = eye1
                    # Search for the left eye
                    Left_Eye = recognizer.Left_Eye_Cascade.detectMultiScale(
                        gray[y: y + int(h / 2), x + int(w / 2): x + w],
                        scaleFactor=1.05, minNeighbors=6, minSize=(10, 10))
                    # check if there only one left eye
                    if len(Left_Eye) > 1:
                        logging.warning("Left Eye detection is not successful")
                        raise Exception
                    for _, eye2 in enumerate(Left_Eye):
                        lx, ly, lw, lh = eye2
                        # Calculation of the angle between the eyes
                        eyeXdis = (lx + w / 2 + lw / 2) - (rx + rw / 2)
                        eyeYdis = (ly + lh / 2) - (ry + rh / 2)
                        angle_rad = np.arctan(eyeYdis / eyeXdis)
                        # convert degree to rad
                        angle_degree = angle_rad * 180 / np.pi
                        logging.info("Rotation angle : {:.2f} degree".format(angle_degree))
                        # draw rectangles
                        Draw_Rect(image, face, [0, 255, 0])
                        cv2.rectangle(image_chunk, (rx, ry), (rx + rw, ry + rh), (255, 255, 255), 2)
                        cv2.rectangle(image_chunk, (lx + int(w / 2), ly), (lx + int(w / 2) + lw, ly + lh),
                                      (0, 255, 255), 2)
                        cv2.imshow('Video', image)
                        # Image rotation
                        # Find the center of the image
                        image_center = tuple(np.array(gray_chunk.shape) / 2)
                        rot_mat = cv2.getRotationMatrix2D(image_center, angle_degree, 1.0)
                        rotated_image = cv2.warpAffine(gray_chunk, rot_mat, gray_chunk.shape,
                                                       flags=cv2.INTER_LINEAR)
                        # print("\n[INFO] Adding image number {} to the dataset".format(count))
                        # Save the correct inverted image
                        cv2.imwrite(
                            config.recognizer_options['user_dataset'] + str(user.id) + '.' + str(count) + ".jpg ",
                            rotated_image)
                        axs[int(count / 5)][count % 5].imshow(rotated_image, cmap='gray', vmin=0, vmax=255)
                        axs[int(count / 5)][count % 5].set_title(
                            str(user.id) + '.' + str(count) + ".jpg ",
                            fontdict={'fontsize': 15, 'fontweight': 'medium'})
                        axs[int(count / 5)][count % 5].axis('off')
                        count += 1
        except Exception as e:
            logging.error(e)
            logging.error("[Warning] Something went wrong!!!")
            continue
        if cv2.waitKey(1) & 0xff == 27:  # To exit the program, press "Esc", wait 100 ms,
            break
        elif count >= numberOfsamples:  # taking pic_num photos
            break
    logging.info("Dataset has been successfully created for this person...")
    cam.release()
    cv2.destroyAllWindows()
    plt.show()
