import cv2

if __name__ == '__main__':
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    image_path = "images/hayri.png"
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
    # According to Google[21], the face alignment increases the accuracy of its
    # face recognition model FaceNet from 98.87 to 99.63.
    camera = cv2.VideoCapture(0)
    camera.set(3, 640)
    camera.set(4, 480)
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
