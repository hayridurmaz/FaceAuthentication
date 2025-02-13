import cv2

lbp_params = dict(
    radius=1,
    neighbour=8,
    grid_x=8,
    grid_y=8
)
recognizer_options = dict(
    number_of_samples=50,
    dataset_name='dataset/',
    user_dataset='dataset/users/',
    user_database='database/',
    model_file='dataset/model_file.pickle',
    camera_id=0,
    timeout=100,
    number_of_recognizing_threshold=15,
    confident_threshold=200,
    disable_blink_detection=True
)
user_file = 'dataset/users.csv'
cascade_files = dict(
    face_cascade_path=''.join([cv2.data.haarcascades, 'haarcascade_frontalface_default.xml']),
    right_eye_cascade_path=''.join([cv2.data.haarcascades, 'haarcascade_righteye_2splits.xml']),
    left_eye_cascade_path=''.join([cv2.data.haarcascades, 'haarcascade_lefteye_2splits.xml']),
    both_eye_cascade_path=''.join([cv2.data.haarcascades, 'haarcascade_eye_tree_eyeglasses.xml'])
)
