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
    file_name='train.yaml',
    videoId=0
)
user_file = 'dataset/users.csv'
cascade_files = dict(
    face_cascade_path=''.join([cv2.data.haarcascades, 'haarcascade_frontalface_default.xml']),
    right_eye_cascade_path=''.join([cv2.data.haarcascades, 'haarcascade_righteye_2splits.xml']),
    left_eye_cascade_path=''.join([cv2.data.haarcascades, 'haarcascade_lefteye_2splits.xml'])
)
