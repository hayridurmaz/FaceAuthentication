import time

from deepface import DeepFace
from matplotlib import pyplot as plt

backends = ['opencv', 'ssd', 'mtcnn']

fig, ax = plt.subplots(nrows=3, ncols=1)

count = 0
for backend in backends:
    start = time.time()
    detected_aligned_face = DeepFace.detectFace(img_path="hayri.jpg", detector_backend=backend)
    end = time.time()
    print("{} took {} secs".format(backend, (end - start)))
    ax[count].imshow(detected_aligned_face, vmin=0, vmax=255)
    ax[int(count)].set_title(
        backend)
    count = count + 1

plt.show()
