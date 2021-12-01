import numpy as np
import torch
import os
import requests
import tarfile
import nrrd
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from skimage.transform import resize
from matplotlib import cm
import cv2

DATA_DIR_NAME = 'MICCAI-STACOM2012'

def import_data():
    if not os.path.isdir(DATA_DIR_NAME):
        url = "http://www.cardiacatlas.org/files/MICCAI-STACOM2012/MICCAI-STACOM2012_human.tar.gz"
        response = requests.get(url, stream=True)
        file = tarfile.open(fileobj=response.raw, mode="r|gz")
        file.extractall(path=".")

img_data, img_head = nrrd.read(os.path.join(DATA_DIR_NAME, 'human', 'training', 'p4_de.nrrd'))
print(img_data.shape)
# print(img_data)
print(img_head)
show_data = img_data[:,:,10]
show_data = show_data / 1000
# other_data = np.random.rand(560,560)
# norm_image = cv2.normalize(show_data, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
plt.imshow(show_data, cmap='Greys')
plt.show()