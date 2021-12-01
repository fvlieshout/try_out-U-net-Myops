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
import SimpleITK as sitk
import cv2

DATA_DIR_NAME = 'Datasets\\MyoPS 2020 Dataset'

# A path to a T1-weighted brain .nii image:
image_path = os.path.join(DATA_DIR_NAME, 'train25', 'train25', 'myops_training_101_DE.nii.gz')
# image_path = "\\amc.intra\\users\\F\\fevanlieshout\\home\\Datasets\\Myops dataset\\MyoPS 2020 Dataset\\train25\\train25\\myops_training_101_C0.nii.gz"

# image_path = "Datasets\\MyoPS 2020 Dataset\\train25\\train25\\myops_training_101_C0.nii.gz"
check_bool = os.path.isfile(image_path)

# Read the .nii image containing the volume with SimpleITK:
sitk_image = sitk.ReadImage(image_path)

# and access the numpy array:
image = sitk.GetArrayFromImage(sitk_image)
for i in range(image.shape[0]):
    plt.imshow(image[i], cmap='Greys')
    plt.show()
print()

