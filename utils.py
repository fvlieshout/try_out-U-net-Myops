# _*_ coding: utf-8 _*_
# Author: Jielong
# @Time: 21/08/2019 22:46
import numpy as np
import os
import random
import torch


def standardize(data):
    avgs = list()
    stds = list()
    for i in range(len(data)):
        avgs.append(np.mean(data[i], axis=(1, 2, 3)))
        stds.append(np.std(data[i], axis=(1, 2, 3)))
    avg = np.mean(np.asarray(avgs), axis=0)
    std = np.mean(np.asarray(stds), axis=0)
    data = (data - avg) / std
    return data
# def normalize(data):
#     min_values = list()
#     max_values = list()
#     for i in range(len(data)):
#         min_values.append(np.min(data[i], axis=(1, 2, 3)))
#         max_values.append(np.max(data[i], axis=(1, 2, 3)))
#     data_min = np.mean(np.asarray(min_values), axis=0)
#     data_max = np.mean(np.asarray(max_values), axis=0)
#     data = (data - data_min) / data_max
#     return data

def label_converter(mask_data):
    new_masks = np.zeros(shape=mask_data.shape, dtype=np.uint8)
    for i in range(len(mask_data)):
        new_masks[i] = np.where(mask_data[i] > 0, 1, 0)

    return new_masks

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def get_model_version_no(log_dir):
    folder_path = os.path.join(log_dir, 'lightning_logs')
    obj_names = os.listdir(folder_path)
    highest_nr = -1
    for fn in obj_names:
        number = fn.split('_')[-1]
        if number.split('.')[-1] == 'txt':
            continue
        number = int(number)
        if number > highest_nr:
            highest_nr = number
    # print(data_paths)
    return highest_nr+1

def normalize(LGE_images):
    norm_images = []
    for image in LGE_images:
        image = image.astype(float)
        image -= np.amin(image)
        image /= np.amax(image)
        norm_images.append(image)
    return norm_images


if __name__ == "__main__":
    import SimpleITK as sitk
    import sys
    np.set_printoptions(threshold=sys.maxsize)
    img = sitk.GetArrayFromImage(sitk.ReadImage("data/STS_001/STS_001_PT_COR_16.tiff"))
    print(img.shape)
    img = img / 255.
    print(img[64])
