import numpy as np
from torch._C import Value
import torch
from tqdm import tqdm
import csv
import os
from matplotlib import pyplot as plt
from matplotlib import patches
from mpl_toolkits.mplot3d import axes3d
from skimage.transform import resize
from matplotlib import cm
import SimpleITK as sitk
import torchvision.transforms as transforms
import cv2
from utils import normalize

if torch.cuda.is_available():
    ORIGINAL_DIR_NAME = 'AUMC_data'
else:
    ORIGINAL_DIR_NAME = 'L:\\basic\\diva1\\Onderzoekers\\DEEP-RISK\\DEEP-RISK\\CMR DICOMS\\Roel&Floor\\sample_niftis\\labels\\labels_model_testing'
NIFTI_SUFFIX = 'LGE_niftis'
MYO_MASK_SUFFIX = 'myo'
AANKLEURING_MASK_SUFFIX = 'aankleuring'
BOUNDING_BOX_FILE = 'bounding_boxes.csv'

def read_in_AUMC_data(mode, resize='resize', size=(256, 256)):
    if mode not in ['train', 'test']:
        raise ValueError("'mode' argument should be either 'train' or 'test'")
    data_folder_path = os.path.join(ORIGINAL_DIR_NAME, mode, NIFTI_SUFFIX)
    myo_mask_folder_path = os.path.join(ORIGINAL_DIR_NAME, mode, MYO_MASK_SUFFIX)
    aankleuring_mask_folder_path = os.path.join(ORIGINAL_DIR_NAME, mode, AANKLEURING_MASK_SUFFIX)

    data_paths = get_data_paths(data_folder_path)
    myo_mask_paths = get_data_paths(myo_mask_folder_path)
    aankleuring_mask_paths = get_data_paths(aankleuring_mask_folder_path)
    no_samples = len(data_paths)

    pat_ids = []
    LGE_imgs = []
    myo_masks = []
    aankleuring_masks = []

    # if resize == 'resize':
    #     resizing = transforms.Resize(size)

    for i, (nifti_path, myo_mask_path, aankleuring_mask_path) in tqdm(enumerate(zip(data_paths, myo_mask_paths, aankleuring_mask_paths)), total=no_samples):
        # print(nifti_path)
        if '\\' in nifti_path:
            pat_id = nifti_path.split('\\')[-1].split('_')[0]
            pat_ids.append(pat_id)
        elif '/' in nifti_path:
            pat_id = nifti_path.split('/')[-1].split('_')[0]
            pat_ids.append(pat_id)
        LGE_img = sitk.GetArrayFromImage(sitk.ReadImage(nifti_path))
        myo_mask = sitk.GetArrayFromImage(sitk.ReadImage(myo_mask_path))
        aankleuring_mask = sitk.GetArrayFromImage(sitk.ReadImage(aankleuring_mask_path))
        if LGE_img.squeeze().shape != myo_mask.shape or LGE_img.squeeze().shape != aankleuring_mask.shape:
            raise ValueError(f"Inconsistent shapes for {mode}-images {pat_id}. LGE-img: {LGE_img.shape}; myo-mask: {myo_mask.shape}; aankleuring-mask: {aankleuring_mask.shape}. {nifti_path} {myo_mask_path} {aankleuring_mask_path}")
        myo_mask = myo_mask.astype(np.int16)
        aankleuring_mask = aankleuring_mask.astype(np.int16)
        # insert one dimension to the existing data as image channel
        # if LGE_img.shape[0] != 1: LGE_img = np.expand_dims(LGE_img, axis=0)
        # if myo_mask.shape[0] != 1: myo_mask = np.expand_dims(myo_mask, axis=0)
        # if aankleuring_mask.shape[0] != 1: aankleuring_mask = np.expand_dims(aankleuring_mask, axis=0)
        LGE_img = LGE_img.squeeze()
        myo_mask = myo_mask.squeeze()
        aankleuring_mask = aankleuring_mask.squeeze()

        if resize == 'resize':
            # LGE_img = resizing(LGE_img)
            # myo_mask = resizing(myo_mask)
            # aankleuring_mask = resizing(aankleuring_mask)
            LGE_img, myo_mask, aankleuring_mask = resize_images(LGE_img, myo_mask, aankleuring_mask, size)

        LGE_imgs.append(LGE_img)
        myo_masks.append(myo_mask)
        aankleuring_masks.append(myo_mask)
    
    if resize == 'crop':
        LGE_imgs, myo_masks, aankleuring_masks = crop_imgs(LGE_imgs, myo_masks, aankleuring_masks)
    LGE_imgs = normalize(LGE_imgs)

    if not os.path.isfile(os.path.join(ORIGINAL_DIR_NAME, mode, BOUNDING_BOX_FILE)):
        header = ['pat_id', 'y1', 'y2', 'x1', 'x2']
        bounding_box_coordinates = compute_bounding_box(myo_masks)
        with open(os.path.join(ORIGINAL_DIR_NAME, mode, BOUNDING_BOX_FILE), 'w', newline='') as f:
            writer = csv.writer(f, delimiter=";")
            for i in range(len(pat_ids)):
                writer.writerow([pat_ids[i], bounding_box_coordinates[i][0], bounding_box_coordinates[i][1], bounding_box_coordinates[i][2], bounding_box_coordinates[i][3]])
    
    bounding_box_dict = {}
    with open(os.path.join(ORIGINAL_DIR_NAME, mode, BOUNDING_BOX_FILE), 'r') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            bounding_box_dict[row[0]] = (int(row[1]), int(row[2]), int(row[3]), int(row[4]))
    bounding_box_coordinates = []
    for pat_id in pat_ids:
        try:
            bounding_box_coordinates.append(np.array(bounding_box_dict[pat_id], dtype=np.int16))
        except:
            raise ValueError(f'Bouding box coordinates not found for subject {pat_id} in file {os.path.join(ORIGINAL_DIR_NAME, mode, BOUNDING_BOX_FILE)}')
    
    # plot_bounding_box(LGE_imgs[0], myo_masks[0], [5,6,7], bounding_box_coordinates[0])
    return LGE_imgs, myo_masks, aankleuring_masks, bounding_box_coordinates

def crop_imgs(LGE_images, myo_masks, aankleuring_masks):
    smallest_height, smallest_width = 1e6, 1e6
    for LGE_image in LGE_images:
        _, h, w = LGE_image.shape
        if h < smallest_height:
            smallest_height = h
        if w < smallest_width:
            smallest_width = w
    new_LGE_imgs, new_myo_masks, new_aankleuring_masks = [], [], []
    for i, (LGE_image, myo_mask, aankleuring_mask) in enumerate(zip(LGE_images, myo_masks, aankleuring_masks)):
        _, img_h, img_w = LGE_image.shape
        crop_height = int((img_h - smallest_height)/2)
        crop_width = int((img_w - smallest_width)/2)
        if crop_height != 0 and crop_width != 0:
            new_LGE_imgs.append(LGE_image[:, crop_height:-crop_height, crop_width: -crop_width])
            new_myo_masks.append(myo_mask[:, crop_height:-crop_height, crop_width: -crop_width])
            new_aankleuring_masks.append(aankleuring_mask[:, crop_height:-crop_height, crop_width: -crop_width])
        elif crop_height != 0:
            new_LGE_imgs.append(LGE_image[:, crop_height:-crop_height, :])
            new_myo_masks.append(myo_mask[:, crop_height:-crop_height, :])
            new_aankleuring_masks.append(aankleuring_mask[:, crop_height:-crop_height, :])
        elif crop_width != 0:
            new_LGE_imgs.append(LGE_image[:, :, crop_width: -crop_width])
            new_myo_masks.append(myo_mask[:, :, crop_width: -crop_width])
            new_aankleuring_masks.append(aankleuring_mask[:, :, crop_width: -crop_width])
        elif crop_height == 0 and crop_width == 0:
            new_LGE_imgs.append(LGE_image)
            new_myo_masks.append(myo_mask)
            new_aankleuring_masks.append(aankleuring_mask)
        else:
            raise ValueError(f"Invalid calculation for cropping width: {crop_width} and/or cropping heigth: {crop_height}")
    shape_list = [LGE.shape for LGE in new_LGE_imgs]
    # new_LGE_imgs = np.stack(new_LGE_imgs)
    # new_myo_masks = np.stack(new_myo_masks)
    # new_aankleuring_masks = np.stack(new_aankleuring_masks)
    return new_LGE_imgs, new_myo_masks, new_aankleuring_masks

def resize_images(LGE_images, myo_masks, aankleuring_masks, size):
    LGE_imgs_new, myo_masks_new, aankleuring_masks_new = np.zeros((LGE_images.shape[0], size[0], size[1])), np.zeros((LGE_images.shape[0], size[0], size[1])), np.zeros((LGE_images.shape[0], size[0], size[1]))
    for i in range(LGE_images.shape[0]):
        LGE_imgs_new[i] = cv2.resize(LGE_images[i], dsize=size, interpolation=cv2.INTER_LINEAR)
        myo_masks_new[i] = cv2.resize(myo_masks[i], dsize=size, interpolation=cv2.INTER_LINEAR)
        aankleuring_masks_new[i] = cv2.resize(aankleuring_masks[i], dsize=size, interpolation=cv2.INTER_LINEAR)
    return LGE_imgs_new, myo_masks_new, aankleuring_masks_new


def get_bounding_box_slice(arr):
    values = {}
    for row in range(arr.shape[0]):
        if 1 in arr[row]:
            values['y1'] = row
            break
        values['y1'] = int(1e3)
    for row in sorted(range(arr.shape[0]), reverse=True):
        if 1 in arr[row]:
            values['y2'] = row
            break
        values['y2'] = -1
    for column in range(arr.shape[1]):
        if 1 in arr[:, column]:
            values['x1'] = column
            break
        values['x1'] = int(1e3)
    for column in sorted(range(arr.shape[1]), reverse=True):
        if 1 in arr[:, column]:
            values['x2'] = column
            break
        values['x2'] = -1
    return values

def compute_bounding_box(mask_images):
    bouding_box_coordinates = []
    for mask in mask_images:
        upper_bound, left_bound = int(1e3), int(1e3)
        lower_bound, right_bound = 0, 0
        for i in range(mask.shape[0]):
            slice = mask[i, :, :]
            coords = get_bounding_box_slice(slice)
            if coords['y1'] < upper_bound: upper_bound = coords['y1']
            if coords['y2'] > lower_bound: lower_bound = coords['y2']
            if coords['x1'] < left_bound: left_bound = coords['x1']
            if coords['x2'] > right_bound: right_bound = coords['x2']
        bouding_box_coordinates.append((upper_bound, lower_bound, left_bound, right_bound))
    return bouding_box_coordinates

def plot_bounding_box(LGE_img, myo_mask=None, slices=[6], pred_box_values=None, true_box_values=None, plot='save', model_name=None):
    if plot not in ['save', 'show']:
        raise ValueError("plot parameter must be either 'save' or 'show'")
    LGE_img = LGE_img.squeeze()
    if myo_mask is not None:
        myo_mask = myo_mask.squeeze()
        for slice in slices:
            LGE_slice = LGE_img[slice, :, :]
            myo_slice = myo_mask[slice, :, :]
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=[10,20])
            plt.axis('off')
            ax1.imshow(LGE_slice)
            ax2.imshow(myo_slice)
            if true_box_values is not None:
                ymin, ymax, xmin, xmax = true_box_values.squeeze()
                box = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=3, edgecolor='y', facecolor='none')
                ax1.add_patch(box)
                box = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=3, edgecolor='y', facecolor='none')
                ax2.add_patch(box)
            if pred_box_values is not None:
                ymin, ymax, xmin, xmax = pred_box_values.squeeze()
                box = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=3, edgecolor='g', facecolor='none')
                ax1.add_patch(box)
                box = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=3, edgecolor='g', facecolor='none')
                ax2.add_patch(box)
            if plot == 'show':
                plt.show()
            if plot == 'save':
                file_name = model_name + '.png'
                plt.savefig(file_name, bbox_inches='tight', pad_inches=0)
                print('Image saved')
    else:
        for slice in slices:
            LGE_slice = LGE_img[slice, :, :]
            fig, ax1 = plt.subplots(1,1, figsize=[10,20])
            ax1.imshow(LGE_slice)
            if true_box_values is not None:
                ymin, ymax, xmin, xmax = true_box_values.squeeze()
                box = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=3, edgecolor='y', facecolor='none')
                ax1.add_patch(box)
            if pred_box_values is not None:
                ymin, ymax, xmin, xmax = pred_box_values.squeeze()
                box = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=3, edgecolor='g', facecolor='none')
                ax1.add_patch(box)
            if plot == 'show':
                plt.show()
            if plot == 'save':
                file_name = model_name + '.png'
                plt.savefig(file_name)

def get_data_paths(data_dir, modality=None):
    """
    Get get image data paths with corresponding modality (e.g. C0, DE, MASK)
    :param data_dir: the root data directories that contains PET, CT and MASK images
    :param modality: C0/DE/MASK
    :return: data paths
    """
    data_paths = []
    obj_names = next(os.walk(data_dir))[2]
    for fn in obj_names:
        path = os.path.join(data_dir, fn)
        data_paths.append(path)
    data_paths = sorted(data_paths)
    return data_paths

if __name__ == '__main__':
    read_in_AUMC_data('train')

