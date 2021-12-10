from torch.utils.data.dataloader import DataLoader
from import_AUMC_dataset import read_in_AUMC_data
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import random
import numpy as np

class AUMCDataset3D(Dataset):
    def __init__(self, LGE_images, myops_masks, aankleuring_masks, bb_coordinates, transform=None) -> None:
        super().__init__()
        self.LGE_images = LGE_images
        self.myops_masks = myops_masks
        self.aankleuring_masks = aankleuring_masks
        self.bb_coordinates = bb_coordinates
        self.transform = transform
    
    def __len__(self):
        return len(self.bb_coordinates)

    def __getitem__(self, index):
        LGE_image = self.LGE_images[index]
        myops_mask = self.myops_masks[index]
        aankleuring_mask = self.aankleuring_masks[index]

        if self.transform is not None:
            LGE_image, myops_mask, aankleuring_mask = perform_transformations(self.transform, LGE_image, myops_mask, aankleuring_mask)

        return LGE_image, myops_mask, aankleuring_mask, self.bb_coordinates[index]
    
class AUMCDataset2D(Dataset):
    def __init__(self, LGE_images, myops_masks, aankleuring_masks, bb_coordinates, transform=None) -> None:
        super().__init__()
        self.LGE_images = LGE_images
        self.myops_masks = myops_masks
        self.aankleuring_masks = aankleuring_masks
        self.bb_coordinates = bb_coordinates
        self.transform = transform
    
    def __len__(self):
        return len(self.bb_coordinates)

    def __getitem__(self, index):
        LGE_image = torch.from_numpy(self.LGE_images[index])
        myops_mask = torch.from_numpy(self.myops_masks[index])
        aankleuring_mask = torch.from_numpy(self.aankleuring_masks[index])

        if self.transform is not None:
            LGE_image, myops_mask, aankleuring_mask = perform_transformations(self.transform, LGE_image, myops_mask, aankleuring_mask)

        return LGE_image, myops_mask, aankleuring_mask, self.bb_coordinates[index]


def load_data(dataset, batch_size=8, num_workers=1, only_test=False, transformations=[], resize=(256, 256)):
    if dataset == 'AUMC2D' or dataset == 'AUMC3D':
        image_folder = 'L:\\basic\\diva1\\Onderzoekers\\DEEP-RISK\\DEEP-RISK\\CMR DICOMS\\Roel&Floor\\sample_niftis\\labels\\labels_model_testing\\LGE_niftis'
        myo_label_folder = 'L:\\basic\\diva1\\Onderzoekers\\DEEP-RISK\\DEEP-RISK\\CMR DICOMS\\Roel&Floor\\sample_niftis\\labels\\labels_model_testing\\myo'
        aankleuring_label_folder = 'L:\\basic\\diva1\\Onderzoekers\\DEEP-RISK\\DEEP-RISK\\CMR DICOMS\\Roel&Floor\\sample_niftis\\labels\\labels_model_testing\\aankleuring'
    
        print(f"transformations: {transformations}, {type(transformations)}")
        if only_test:
            _, _, test_dataset = get_data(dataset, only_test=only_test, transforms=transformations, size=resize)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            return test_loader
        else:
            train_dataset, val_dataset, test_dataset = get_data(dataset, transforms=transformations, size=resize)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    elif dataset == 'Myops':
        print()
    else:
        raise ValueError(f"Dataset name {dataset} is not recognized. Choose either 'AUMC2D', 'AUMC3D' or 'Myops'")
    
    return train_loader, val_loader, test_loader

def get_data(dataset, val_size=0.2, seed=42, only_test=False, transforms=[], size=(256, 256)):
    if dataset == 'AUMC3D':
        if only_test:
            LGE_imgs_test, myo_masks_test, aankleuring_masks_test, bounding_box_coordinates_test = read_in_AUMC_data('test', resize='crop')
            test_dataset = AUMCDataset3D(LGE_imgs_test, myo_masks_test, aankleuring_masks_test, bounding_box_coordinates_test, transform=None)
            train_dataset, val_dataset = None, None
        else:
            LGE_imgs_train, myo_masks_train, aankleuring_masks_train, bounding_box_coordinates_train = read_in_AUMC_data('train', resize='crop')
            LGE_imgs_test, myo_masks_test, aankleuring_masks_test, bounding_box_coordinates_test = read_in_AUMC_data('test', resize='crop')
            validation_items = int(len(LGE_imgs_train) * val_size)
            validation_indices = random.sample(range(len(LGE_imgs_train)), validation_items)
            train_indices = [i for i in range(len(LGE_imgs_train)) if i not in validation_indices]
            LGE_img_val = [LGE_imgs_train[i] for i in validation_indices]
            myo_masks_val = [myo_masks_train[i] for i in validation_indices]
            aankleuring_masks_val = [aankleuring_masks_train[i] for i in validation_indices]
            bounding_box_coordinates_val = [bounding_box_coordinates_train[i] for i in validation_indices]
            LGE_imgs_train = [LGE_imgs_train[i] for i in train_indices]
            myo_masks_train = [myo_masks_train[i] for i in train_indices]
            aankleuring_masks_train = [aankleuring_masks_train[i] for i in train_indices]
            bounding_box_coordinates_train = [bounding_box_coordinates_train[i] for i in train_indices]
            train_dataset = AUMCDataset3D(LGE_imgs_train, myo_masks_train, aankleuring_masks_train, bounding_box_coordinates_train, transform=transforms)
            val_dataset = AUMCDataset3D(LGE_img_val, myo_masks_val, aankleuring_masks_val, bounding_box_coordinates_val, transform=None)
            test_dataset = AUMCDataset3D(LGE_imgs_test, myo_masks_test, aankleuring_masks_test, bounding_box_coordinates_test, transform=None)
    elif dataset == 'AUMC2D':
        if only_test:
            LGE_imgs_test, myo_masks_test, aankleuring_masks_test, bounding_box_coordinates_test = read_in_AUMC_data('test', resize='resize', size=size)
            bounding_box_coordinates_test = get_bouding_boxes_slices(LGE_imgs_test, bounding_box_coordinates_test)
            LGE_imgs_test = get_all_slices(LGE_imgs_test)
            myo_masks_test = get_all_slices(myo_masks_test)
            aankleuring_masks_test = get_all_slices(aankleuring_masks_test)
            test_dataset = AUMCDataset2D(LGE_imgs_test, myo_masks_test, aankleuring_masks_test, bounding_box_coordinates_test, transform=None)
            train_dataset, val_dataset = None, None
        else:
            LGE_imgs_train, myo_masks_train, aankleuring_masks_train, bounding_box_coordinates_train = read_in_AUMC_data('train', resize='resize', size=size)
            LGE_imgs_test, myo_masks_test, aankleuring_masks_test, bounding_box_coordinates_test = read_in_AUMC_data('test', resize='resize', size=size)
            bounding_box_coordinates_test = get_bouding_boxes_slices(LGE_imgs_test, bounding_box_coordinates_test)
            LGE_imgs_test = get_all_slices(LGE_imgs_test)
            myo_masks_test = get_all_slices(myo_masks_test)
            aankleuring_masks_test = get_all_slices(aankleuring_masks_test)
            bounding_box_coordinates_train = get_bouding_boxes_slices(LGE_imgs_train, bounding_box_coordinates_train)
            LGE_imgs_train = get_all_slices(LGE_imgs_train)
            myo_masks_train = get_all_slices(myo_masks_train)
            aankleuring_masks_train = get_all_slices(aankleuring_masks_train)
            validation_items = int(len(LGE_imgs_train) * val_size)
            validation_indices = random.sample(range(len(LGE_imgs_train)), validation_items)
            train_indices = [i for i in range(len(LGE_imgs_train)) if i not in validation_indices]
            LGE_img_val = [LGE_imgs_train[i] for i in validation_indices]
            myo_masks_val = [myo_masks_train[i] for i in validation_indices]
            aankleuring_masks_val = [aankleuring_masks_train[i] for i in validation_indices]
            bounding_box_coordinates_val = [bounding_box_coordinates_train[i] for i in validation_indices]
            LGE_imgs_train = [LGE_imgs_train[i] for i in train_indices]
            myo_masks_train = [myo_masks_train[i] for i in train_indices]
            aankleuring_masks_train = [aankleuring_masks_train[i] for i in train_indices]
            bounding_box_coordinates_train = [bounding_box_coordinates_train[i] for i in train_indices]
            train_dataset = AUMCDataset2D(LGE_imgs_train, myo_masks_train, aankleuring_masks_train, bounding_box_coordinates_train, transform=transformations)
            val_dataset = AUMCDataset2D(LGE_img_val, myo_masks_val, aankleuring_masks_val, bounding_box_coordinates_val, transform=None)
            test_dataset = AUMCDataset2D(LGE_imgs_test, myo_masks_test, aankleuring_masks_test, bounding_box_coordinates_test, transform=None)

            # validation_items = int(len(LGE_imgs_train) * val_size)
            # train_dataset, val_dataset = random_split(train_dataset, [len(LGE_imgs_train) - validation_items, validation_items], generator=torch.Generator().manual_seed(seed))
            # val_dataset.disable_transformations()
        return train_dataset, val_dataset, test_dataset

def perform_transformations(transform, LGE_image, myops_mask, aankleuring_mask):
    if 'hflip' in transform:
        if random.random() < 0.5:
            LGE_image = transforms.RandomHorizontalFlip(p=1)(LGE_image)
            myops_mask = transforms.RandomHorizontalFlip(p=1)(myops_mask)
            aankleuring_mask = transforms.RandomHorizontalFlip(p=1)(aankleuring_mask)
    if 'vflip' in transform:
        if random.random() < 0.5:
            LGE_image = transforms.RandomVerticalFlip(p=1)(LGE_image)
            myops_mask = transforms.RandomVerticalFlip(p=1)(myops_mask)
            aankleuring_mask = transforms.RandomVerticalFlip(p=1)(aankleuring_mask)
    if 'rotate' in transform:
        if random.random() < 0.5:
            times = random.choice([1, 2, 3])
            LGE_image = torch.rot90(LGE_image, times, [1,2])
            myops_mask = torch.rot90(myops_mask, times, [1,2])
            aankleuring_mask = torch.rot90(aankleuring_mask, times, [1,2])
    return LGE_image, myops_mask, aankleuring_mask

def get_all_slices(img_data):
    all_slices = []
    for img in img_data:
        splits = np.split(img, img.shape[0])
        all_slices.extend(splits)
    return all_slices

def get_bouding_boxes_slices(img_data, bounding_boxes):
    all_bounding_boxes = []
    for i, bb_value in enumerate(bounding_boxes):
        slices_count = img_data[i].shape[0]
        repeated_bb_values = np.repeat(np.expand_dims(bb_value, 0), slices_count, 0)
        all_bounding_boxes.extend(repeated_bb_values)
    return all_bounding_boxes
