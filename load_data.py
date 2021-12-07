from torch.utils.data.dataloader import DataLoader
from import_AUMC_dataset import read_in_AUMC_data
import torch
from torch.utils.data import Dataset, random_split

class AUMCDataset(Dataset):
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
        if self.transform:
            LGE_image, myops_mask, aankleuring_mask = self.transform(LGE_image, myops_mask, aankleuring_mask)
        return LGE_image, myops_mask, aankleuring_mask, self.bb_coordinates[index]


def load_data(dataset, batch_size=8, num_workers=1, only_test=False):
    if dataset == 'AUMC':
        image_folder = 'L:\\basic\\diva1\\Onderzoekers\\DEEP-RISK\\DEEP-RISK\\CMR DICOMS\\Roel&Floor\\sample_niftis\\labels\\labels_model_testing\\LGE_niftis'
        myo_label_folder = 'L:\\basic\\diva1\\Onderzoekers\\DEEP-RISK\\DEEP-RISK\\CMR DICOMS\\Roel&Floor\\sample_niftis\\labels\\labels_model_testing\\myo'
        aankleuring_label_folder = 'L:\\basic\\diva1\\Onderzoekers\\DEEP-RISK\\DEEP-RISK\\CMR DICOMS\\Roel&Floor\\sample_niftis\\labels\\labels_model_testing\\aankleuring'

        if only_test:
            _, _, test_dataset = get_data(dataset, only_test)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            return test_loader
        else:
            train_dataset, val_dataset, test_dataset = get_data(dataset)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    elif dataset == 'Myops':
        print()
    else:
        raise ValueError(f"Dataset name {dataset} is not recognized. Choose either 'AUMC' or 'Myops'")
    
    return train_loader, val_loader, test_loader

def get_data(dataset, val_size=0.2, seed=42, only_test=False):
    if dataset == 'AUMC':
        if only_test:
            test_dataset = AUMCDataset(LGE_imgs_test, myo_masks_test, aankleuring_masks_test, bounding_box_coordinates_test, transform=None)
            train_dataset, val_dataset = None, None
        else:
            LGE_imgs_train, myo_masks_train, aankleuring_masks_train, bounding_box_coordinates_train = read_in_AUMC_data('train')
            LGE_imgs_test, myo_masks_test, aankleuring_masks_test, bounding_box_coordinates_test = read_in_AUMC_data('test')
            train_dataset = AUMCDataset(LGE_imgs_train, myo_masks_train, aankleuring_masks_train, bounding_box_coordinates_train, transform=None)
            test_dataset = AUMCDataset(LGE_imgs_test, myo_masks_test, aankleuring_masks_test, bounding_box_coordinates_test, transform=None)

            validation_items = int(len(LGE_imgs_train) * val_size)
            train_dataset, val_dataset = random_split(train_dataset, [len(LGE_imgs_train) - validation_items, validation_items], generator=torch.Generator().manual_seed(seed))
        return train_dataset, val_dataset, test_dataset