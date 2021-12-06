import torch
from torch.nn import CrossEntropyLoss
from unet3d_model.unet3d import UnetModel, Trainer
# from unet3d_model.tmp import UNet
from unet3d_model.loss import DiceLoss
from data_gen import get_data_paths, data_gen, batch_data_gen


def train_main(data_folders, in_channels, out_channels, learning_rate, no_epochs):
    """
    Train module
    :param data_folders: data folder
    :param in_channels: the input channel of input images
    :param out_channels: the final output channel
    :param learning_rate: set learning rate for training
    :param no_epochs: number of epochs to train model
    :return: None
    """
    model = UnetModel(in_channels=in_channels, out_channels=out_channels)
    model = model.float()
    optim = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    criterion = DiceLoss()
    trainer = Trainer(data_dirs=data_folders, net=model, optimizer=optim, criterion=criterion, no_epochs=no_epochs)
    trainer.train(data_paths_loader=get_data_paths, dataset_loader=data_gen, batch_data_loader=batch_data_gen)


if __name__ == "__main__":
    dataset = 'myops'
    # dataset = 'MICCAI'
    if dataset == 'myops':
        data_dirs = {'C0' : 'H:\\Deep_Risk_Floor\Modellen\\Datasets\\MyoPS 2020 Dataset\\train25\\train25',
                        'DE' : 'H:\\Deep_Risk_Floor\Modellen\\Datasets\\MyoPS 2020 Dataset\\train25\\train25',
                        'MASK' : 'H:\\Deep_Risk_Floor\Modellen\\Datasets\\MyoPS 2020 Dataset\\train25_myops_gd\\train25_myops_gd'}
    elif dataset == 'MICCAI': #doesn't work yet
        data_dirs = {'C0' : 'H:\\Deep_Risk_Floor\\Modellen\\Datasets\\MICCAI-STACOM2012\\human\\training',
                        'DE' : 'H:\\Deep_Risk_Floor\Modellen\\Datasets\\MICCAI-STACOM2012\\human\\training',
                        'MASK' : 'H:\\Deep_Risk_Floor\Modellen\\Datasets\\MICCAI-STACOM2012\\human\\training'}
    "./processed"
    train_main(data_folders=data_dirs, in_channels=1, out_channels=1, learning_rate=0.0001, no_epochs=10)