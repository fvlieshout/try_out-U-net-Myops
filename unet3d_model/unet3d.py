# _*_ coding: utf-8 _*_
# Author: Jielong
# @Time: 21/08/2019 15:52
import sys
import time
import torch
import torch.nn as nn
import numpy as np
from unet3d_model.building_components import EncoderBlock, DecoderBlock
sys.path.append("..")

cuda = True if torch.cuda.is_available() else False


class UnetModel(nn.Module):

    def __init__(self, in_channels, out_channels, model_depth=4, final_activation="sigmoid"):
        super(UnetModel, self).__init__()
        self.encoder = EncoderBlock(in_channels=in_channels, model_depth=model_depth)
        self.decoder = DecoderBlock(out_channels=out_channels, model_depth=model_depth)
        if final_activation == "sigmoid":
            self.sigmoid = nn.Sigmoid()
        else:
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x, downsampling_features = self.encoder(x)
        x = self.decoder(x, downsampling_features)
        x = self.sigmoid(x)
        print("Final output shape: ", x.shape)
        return x


class Trainer(object):

    def __init__(self, data_dirs, net, optimizer, criterion, no_epochs, batch_size=1):
        """
        Parameter initialization
        :param data_dir: folder that stores images for each modality
        :param net: the created model
        :param optimizer: the optimizer mode
        :param criterion: loss function
        :param no_epochs: number of epochs to train the model
        :param batch_size: batch size for generating data during training
        """
        self.data_dirs = data_dirs
        self.modalities = ["DE", "MASK"]
        self.net = net
        if torch.cuda.is_available():
            self.net.cuda()
        self.optimizer = optimizer
        self.criterion = criterion
        self.no_epochs = no_epochs
        self.batch_size = batch_size

    def train(self, data_paths_loader, dataset_loader, batch_data_loader):
        """
        Load corresponding data and start training
        :param data_paths_loader: get data paths ready for loading
        :param dataset_loader: get images and masks data
        :param batch_data_loader: generate batch data
        :return: None
        """
        # self.net.train()
        pet_paths = data_paths_loader(self.data_dirs, self.modalities[0])
        print(pet_paths)
        mask_paths = data_paths_loader(self.data_dirs, self.modalities[1])
        pets, masks = dataset_loader(pet_paths, mask_paths)
        training_steps = len(pets) // self.batch_size

        for epoch in range(self.no_epochs):
            start_time = time.time()
            train_losses, train_iou = 0, 0
            step = 0
            while step < training_steps:
            # for step in range(training_steps):
                print("Training step {}".format(step))

                x_batch, y_batch = batch_data_loader(pets, masks, iter_step=step, batch_size=self.batch_size)
                if x_batch[0].shape[1] < 6:
                    print(f'Slice number of {x_batch[0].shape[1]} is too small. Training example is skipped.')
                    continue

                x_batch = np.array(x_batch)
                y_batch = np.array(y_batch)
                x_batch = torch.from_numpy(x_batch)
                y_batch = torch.from_numpy(y_batch)
                if cuda:
                    x_batch = x_batch.cuda()
                    y_batch = y_batch.cuda()

                self.optimizer.zero_grad()

                logits = self.net(x_batch.float())
                y_batch = y_batch.type(torch.int8)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                # train_iou += mean_iou(y_batch, logits)
                train_losses += loss.item()
                step += 1
            end_time = time.time()
            print("Epoch {}, training loss {:.4f}, time {:.2f}".format(epoch, train_losses / training_steps,
                                                                       end_time - start_time))

    def predict(self):
        pass

    def _save_checkpoint(self):
        pass


if __name__ == "__main__":
    inputs = torch.randn(1, 1, 96, 96, 96)
    print("The shape of inputs: ", inputs.shape)
    data_folder = "../processed"
    model = UnetModel(in_channels=1, out_channels=1)
    inputs = inputs.cuda()
    model.cuda()
    x = model(inputs)
    print(model)
