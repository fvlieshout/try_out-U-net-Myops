import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from load_data import load_data
from unet3d_model.unet3d import UnetModel
from criterions import Diceloss

class SegmentationModel(pl.LightningsModule):

    def __init__(self, dimension, in_channels, out_channels, loss_function_string, lr):
        super().__init__()
        self.loss_function_string = loss_function_string
        self.test_val_mode = 'test'
        self.save_hyperparameters()
        if loss_function_string == 'dice':
            self.loss_function = Diceloss()
        else:
            raise ValueError(f"Loss function {loss_function_string} not known")

        if dimension == '3D':
            self.model = UnetModel(in_channels=in_channels, out_channels=out_channels)
    
    def forward(self, imgs):
        output = self.model(imgs)
        return output
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        imgs, masks = batch
        output = self.forward(imgs)
        loss = self.loss_function(output, masks)
        loss_name = f"train_{str(self.loss_function_string)}_loss"
        self.log(loss_name, loss, on_step=False, on_epoch=True)
        dice_name = f"train_{str(self.loss_function_string)}_dicescore"
        prediction = round(output)
        self.log(dice_name, self.dice_coef(prediction, masks).item(), on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        imgs, masks = batch
        output = self.forward(imgs)
        loss = self.loss_function(output, masks)
        loss_name = f"val_{str(self.loss_function_string)}_loss"
        self.log(loss_name, loss, on_step=False, on_epoch=True)
        dice_name = f"val_{str(self.loss_function_string)}_dicescore"
        prediction = round(output)
        self.log(dice_name, self.dice_coef(prediction, masks).item(), on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        # Make use of the forward function, and add logging statements
        imgs, masks = batch
        output = self.forward(imgs)
        loss = self.loss_function(output, masks)
        loss_name = f"test_{str(self.loss_function_string)}_loss"
        self.log(loss_name, loss, on_step=False, on_epoch=True)
        dice_name = f"test_{str(self.loss_function_string)}_dicescore"
        prediction = round(output)
        self.log(dice_name, self.dice_coef(prediction, masks).item(), on_step=False, on_epoch=True)
    
    @staticmethod
    def dice_coef(img, img2):
        if not ((img==0) | (img==1)).all() or not ((img2==0) | (img2==1)).all():
            raise ValueError("Images need to be binary")
        if img.shape != img2.shape:
            raise ValueError("Shape mismatch: img and img2 must have to be of the same shape.")
        else:
            
            lenIntersection=0
            
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    if ( np.array_equal(img[i][j],img2[i][j]) ):
                        lenIntersection+=1
             
            lenimg=img.shape[0]*img.shape[1]
            lenimg2=img2.shape[0]*img2.shape[1]  
            value = (2. * lenIntersection  / (lenimg + lenimg2))
        return value

    @staticmethod
    def accuracy(predictions, labels):
        # noinspection PyUnresolvedReferences
        return (labels == predictions.argmax(dim=-1)).float().mean()

class GenerateCallback(pl.Callback):
    def __init__(self, batch_size=64, every_n_epochs=5, save_to_disk=False):
        """
        Inputs:
            batch_size - Number of images to generate
            every_n_epochs - Only save those images every N epochs (otherwise tensorboard gets quite large)
            save_to_disk - If True, the samples and image means should be saved to disk as well.
        """
        super().__init__()
    
    def on_epoch_end(self, trainer, pl_module):
        """
        This function is called after every epoch.
        Call the save_and_sample function every N epochs.
        """
        if (trainer.current_epoch+1) % self.every_n_epochs == 0:
            self.sample_and_save(trainer, pl_module, trainer.current_epoch+1)

def train(args):

    os.makedirs(args.log_dir, exist_ok=True)
    train_loader, val_loader, test_loader = load_data(dataset=args.data_set,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.num_workers)
    val_loss = f"val_{str(args.loss_function)}_loss"
    train_loss = f"train_{str(args.loss_function)}_loss"
    trainer = pl.Trainer(default_root_dir=args.log_dir,
                         checkpoint_callback=ModelCheckpoint(save_weights_only=True, mode="min", monitor=val_loss),
                         gpus=1 if torch.cuda.is_available() else 0,
                         max_epochs=args.epochs,
                         progress_bar_refresh_rate=1 if args.progress_bar else 0) 
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Create model
    pl.seed_everything(args.seed)  # To be reproducible        
    model = SegmentationModel(dimension=args.dimension, in_channels=args.in_channels, out_channels=args.out_channels, loss_function=args.loss_function, lr=args.lr)
    trainer.fit(model, train_loader, val_loader)
    
    #Testing
    model = model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    test_result = trainer.test(model, test_dataloaders=test_loader, verbose=True)
    # test_dice, val_dice = evaluate(trainer, model, test_loader, val_loader)
    return test_result

def evaluate(trainer, model, test_dataloader, val_dataloader, loss_function):
    """
    Tests a model on test and validation set.
    Args:
        trainer (pl.Trainer) - Lightning trainer to use.
        model (DocumentClassifier) - The Lightning Module which should be used.
        test_dataloader (DataLoader) - Data loader for the test split.
        val_dataloader (DataLoader) - Data loader for the validation split.
    Returns:
        test_accuracy (float) - The achieved test accuracy.
        val_accuracy (float) - The achieved validation accuracy.
    """

    print('Testing model on validation and test ..........\n')

    test_start = time.time()

    model.test_val_mode = 'test'
    test_result = trainer.test(model, test_dataloaders=test_dataloader, verbose=False)[0]
    test_accuracy = test_result[f"test_{str(loss_function)}_dice"]

    model.test_val_mode = 'val'
    val_result = trainer.test(model, test_dataloaders=val_dataloader, verbose=False)[0]
    val_accuracy = val_result["test_accuracy"] if "val_accuracy" not in val_result else val_result["val_accuracy"]
    model.test_val_mode = 'test'

    test_end = time.time()
    test_elapsed = test_end - test_start

    print(f'\nRequired time for testing: {int(test_elapsed / 60)} minutes.\n')
    print(f'Test Results:\n test accuracy: {round(test_accuracy, 3)} ({test_accuracy})\n '
          f'validation accuracy: {round(val_accuracy, 3)} ({val_accuracy})'
          f'\n epochs: {trainer.current_epoch + 1}\n')

    return test_accuracy, val_accuracy

if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--model', default='Unet', type=str,
                        help='What model to use for the segmentation',
                        choices=['Unet', 'FCNN'])
    parser.add_argument('--in_channels', default=1, type=int,
                        help='Number of input channels for the convolutional networks.')
    parser.add_argument('--out_channels', default=1, type=int,
                        help='Number of output channels for the convolutional networks.')
    parser.add_argument('--dimension', default='3D', type=str,
                        help='What kind of model dimensions we want to use',
                        choices=['2D', '3D'])

    # Optimizer hyperparameters
    parser.add_argument('--loss_function', default='dice', type=str,
                        help='What loss funciton to use for the segmentation',
                        choices=['dice'])
    parser.add_argument('--lr', default=1e-3, type=float,
                        help='Learning rate to use')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--dataset', default='AUMC', type=str,
                        help='What dataset to use for the segmentation',
                        choices=['AUMC', 'Myops'])
    parser.add_argument('--epochs', default=80, type=int,
                        help='Max number of epochs')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=0, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0.')
    parser.add_argument('--log_dir', default='segment_logs', type=str,
                        help='Directory where the PyTorch Lightning logs should be created.')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))

    args = parser.parse_args()

    train(args)