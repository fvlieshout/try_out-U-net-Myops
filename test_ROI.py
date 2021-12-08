from load_data import load_data
from train_ROI import ROIModel
from import_AUMC_dataset import plot_bounding_box
from criterions import L1loss
import torch
import tqdm
from datetime import datetime
import os
import argparse
import sys

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)

def test(args, plot=None):
    loss_function = L1loss()
    test_loader = load_data(dataset=args.dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            only_test=True)
    model = ROIModel.load_from_checkpoint(args.checkpoint_path)
    model.eval()

    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            LGE_image, _, _, bb_coordinates = test_data
            output = model.forward(LGE_image.float())
            loss = loss_function(output, bb_coordinates)
            print('Test loss:', loss)
            if plot is not None:
                if plot == 'save':
                    save_dir = os.path.join('.', args.checkpoint_path.split('/')[0], args.checkpoint_path.split('/')[1], args.checkpoint_path.split('/')[2], 'test_imgs')
                    os.makedirs(save_dir, exist_ok=True)
                    file_name = os.path.join(save_dir, f"{args.checkpoint_path.split('/')[4].split('.')[0]}_[{i}]")
                else:
                    file_name = None
                plot_bounding_box(LGE_image, myo_mask=None, slices=[5], pred_box_values=output.cpu().detach().numpy(), true_box_values=bb_coordinates.cpu().detach().numpy(), plot=plot, model_name=file_name)

if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Optimizer hyperparameters
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--dataset', default='AUMC', type=str,
                        help='What dataset to use for the segmentation',
                        choices=['AUMC', 'Myops'])
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed to use for reproducing results')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of workers to use in the data loaders. To have a truly deterministic run, this has to be 0.')
    parser.add_argument('--checkpoint_path', default='ROI_logs/lightning_logs/version_1/checkpoints/epoch=79-step=1519.ckpt', type=str,
                        help='Directory where the PyTorch Lightning checkpoint is created.')
    parser.add_argument('--print_dir', default='output', type=str,
                        help='Directory where the printing files should be created.')
    parser.add_argument('--progress_bar', action='store_true',
                        help=('Use a progress bar indicator for interactive experimentation. '
                              'Not to be used in conjuction with SLURM jobs'))

    args = parser.parse_args()

    #write prints to file
    file_name = f'test_ROI_{datetime.now()}.txt'
    if str(device) == 'cuda:0':
        sys.stdout = open(os.path.join(args.print_dir, file_name), "w")
    else:
        file_name = file_name.replace(':', ';')
        sys.stdout = open(os.path.join('.', args.print_dir, file_name), "w")
    test(args, plot='save')
    sys.stdout.close()