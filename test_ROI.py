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

def test(args, plot=None):
    device = torch.device(args.cuda_device) if torch.cuda.is_available() else torch.device("cpu")
    print("Device", device)
    loss_function = L1loss()
    test_loader = load_data(dataset=args.dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            only_test=True)
    model = ROIModel.load_from_checkpoint(args.checkpoint_path)
    model.to(device)
    model.eval()

    with torch.no_grad():
        for i, test_data in enumerate(test_loader):
            LGE_image, _, _, bb_coordinates = test_data
            LGE_image = LGE_image.to(device)
            bb_coordinates = bb_coordinates.to(device)
            output = model.forward(LGE_image.float())
            loss = loss_function(output, bb_coordinates)
            print('Test loss:', loss)
            if plot is not None:
                if plot == 'save':
                    save_dir = os.path.join('.', args.checkpoint_path.split('/')[0], args.checkpoint_path.split('/')[1], args.checkpoint_path.split('/')[2], 'test_imgs')
                    print(save_dir)
                    os.makedirs(save_dir, exist_ok=True)
                    file_name = os.path.join(save_dir, f"{args.checkpoint_path.split('/')[4].split('.')[0]}_[{i}]")
                else:
                    file_name = None
                plot_bounding_box(LGE_image.cpu().detach().numpy(), myo_mask=None, slices=[5], pred_box_values=output.cpu().detach().numpy(), true_box_values=bb_coordinates.cpu().detach().numpy(), plot=plot, model_name=file_name)

if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Optimizer hyperparameters
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Minibatch size')

    # Other hyperparameters
    parser.add_argument('--cuda_device', default='cuda', type=str,
                        help='Which GPU node to use if available',
                        choices=['cuda', 'cuda:1', 'cuda:2'])
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
    log_dir = args.checkpoint_path.split('/')[0]
    version_nr = int(args.checkpoint_path.split('/')[2].split('_')[-1])
    file_name = f'test_ROI_version_{version_nr}.txt'
    first_path = os.path.join(log_dir, 'lightning_logs', file_name)
    second_path = os.path.join(log_dir, 'lightning_logs', f"version_{version_nr}", file_name)
    sys.stdout = open(first_path, "w")
    test(args, plot='save')
    sys.stdout.close()
    os.rename(first_path, second_path)