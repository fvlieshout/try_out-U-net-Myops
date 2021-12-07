from load_data import load_data
from train_ROI import ROIModel
from import_AUMC_dataset import plot_bounding_box
import argparse
import sys

def test(args):
    test_loader = load_data(dataset=args.dataset,
                                                    batch_size=args.batch_size,
                                                    num_workers=args.num_workers,
                                                    only_test=True)
    model = ROIModel.load_from_checkpoint(args.checkpoint_path)

    for test_data in tqdm(test_loader):
        LGE_image, _, _, bb_coordinates = test_data
        output = model.forward(LGE_image.float()).cpu().detach().numpy()
        plot_bounding_box(LGE_image, myo_mask=None, slices=[5], box_values=output)

if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Model hyperparameters
    parser.add_argument('--model', default='ResNet', type=str,
                        help='What model to use for the segmentation',
                        choices=['ResNet'])
    parser.add_argument('--in_channels', default=1, type=int,
                        help='Number of input channels for the convolutional networks.')
    parser.add_argument('--out_channels', default=1, type=int,
                        help='Number of output channels for the convolutional networks.')
    parser.add_argument('--dimension', default='3D', type=str,
                        help='What kind of model dimensions we want to use',
                        choices=['2D', '3D'])

    # Optimizer hyperparameters
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Minibatch size')

    # Other hyperparameters
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
    file_name = f'train_ROI_{datetime.now()}.txt'
    sys.stdout = open(os.path.join(args.print_dir, file_name), "w")
    train(args)
    sys.stdout.close()