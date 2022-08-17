import argparse

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
base = parser.add_argument_group('Base options')
expr = parser.add_argument_group('Experiment parameters')
param = parser.add_argument_group('Hyperparameters')
dirs = parser.add_argument_group('Storage directories')
misc = parser.add_argument_group('Miscellaneous')
data = parser.add_argument_group('Dataset')

base.add_argument('--dataset', type=str, default='S3DIS')
base.add_argument('--data_dir', type=str, help='location of the dataset', default='/home/y/LongguangWang/Data/input_0.040')
expr.add_argument('--idx_test_area', type=int, help='Index of the Test Area', default=5)

expr.add_argument('--epochs', type=int, help='number of epochs', default=100)
expr.add_argument('--resume', type=int, help='model to load', default=0)

expr.add_argument('--augment_scale_min', type=float, help='model to load', default=0.8)
expr.add_argument('--augment_scale_max', type=float, help='model to load', default=1.2)
expr.add_argument('--augment_noise', type=float, help='model to load', default=0.001)

param.add_argument('--lr', type=float, help='learning rate of the optimizer', default=1e-2)
param.add_argument('--batch_size', type=int, help='batch size', default=6)
param.add_argument('--val_batch_size', type=int, help='batch size', default=6)
param.add_argument('--n_neighbors', type=int, help='number of neighbors considered by k-NN', default=16)
param.add_argument('--gamma', type=float, help='gamma of the learning rate scheduler', default=0.95)

dirs.add_argument('--logs_dir', type=str, help='path to tensorboard logs', default='runs')

misc.add_argument('--n_gpus', type=int, help='which GPU to use (-1 for CPU)', default=1)
misc.add_argument('--n_workers', type=int, help='number of threads for loading data', default=10)
misc.add_argument('--save_freq', type=int, help='frequency of saving checkpoints', default=1)
misc.add_argument('--train_steps', type=int, help='frequency of saving checkpoints', default=500)
misc.add_argument('--val_steps', type=int, help='frequency of saving checkpoints', default=100)

data.add_argument('--n_points', type=int, help='number of points', default=40960)
data.add_argument('--sub_ratio', type=float, help='number of points', default=0.04)
data.add_argument('--d_in', type=int, help='number of input features', default=6)
data.add_argument('--n_classes', type=int, help='number of class', default=13)

args = parser.parse_args()
args.class_weights = [3370714, 2856755, 4919229, 318158,  375640, 478001, 974733,
                      650464,  791496,  88727,   1284130, 229758, 2272837]

