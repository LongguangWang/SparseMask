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

base.add_argument('--dataset', type=str, default='SemanticKITTI')
base.add_argument('--data_dir', type=str, help='location of the dataset',
                  default='/home/y/LongguangWang/Data/SemanticKITTI/sequences_0.06')
expr.add_argument('--test_id', type=str, help='Index of the Test Area', default='21')

expr.add_argument('--epochs', type=int, help='number of epochs', default=100)
expr.add_argument('--resume', type=int, help='model to load', default=0)

expr.add_argument('--augment_scale_min', type=float, help='model to load', default=0.8)
expr.add_argument('--augment_scale_max', type=float, help='model to load', default=1.2)
expr.add_argument('--augment_noise', type=float, help='model to load', default=0.001)

param.add_argument('--lr', type=float, help='learning rate of the optimizer', default=1e-2)
param.add_argument('--batch_size', type=int, help='batch size', default=10)
param.add_argument('--val_batch_size', type=int, help='batch size', default=6)
param.add_argument('--n_neighbors', type=int, help='number of neighbors considered by k-NN', default=16)
param.add_argument('--gamma', type=float, help='gamma of the learning rate scheduler', default=0.95)

dirs.add_argument('--logs_dir', type=str, help='path to tensorboard logs', default='runs')

misc.add_argument('--n_gpus', type=int, help='which GPU to use (-1 for CPU)', default=1)
misc.add_argument('--name', type=str, help='name of the experiment', default='PCSM')
misc.add_argument('--n_workers', type=int, help='number of threads for loading data', default=10)
misc.add_argument('--save_freq', type=int, help='frequency of saving checkpoints', default=1)
misc.add_argument('--train_steps', type=int, help='frequency of saving checkpoints', default=3000)
misc.add_argument('--val_steps', type=int, help='frequency of saving checkpoints', default=500)

data.add_argument('--n_points', type=int, help='number of points', default=45056)
data.add_argument('--sub_ratio', type=float, help='number of points', default=0.06)
data.add_argument('--d_in', type=int, help='number of input features', default=3)
data.add_argument('--n_classes', type=int, help='number of class', default=19)

args = parser.parse_args()
args.class_weights = [55437630, 320797, 541736, 2578735, 3274484, 552662, 184064, 78858,
                      240942562, 17294618, 170599734, 6369672, 230413074, 101130274, 476491114, 9833174,
                      129609852, 4506626, 1168181]

