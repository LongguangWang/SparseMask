import torch
import pickle, time, warnings
import glob
import random
import math
import torchvision.transforms as transforms
from utils.helper_ply import *
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
from torch.utils.data.dataset import IterableDataset
from utils.helper_tool import DataProcessing as DP
from nearest_neighbors import knn
from multiprocessing import Lock
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())


def data_loaders(dataset, args, **kwargs):
    batch_size = kwargs.get('batch_size', 6)
    sampler = ActiveLearningSampler(
        dataset,
        args,
        batch_size=batch_size,
    )

    return DataLoaderX(sampler, **kwargs)


class dataset_S3DIS(Dataset):
    def __init__(self, args, training=True):
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'table',
                               8: 'chair',
                               9: 'sofa',
                               10: 'bookcase',
                               11: 'board',
                               12: 'clutter'}
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])

        # paths
        self.path = args.data_dir
        if training:
            paths = []
            for i in range(6):  # 6 Areas
                if i+1 != args.idx_test_area:
                    paths += list(glob.glob(self.path + '/Area_' + str(i+1) + '*.ply'))
            self.paths = paths
        else:
            self.paths = list(glob.glob(self.path + '/Area_' + str(args.idx_test_area) + '*.ply'))

        self.training = training
        self.size = len(self.paths)
        self.input_trees = []
        self.input_colors = []
        self.input_labels = []
        self.input_names = []
        if not training:
            self.val_proj = []
            self.val_labels = []
            self.val_split = '1_'

        self.load_data()

        # random possibility
        self.possibility = []
        self.min_possibility = []
        for i, tree in enumerate(self.input_colors):
            self.possibility += [torch.from_numpy(np.random.rand(tree.data.shape[0]) * 1e-3)]
            self.min_possibility += [float(torch.min(self.possibility[-1]))]

        for i in range(len(self.possibility)):
            self.possibility[i].share_memory_()
        self.min_possibility = torch.from_numpy(np.array(self.min_possibility))
        self.min_possibility.share_memory_()

        # threadlock
        self.worker_lock = Lock()

    def load_data(self):
        # training data
        for i, file_path in enumerate(self.paths):
            cloud_name = file_path.split('/')[-1].split('.ply')[0]

            kd_tree_file = self.path + '/' + cloud_name + '_KDTree.pkl'
            sub_ply_file = self.path + '/' + cloud_name + '.ply'

            # read ply
            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            sub_labels = data['class']

            # read pkl
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees.append(search_tree)
            self.input_colors.append(sub_colors)
            self.input_labels.append(sub_labels)
            self.input_names.append(cloud_name)

        # validation data
        if not self.training:
            for i, file_path in enumerate(self.paths):
                cloud_name = file_path.split('/')[-1].split('.ply')[0]

                # read projection files
                proj_file = self.path + '/' + cloud_name + '_proj.pkl'
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)

                self.val_proj += [proj_idx]
                self.val_labels += [labels]

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return self.size


class ActiveLearningSampler(IterableDataset):
    def __init__(self, dataset, args, batch_size=6):
        self.dataset = dataset
        self.args = args
        self.batch_size = batch_size

        if self.dataset.training:
            self.n_samples = args.train_steps
            self.epochs = args.epochs
        else:
            self.n_samples = args.val_steps
            self.epochs = 1

    def __iter__(self):
        return self.spatially_regular_gen()

    def __len__(self):
        return self.n_samples * self.batch_size

    def augmentation_transform(self, points, colors):
        # random 2D rotations
        theta = np.random.rand() * np.pi * 2
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1)

        # random scaling
        min_s = self.args.augment_scale_min
        max_s = self.args.augment_scale_max
        scale = np.random.rand() * (max_s - min_s) + min_s
        augmented_points = augmented_points * scale

        # drop color
        aug_colors = colors
        # if random.random() > 0.8:
        #     aug_colors *= 0.0

        return augmented_points, aug_colors

    def spatially_regular_gen(self):
        # choose the least known point as center of a new cloud each time.
        for _ in range(self.epochs * self.n_samples * self.batch_size // self.args.n_workers):
            with self.dataset.worker_lock:
                # choose a cloud with the lowest possibility
                cloud_idx = int(torch.argmin(self.dataset.min_possibility))

                # choose the point with the minimum of possibility as query point
                point_ind = int(torch.argmin(self.dataset.possibility[cloud_idx]))

                # get points from tree structure
                points = np.array(self.dataset.input_trees[cloud_idx].data, copy=False)

                # center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # add noise to the center point
                noise = np.random.normal(scale=3.5 / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                # get N nearest points
                if len(points) < self.args.n_points:
                    queried_idx = self.dataset.input_trees[cloud_idx].query(pick_point, k=len(points))[1][0]
                else:
                    queried_idx = self.dataset.input_trees[cloud_idx].query(pick_point, k=self.args.n_points)[1][0]

                queried_idx = DP.shuffle_idx(queried_idx)

                # collect points and colors
                queried_pc_xyz = points[queried_idx]
                queried_pc_xyz = queried_pc_xyz - pick_point
                queried_pc_colors = self.dataset.input_colors[cloud_idx][queried_idx]
                queried_pc_labels = self.dataset.input_labels[cloud_idx][queried_idx]

                # update possibility
                dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.dataset.possibility[cloud_idx][queried_idx] += delta
                self.dataset.min_possibility[cloud_idx] = float(torch.min(self.dataset.possibility[cloud_idx]))

            if self.dataset.training:
                # data augmentation
                queried_pc_xyz, queried_pc_colors = self.augmentation_transform(queried_pc_xyz, queried_pc_colors)

            if len(points) < self.args.n_points:
                queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                    DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, self.args.n_points)

            queried_pc_xyz = torch.from_numpy(queried_pc_xyz).float()
            queried_pc_colors = torch.from_numpy(queried_pc_colors).float()
            queried_pc_labels = torch.from_numpy(queried_pc_labels).long()
            queried_pc = torch.cat((queried_pc_xyz, queried_pc_colors), 1)

            # neighboring idx
            ## 1 -> 1
            xyz_1 = queried_pc_xyz
            neighbor_idx_1 = knn(xyz_1, xyz_1, self.args.n_neighbors, omp=True).astype(np.int32)
            neighbor_idx_1 = torch.from_numpy(neighbor_idx_1).long()

            ## 1/4 -> 1/4
            xyz_2 = xyz_1[:xyz_1.shape[0]//4, :]
            neighbor_idx_2 = knn(xyz_2, xyz_2, self.args.n_neighbors, omp=True).astype(np.int32)
            neighbor_idx_2 = torch.from_numpy(neighbor_idx_2).long()

            ## 1/16 -> 1/16
            xyz_3 = xyz_2[:xyz_2.shape[0]//4, :]
            neighbor_idx_3 = knn(xyz_3, xyz_3, self.args.n_neighbors, omp=True).astype(np.int32)
            neighbor_idx_3 = torch.from_numpy(neighbor_idx_3).long()

            ## 1/64 -> 1/64
            xyz_4 = xyz_3[:xyz_3.shape[0]//4, :]
            neighbor_idx_4 = knn(xyz_4, xyz_4, self.args.n_neighbors, omp=True).astype(np.int32)
            neighbor_idx_4 = torch.from_numpy(neighbor_idx_4).long()

            ## 1/256 -> 1/256
            xyz_5 = xyz_4[:xyz_4.shape[0]//4, :]
            neighbor_idx_5 = knn(xyz_5, xyz_5, self.args.n_neighbors, omp=True).astype(np.int32)
            neighbor_idx_5 = torch.from_numpy(neighbor_idx_5).long()

            ## 1/256 -> 1/64
            up_idx_2 = knn(xyz_5, xyz_4, 1, omp=True).astype(np.int32)
            up_idx_2 = torch.from_numpy(up_idx_2).long()

            ## 1/64 -> 1/16
            up_idx_3 = knn(xyz_4, xyz_3, 1, omp=True).astype(np.int32)
            up_idx_3 = torch.from_numpy(up_idx_3).long()

            ## 1/16 -> 1/4
            up_idx_4 = knn(xyz_3, xyz_2, 1, omp=True).astype(np.int32)
            up_idx_4 = torch.from_numpy(up_idx_4).long()

            ## 1/4 -> 1
            up_idx_5 = knn(xyz_2, xyz_1, 1, omp=True).astype(np.int32)
            up_idx_5 = torch.from_numpy(up_idx_5).long()

            if self.dataset.training:
                yield queried_pc, queried_pc_labels, \
                      [neighbor_idx_1, neighbor_idx_2, neighbor_idx_3, neighbor_idx_4, neighbor_idx_5,
                       up_idx_2, up_idx_3, up_idx_4, up_idx_5]
            else:
                yield queried_pc, queried_pc_labels, \
                      [neighbor_idx_1, neighbor_idx_2, neighbor_idx_3, neighbor_idx_4, neighbor_idx_5,
                       up_idx_2, up_idx_3, up_idx_4, up_idx_5], \
                      cloud_idx, queried_idx