import pickle, time, warnings
import torch
import os
import random
from torch.utils.data import Dataset, DataLoader, Sampler, BatchSampler
from torch.utils.data.dataset import IterableDataset
from utils.helper_tool import DataProcessing as DP
from nearest_neighbors import knn
from utils.helper_ply import *
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


class dataset_SemanticKITTI(Dataset):
    def __init__(self, args, mode):
        self.label_to_names = {0: 'unlabeled',
                               1: 'car',
                               2: 'bicycle',
                               3: 'motorcycle',
                               4: 'truck',
                               5: 'other-vehicle',
                               6: 'person',
                               7: 'bicyclist',
                               8: 'motorcyclist',
                               9: 'road',
                               10: 'parking',
                               11: 'sidewalk',
                               12: 'other-ground',
                               13: 'building',
                               14: 'fence',
                               15: 'vegetation',
                               16: 'trunk',
                               17: 'terrain',
                               18: 'pole',
                               19: 'traffic-sign'}
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.path = args.data_dir
        self.ignored_labels = np.sort([0])

        self.val_split = '08'

        self.train_list, self.val_list, self.test_list = DP.get_file_list(self.path, args.test_id)

        if mode == 'training':
            self.paths = DP.shuffle_list(self.train_list)
        elif mode == 'validation':
            self.paths = DP.shuffle_list(self.val_list)
        elif mode == 'test':
            self.paths = DP.shuffle_list(self.test_list)

        self.mode = mode
        self.size = len(self.paths)

        if mode != 'training':
            # random possibility
            self.possibility = []
            self.min_possibility = []
            for i in range(len(self.paths)):
                tree, points, labels = self.load_data(i)
                self.possibility += [torch.from_numpy(np.random.rand(tree.data.shape[0]) * 1e-3)]
                self.min_possibility += [float(torch.min(self.possibility[-1]))]

            for i in range(len(self.possibility)):
                self.possibility[i].share_memory_()
            self.min_possibility = torch.from_numpy(np.array(self.min_possibility))
            self.min_possibility.share_memory_()

            # threadlock
            self.worker_lock = Lock()

    def load_data(self, i):
        file_path = self.paths[i]
        seq_id = file_path.split('/')[-3]

        kdtree_file = file_path.replace('velodyne', 'KDTree')
        kdtree_file = kdtree_file.replace('npy', 'pkl')
        label_file = file_path.replace('velodyne', 'labels')

        # read pkl with search tree
        with open(kdtree_file, 'rb') as f:
            search_tree = pickle.load(f)
        points = np.array(search_tree.data, copy=False)

        # load labels
        if int(seq_id) >= 11:
            labels = np.zeros(np.shape(points)[0], dtype=np.uint8)
        else:
            label_path = os.path.join(label_file)
            labels = np.squeeze(np.load(label_path))

        return search_tree, points, labels

    def __getitem__(self, idx):
        pass

    def __len__(self):
        return self.size


class ActiveLearningSampler(IterableDataset):
    def __init__(self, dataset, args, batch_size=6):
        self.dataset = dataset
        self.args = args
        self.batch_size = batch_size

        if self.dataset.mode == 'training':
            self.n_samples = args.train_steps
            self.epochs = args.epochs
        elif self.dataset.mode == 'validation':
            self.n_samples = args.val_steps
            self.epochs = 1
        elif self.dataset.mode == 'test':
            self.n_samples = len(self.dataset.paths) // args.val_batch_size * 4
            self.epochs = 1

    def __iter__(self):
        return self.spatially_regular_gen()

    def __len__(self):
        return self.n_samples * self.batch_size

    def augmentation_transform(self, points):
        # rotations
        ## z-axis
        theta = np.random.rand() * np.pi * 2
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        augmented_points = np.sum(np.expand_dims(points, 2) * R, axis=1)

        ## x-axis
        phi = (np.random.rand() * 10 - 5) / 180 * np.pi
        c, s = np.cos(phi), np.sin(phi)
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
        augmented_points = np.sum(np.expand_dims(augmented_points, 2) * R, axis=1)

        ## y-axis
        alpha = (np.random.rand() * 10 - 5) / 180 * np.pi
        c, s = np.cos(alpha), np.sin(alpha)
        R = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], dtype=np.float32)
        augmented_points = np.sum(np.expand_dims(augmented_points, 2) * R, axis=1)

        # scaling
        min_s = self.args.augment_scale_min
        max_s = self.args.augment_scale_max
        scale = np.random.rand() * (max_s - min_s) + min_s
        augmented_points = augmented_points * scale

        return augmented_points

    def spatially_regular_gen(self):
        # choose the least known point as center of a new cloud each time.
        for _ in range(self.epochs * self.n_samples * self.batch_size // self.args.n_workers):
            if self.dataset.mode == 'training':
                # choose a random cloud
                cloud_ind = random.randint(0, len(self.dataset.paths)-1)
                tree, points, labels = self.dataset.load_data(cloud_ind)

                # choose a random point as the center point
                pick_idx = np.random.choice(len(points), 1)
                center_point = points[pick_idx, :].reshape(1, -1)

                # get N nearest points
                if len(points) < self.args.n_points:
                    queried_idx = tree.query(center_point, k=len(points))[1][0]
                else:
                    queried_idx = tree.query(center_point, k=self.args.n_points)[1][0]

                # collect points and colors
                queried_idx = DP.shuffle_idx(queried_idx)
                queried_pc_xyz = points[queried_idx]
                queried_pc_labels = labels[queried_idx]

            else:
                with self.dataset.worker_lock:
                    # choose a cloud with the minimum of possibility
                    cloud_idx = int(torch.argmin(self.dataset.min_possibility))

                    # choose the point with the minimum of possibility as query point
                    pick_idx = int(torch.argmin(self.dataset.possibility[cloud_idx]))

                    # load data
                    tree, points, labels = self.dataset.load_data(cloud_idx)

                    # get points from tree structure
                    points = np.array(tree.data, copy=False)

                    # center point of input region
                    center_point = points[pick_idx, :].reshape(1, -1)

                    if len(points) < self.args.n_points:
                        queried_idx = tree.query(center_point, k=len(points))[1][0]
                    else:
                        queried_idx = tree.query(center_point, k=self.args.n_points)[1][0]

                    queried_idx = DP.shuffle_idx(queried_idx)

                    # collect points and colors
                    queried_pc_xyz = points[queried_idx]
                    queried_pc_labels = labels[queried_idx]

                    # update possibility
                    dists = np.sum(np.square((points[queried_idx] - center_point).astype(np.float32)), axis=1)
                    delta = np.square(1 - dists / np.max(dists))
                    self.dataset.possibility[cloud_idx][queried_idx] += delta
                    self.dataset.min_possibility[cloud_idx] = float(torch.min(self.dataset.possibility[cloud_idx]))

            # data augmentation
            if self.dataset.mode == 'training':
                queried_pc_xyz = self.augmentation_transform(queried_pc_xyz)

            queried_pc_xyz = torch.from_numpy(queried_pc_xyz).float()
            queried_pc_labels = torch.from_numpy(queried_pc_labels).long()
            queried_pc = queried_pc_xyz

            # neighboring idx
            ## 1 -> 1
            xyz_1 = queried_pc_xyz
            neighbor_idx_1 = knn(xyz_1, xyz_1, self.args.n_neighbors, omp=True).astype(np.int32)
            neighbor_idx_1 = torch.from_numpy(neighbor_idx_1).long()

            ## 1/4 -> 1/4
            xyz_2 = xyz_1[:xyz_1.shape[0] // 4, :]
            neighbor_idx_2 = knn(xyz_2, xyz_2, self.args.n_neighbors, omp=True).astype(np.int32)
            neighbor_idx_2 = torch.from_numpy(neighbor_idx_2).long()

            ## 1/16 -> 1/16
            xyz_3 = xyz_2[:xyz_2.shape[0] // 4, :]
            neighbor_idx_3 = knn(xyz_3, xyz_3, self.args.n_neighbors, omp=True).astype(np.int32)
            neighbor_idx_3 = torch.from_numpy(neighbor_idx_3).long()

            ## 1/64 -> 1/64
            xyz_4 = xyz_3[:xyz_3.shape[0]//4, :]
            neighbor_idx_4 = knn(xyz_4, xyz_4, self.args.n_neighbors, omp=True).astype(np.int32)
            neighbor_idx_4 = torch.from_numpy(neighbor_idx_4).long()

            ## 1/64 -> 1/16
            up_idx_3 = knn(xyz_4, xyz_3, 1, omp=True).astype(np.int32)
            up_idx_3 = torch.from_numpy(up_idx_3).long()

            ## 1/16 -> 1/4
            up_idx_4 = knn(xyz_3, xyz_2, 1, omp=True).astype(np.int32)
            up_idx_4 = torch.from_numpy(up_idx_4).long()

            ## 1/4 -> 1
            up_idx_5 = knn(xyz_2, xyz_1, 1, omp=True).astype(np.int32)
            up_idx_5 = torch.from_numpy(up_idx_5).long()

            if self.dataset.mode == 'training':
                yield queried_pc, queried_pc_labels, \
                      [neighbor_idx_1, neighbor_idx_2, neighbor_idx_3, neighbor_idx_4,
                       up_idx_3, up_idx_4, up_idx_5]
            else:
                yield queried_pc, queried_pc_labels, \
                      [neighbor_idx_1, neighbor_idx_2, neighbor_idx_3, neighbor_idx_4,
                       up_idx_3, up_idx_4, up_idx_5], \
                      cloud_idx, queried_idx
