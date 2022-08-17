import numpy as np
import torch.nn as nn
import warnings, pickle, yaml, os
from model.SMPointSeg import SMPointSeg
from utils.helper_ply import read_ply, write_ply
from utils.helper_tool import DataProcessing as DP
from utils.option_SemanticKITTI import args
from tqdm import tqdm
import torch
from data.SemanticKitti import dataset_SemanticKITTI, data_loaders
from sklearn.metrics import confusion_matrix
from os.path import join, exists, dirname, abspath
from utils.helper_tool import Plot
warnings.filterwarnings("ignore")


# read yaml file
kitti_yaml_file = './utils/semantic-kitti.yaml'
kitti_yaml = yaml.safe_load(open(kitti_yaml_file, 'r'))
remap_dict = kitti_yaml['learning_map_inv']

# make lookup table for mapping
max_key = max(remap_dict.keys())
remap_lut = np.zeros((max_key + 100), dtype=np.int32)
remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

remap_dict_val = kitti_yaml['learning_map']
max_key = max(remap_dict_val.keys())
remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
remap_lut_val[list(remap_dict_val.keys())] = list(remap_dict_val.values())


def test():
    print('######### Test Area ' + str(args.test_id) + ' #########')

    model = SMPointSeg(args.d_in, args.n_classes,
                       channels=[36, 72, 144, 288],
                       layers=[2, 2, 2, 2],
                       ratio=[4, 4, 4],
                       n_neighbors=args.n_neighbors,
                       radius=0.1).cuda()
    model = nn.DataParallel(model)

    test_dataset = dataset_SemanticKITTI(args, 'test')
    test_loader = data_loaders(test_dataset, args, batch_size=args.val_batch_size, num_workers=args.n_workers,
                               drop_last=True)

    ckpt = torch.load('runs/SemanticKITTI/epoch_100.pth')
    # ckpt = torch.load('runs/SemanticKITTI/SemanticKITTI.pth')   # use pre-trained models
    model.load_state_dict(ckpt, strict=False)

    # evaluation mode
    model.eval()
    for m in model.module.encoder:
        if hasattr(m, '_prepare'):
            m._prepare()

    step_id = 0
    test_smooth = 0.98

    test_probs = [np.zeros(shape=[l.shape[0], args.n_classes], dtype=np.float32)
                  for l in test_dataset.possibility]

    with torch.no_grad():
        while test_dataset.min_possibility.min().item() < 0.5:
            for points, labels, neighbor_idx, cloud_idx, point_idx in tqdm(test_loader, desc='Test', leave=False):
                points = points.cuda()
                neighbor_idx = [idx.cuda() for idx in neighbor_idx]

                # inference
                stacked_probs = model([points, neighbor_idx])
                stacked_probs = stacked_probs.softmax(1)

                # visualization
                color = Plot.random_colors(19)
                predictions = torch.max(stacked_probs, dim=-2)[1]
                Plot.draw_pc_sem_ins(points.permute(0, 2, 1)[0, :, :3].data.cpu(), predictions[0, :].data.cpu(), color)

                stacked_probs = stacked_probs.data.cpu().numpy()
                stacked_probs = np.transpose(stacked_probs, (0, 2, 1))
                cloud_idx = cloud_idx.data.cpu().numpy()
                point_idx = point_idx.data.cpu().numpy()

                # update probability in test_probs
                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]
                    p_idx = point_idx[j, :]
                    c_i = cloud_idx[j]
                    test_probs[c_i][p_idx] = test_smooth * test_probs[c_i][p_idx] + (1 - test_smooth) * probs

                step_id += 1

        gt_calsses = [0 for _ in range(args.n_classes)]
        pos_calsses = [0 for _ in range(args.n_classes)]
        true_pos_calsses = [0 for _ in range(args.n_classes)]
        total_correct = 0
        total_seen = 0

        for j in range(len(test_probs)):
            # load projection file
            test_file_name = test_dataset.paths[j]
            test_id = test_file_name.split('/')[-1].split('.npy')[0]
            proj_file = test_file_name[:-19] + '/proj/' + test_id + '_proj.pkl'
            with open(proj_file, 'rb') as f:
                proj_inds = pickle.load(f)

            # reproject probs back to the evaluations points
            probs = test_probs[j][proj_inds[0], :]
            pred = np.argmax(probs, 1)

            if args.mode == 'validation':
                # calculate metrics
                label_file = test_file_name.replace('velodyne', 'labels')
                labels = DP.load_label_kitti(label_file, remap_lut_val)
                invalid_idx = np.where(labels == 0)[0]
                labels_valid = np.delete(labels, invalid_idx)
                pred_valid = np.delete(pred, invalid_idx)
                labels_valid = labels_valid + 1
                correct = np.sum(pred_valid == labels_valid)
                total_correct += correct
                total_seen += len(labels_valid)

                conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, args.n_classes, 1))
                gt_calsses += np.sum(conf_matrix, axis=1)
                pos_calsses += np.sum(conf_matrix, axis=0)
                true_pos_calsses += np.diagonal(conf_matrix)

                iou_list = []
                for n in range(0, args.n_classes, 1):
                    iou = true_pos_calsses[n] / float(gt_calsses[n] + pos_calsses[n] - true_pos_calsses[n])
                    iou_list.append(iou)
                mean_iou = sum(iou_list) / float(args.n_classes)

                print('acc: ' + str(total_correct / float(total_seen)))
                s = '{:5.2f} | '.format(100 * mean_iou)
                for iou in iou_list:
                    s += '{:5.2f} '.format(100 * iou)
                print(s)

            else:
                # save results
                os.mkdir('./runs/preds') if not os.path.exists('./runs/preds') else None
                os.mkdir('./runs/preds/SemanticKITTI') if not os.path.exists('./runs/preds/SemanticKITTI') else None
                os.mkdir('./runs/preds/SemanticKITTI/sequences/') if not os.path.exists('./runs/preds/SemanticKITTI/sequences/') else None
                os.mkdir('./runs/preds/SemanticKITTI/sequences/' + args.test_id) if not os.path.exists('./runs/preds/SemanticKITTI/sequences/' + args.test_id) else None

                save_path = './runs/preds/SemanticKITTI/sequences/' + args.test_id + '/predictions'
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
                pred = pred + 1
                pred = pred.astype(np.uint32)
                upper_half = pred >> 16
                lower_half = pred & 0xFFFF
                lower_half = remap_lut[lower_half]
                pred = (upper_half << 16) + lower_half
                pred = pred.astype(np.uint32)
                pred.tofile(save_path + '/' + test_id + '.label')



if __name__ == '__main__':
    test()