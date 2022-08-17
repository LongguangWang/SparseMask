import numpy as np
import time
import torch
import torch.nn as nn
import os
import warnings
from model.SMPointSeg import SMPointSeg
from utils.helper_ply import read_ply, write_ply
from utils.helper_tool import DataProcessing as DP
from utils.option_S3DIS import args
from tqdm import tqdm
from data.S3DIS import dataset_S3DIS, data_loaders
from sklearn.metrics import confusion_matrix
from os.path import exists, join
from utils.helper_tool import Plot
warnings.filterwarnings("ignore")
torch.backends.cudnn.benchmark = True


def test():
    print('######### Test Area ' + str(args.idx_test_area) + ' #########')

    model = SMPointSeg(args.d_in, args.n_classes,
                       channels=[32, 64, 128, 256, 512],
                       layers=[2, 2, 2, 2, 2],
                       n_neighbors=args.n_neighbors).cuda()
    model = nn.DataParallel(model)

    val_dataset = dataset_S3DIS(args, training=False)
    val_loader = data_loaders(val_dataset, args, batch_size=args.val_batch_size, num_workers=args.n_workers,
                              drop_last=True)
    
    ckpt = torch.load('runs/S3DIS/Area'+str(args.idx_test_area)+'/epoch_100.pth')
    # ckpt = torch.load('runs/S3DIS/Area'+str(args.idx_test_area)+'.pth')   # use pre-trained models
    model.load_state_dict(ckpt, strict=False)

    # evaluation mode
    model.eval()
    for m in model.module.encoder:
        if hasattr(m, '_prepare'):
            m._prepare()

    step_id = 0
    last_min = -0.5
    num_votes = 100
    test_smooth = 0.95

    test_probs = [np.zeros(shape=[l.shape[0], args.n_classes], dtype=np.float32)
                  for l in val_dataset.input_labels]
    val_proportions = np.zeros(args.n_classes, dtype=np.float32)

    for i, label_val in enumerate(range(args.n_classes)):
        val_proportions[i] = np.sum([np.sum(labels == label_val) for labels in val_dataset.input_labels])

    with torch.no_grad():
        while last_min < num_votes:
            for points, labels, neighbor_idx, cloud_idx, point_idx in tqdm(val_loader, desc='Test', leave=False):
                points = points.cuda()
                neighbor_idx = [idx.cuda() for idx in neighbor_idx]

                # inference
                stacked_probs = model([points, neighbor_idx])
                stacked_probs = stacked_probs.softmax(1)

                # visualization
                color = Plot.random_colors(13)
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

            new_min = np.min(val_dataset.min_possibility.data.cpu().numpy())
            if last_min + 1 < new_min:
                # update last_min
                last_min += 1

                confusion_list = []
                num_val = len(val_dataset.input_labels)

                for i_test in range(num_val):
                    probs = test_probs[i_test]
                    preds = np.argmax(probs, axis=1).astype(np.int32)
                    labels = val_dataset.input_labels[i_test]

                    # confs
                    confusion_list += [confusion_matrix(labels, preds, val_dataset.label_values)]

                # regroup confusions
                C = np.sum(np.stack(confusion_list), axis=0).astype(np.float32)

                # rescale with the right number of point per class
                C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                # compute IoUs
                IoUs = DP.IoU_from_confusions(C)
                m_IoU = np.mean(IoUs)
                s = '{:5.2f} | '.format(100 * m_IoU)
                for IoU in IoUs:
                    s += '{:5.2f} '.format(100 * IoU)

                # reproject probs back to the evaluations points
                proj_probs_list = []
                for i_val in range(num_val):
                    proj_idx = val_dataset.val_proj[i_val]
                    probs = test_probs[i_val][proj_idx, :]
                    proj_probs_list += [probs]

                confusion_list = []
                for i_test in range(num_val):
                    # get the predicted labels
                    preds = np.argmax(proj_probs_list[i_test], axis=1).astype(np.uint8)

                    # confusion
                    labels = val_dataset.val_labels[i_test]

                    confusion_list += [confusion_matrix(labels, preds, val_dataset.label_values)]
                    name = val_dataset.input_names[i_test] + '.ply'

                    # save results
                    os.mkdir('./runs/preds') if not os.path.exists('./runs/preds') else None
                    os.mkdir('./runs/preds/S3DIS') if not os.path.exists('./runs/preds/S3DIS') else None

                    write_ply(join('runs/preds/S3DIS', name), [preds, labels], ['pred', 'label'])

                # regroup confusions
                C = np.sum(np.stack(confusion_list), axis=0)

                # calculate miou
                IoUs = DP.IoU_from_confusions(C)
                m_IoU = np.mean(IoUs)
                s = '{:5.2f} | '.format(100 * m_IoU)
                for IoU in IoUs:
                    s += '{:5.2f} '.format(100 * IoU)

                print('finished \n')
                print(s)
                return

            step_id = 0



if __name__ == '__main__':
    test()