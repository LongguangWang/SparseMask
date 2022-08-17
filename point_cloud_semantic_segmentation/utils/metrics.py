import numpy as np
import torch
from sklearn.metrics import confusion_matrix


class Evalulator:
    def __init__(self, num_classes, ignore=-1):
        self.gt_classes = [0 for _ in range(num_classes)]
        self.positive_classes = [0 for _ in range(num_classes)]
        self.true_positive_classes = [0 for _ in range(num_classes)]
        self.num_classes = num_classes
        self.ignore = ignore

    def add_data(self, logits, labels):
        pred = logits.max(dim=1)[1]                                 # B, N
        pred_valid = pred.detach().view(-1)                         # BN
        labels_valid = labels.detach().view(-1)
        pred_valid = pred_valid[labels_valid!=self.ignore]
        labels_valid = labels_valid[labels_valid!=self.ignore]

        val_total_correct = 0
        val_total_seen = 0

        correct = (pred_valid == labels_valid).sum()
        val_total_correct += correct.item()
        val_total_seen += len(labels_valid)

        conf_matrix = confusion_matrix(labels_valid.cpu(), pred_valid.cpu(), np.arange(0, self.num_classes, 1))
        self.gt_classes += np.sum(conf_matrix, axis=1)
        self.positive_classes += np.sum(conf_matrix, axis=0)
        self.true_positive_classes += np.diagonal(conf_matrix)

    def compute(self):
        iou_list = []
        acc_list = []
        for n in range(0, self.num_classes):
            if float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n]) != 0:
                iou = self.true_positive_classes[n] / float(self.gt_classes[n] + self.positive_classes[n] - self.true_positive_classes[n])
                iou_list.append(iou)
                acc = self.true_positive_classes[n] / float(self.gt_classes[n])
                acc_list.append(acc)
            else:
                iou_list.append(0.0)
                acc_list.append(0.0)
        mean_iou = sum(iou_list) / float(self.num_classes)
        mean_acc = sum(acc_list) / float(self.num_classes)
        OA = sum(self.true_positive_classes) / sum(self.gt_classes)
        return mean_iou, iou_list, mean_acc, acc_list, OA


def accuracy(scores, labels):
    r"""
        Compute the per-class accuracies and the overall accuracy # TODO: complete doc

        Parameters
        ----------
        scores: torch.FloatTensor, shape (B?, C, N)
            raw scores for each class
        labels: torch.LongTensor, shape (B?, N)
            ground truth labels

        Returns
        -------
        list of floats of length num_classes+1 (last item is overall accuracy)
    """
    num_classes = scores.size(-2) # we use -2 instead of 1 to enable arbitrary batch dimensions

    predictions = torch.max(scores, dim=-2).indices

    accuracies = []

    accuracy_mask = predictions == labels
    for label in range(num_classes):
        label_mask = labels == label
        per_class_accuracy = (accuracy_mask & label_mask).float().sum()
        per_class_accuracy /= label_mask.float().sum()
        accuracies.append(per_class_accuracy.cpu().item())
    # overall accuracy
    accuracies.append(accuracy_mask.float().mean().cpu().item())
    return accuracies


def intersection_over_union(scores, labels):
    r"""
        Compute the per-class IoU and the mean IoU # TODO: complete doc

        Parameters
        ----------
        scores: torch.FloatTensor, shape (B?, C, N)
            raw scores for each class
        labels: torch.LongTensor, shape (B?, N)
            ground truth labels

        Returns
        -------
        list of floats of length num_classes+1 (last item is mIoU)
    """
    num_classes = scores.size(-2) # we use -2 instead of 1 to enable arbitrary batch dimensions

    predictions = torch.max(scores, dim=-2).indices

    ious = []

    for label in range(num_classes):
        pred_mask = predictions == label
        labels_mask = labels == label
        iou = (pred_mask & labels_mask).float().sum() / (pred_mask | labels_mask).float().sum()
        ious.append(iou.cpu().item())
    ious.append(np.nanmean(ious))

    return ious
