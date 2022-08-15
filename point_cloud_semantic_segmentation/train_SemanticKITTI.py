import torch
import torch.nn as nn
import warnings, os, time
from utils.metrics import accuracy, intersection_over_union, Evalulator
from model.SMPointSeg import SMPointSeg
from utils.option_SemanticKITTI import args
from data.SemanticKitti import dataset_SemanticKITTI, data_loaders
from utils.helper_tool import *
warnings.filterwarnings("ignore")


def evaluate(args, model, loader, criterion, log_file):
    model.eval()
    losses = []
    sparsity_list = []
    evaluator = Evalulator(args.n_classes)
    time_val = time.time()

    with torch.no_grad():
        i = 0
        for _, (points, labels, neighbor_idx) in enumerate(loader):
            i += 1
            process_bar(i / args.val_steps, start_str='Validation:  ', end_str='100%')

            points = points.cuda()
            labels = labels.cuda() - 1
            neighbor_idx = [idx.cuda() for idx in neighbor_idx]

            # inference
            scores, sparsity = model([points, neighbor_idx])
            sparsity_list.append(sparsity)

            # losses
            loss = criterion(scores, labels)
            losses.append(loss.cpu().item())

            # metrics
            evaluator.add_data(scores, labels)

            # display results
            if i % args.val_steps == 0:
                time_val = time.time() - time_val

                log_file.write(f'Test Sparsity: {torch.stack(sparsity_list).mean().data.cpu().item(): .2f}')
                log_file.write('Test Time:  ' + '{:.0f} s'.format(time_val))
                mean_iou, iou_list, mean_acc, acc_list, OA = evaluator.compute()
                acc_list.append(float(mean_acc))
                iou_list.append(float(mean_iou))

                return np.mean(losses), np.array(acc_list), np.array(iou_list)


def train(args, log_file):
    # dataloader
    train_dataset = dataset_SemanticKITTI(args, mode='training')
    train_loader = data_loaders(train_dataset, args, batch_size=args.batch_size, num_workers=args.n_workers, pin_memory=True, drop_last=True)
    val_dataset = dataset_SemanticKITTI(args, mode='validation')
    val_loader = data_loaders(val_dataset, args, batch_size=args.val_batch_size, num_workers=args.n_workers, pin_memory=True, drop_last=True)

    # model
    model = SMPointSeg(args.d_in, args.n_classes,
                       channels=[36, 72, 144, 288],
                       layers=[2, 2, 2, 2],
                       ratio=[4, 4, 4],
                       n_neighbors=args.n_neighbors,
                       radius=0.1).cuda()
    model = nn.DataParallel(model, range(args.n_gpus))

    # loss
    class_weights = torch.tensor(args.class_weights, dtype=torch.float).cuda()
    class_weights = class_weights / class_weights.sum()
    weights = 1 / (class_weights + 0.02)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=-1)

    # optimizer & scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)

    # resume
    if args.resume > 0:
        resume_file = args.logs_dir + '/' + args.dataset + '/epoch_' + str(args.resume) + '.pth'
        log_file.write(f'Loading {resume_file}...')
        ckpt = torch.load(resume_file)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])

    # train
    model.train()

    epoch = args.resume + 1
    log_file.write(f'========= EPOCH {epoch:d}/{args.epochs:d} =========')

    # update tau
    tau = max(1 - (epoch - 1) / 30, 0.1)
    for m in model.modules():
        if hasattr(m, '_update_tau'):
            m._update_tau(tau)

    # metrics
    losses = []
    accuracies = []
    ious = []
    sparsity_list = []
    i = 0
    time_train = time.time()

    # iteration over the dataset
    for _, (points, labels, neighbor_idx) in enumerate(train_loader):
        # progress bar
        i += 1
        process_bar(i / args.train_steps, start_str='Training:  ', end_str='100%')

        points = points.cuda()
        labels = labels.cuda() - 1
        neighbor_idx = [idx.cuda() for idx in neighbor_idx]

        optimizer.zero_grad()

        # forward
        scores, sparsity = model([points, neighbor_idx])
        sparsity_list.append(sparsity)

        # calculate loss
        loss = criterion(scores, labels) + 0.06 * sparsity.mean()

        # backward
        loss.backward()
        optimizer.step()

        # metrics
        losses.append(loss.cpu().item())
        accuracies.append(accuracy(scores, labels))
        ious.append(intersection_over_union(scores, labels))

        # one epoch
        if i % args.train_steps == 0:
            scheduler.step()

            # metrics
            accs = np.nanmean(np.array(accuracies), axis=0)
            ious = np.nanmean(np.array(ious), axis=0)

            # save model
            if epoch % args.save_freq == 0:
                torch.save(
                    dict(
                        epoch=epoch,
                        model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        scheduler_state_dict=scheduler.state_dict()
                    ),
                    args.logs_dir + '/' + args.dataset + '/epoch_' + str(epoch) + '.pth'
                )
            time_train = time.time() - time_train

            # evaluation on validation set
            val_loss, val_accs, val_ious = evaluate(args, model, val_loader, criterion, log_file)

            # display results
            log_file.write(f'Training loss: {np.mean(losses):.7f}')
            log_file.write(f'Training Sparsity: {torch.stack(sparsity_list).mean().data.cpu().item(): .2f}')
            log_file.write('Training Time:  ' + '{:.0f} s'.format(time_train))
            log_file.write('Training:    ' + ''.join([f'{i:>5d}|' for i in range(args.n_classes)]) + '   OA')
            log_file.write('Accuracy:    ' + ''.join([f'{acc:.3f}|' if not np.isnan(acc) else '  nan' for acc in accs]))
            log_file.write('IoU:         ' + ''.join([f'{iou:.3f}|' if not np.isnan(iou) else '  nan' for iou in ious]))
            log_file.write('Validation   ' + ''.join([f'{i:>5d}|' for i in range(args.n_classes)]) + '   OA')
            log_file.write('Accuracy:    ' + ''.join([f'{acc:.3f}|' if not np.isnan(acc) else '  nan' for acc in val_accs]))
            log_file.write('IoU:         ' + ''.join([f'{iou:.3f}|' if not np.isnan(iou) else '  nan' for iou in val_ious]))

            i = 0
            epoch += 1

            if epoch > args.epochs:
                break
            else:
                # update tau
                tau = max(1 - (epoch - 1) / 30, 0.1)
                for m in model.modules():
                    if hasattr(m, '_update_tau'):
                        m._update_tau(tau)

                time_train = time.time()
                losses = []
                accuracies = []
                ious = []
                sparsity_list = []
                log_file.write(f'========= EPOCH {epoch:d}/{args.epochs:d} =========')


if __name__ == '__main__':
    log_file = checkpoint(args)

    t0 = time.time()
    train(args, log_file)
    t1 = time.time()
    d = t1 - t0

    log_file.done()

    print('Done. Time elapsed:', '{:.0f} s.'.format(d) if d < 60 else '{:.0f} min {:.0f} s.'.format(*divmod(d, 60)))
