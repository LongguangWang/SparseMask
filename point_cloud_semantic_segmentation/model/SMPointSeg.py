import torch
import torch.nn as nn
from model.modules import PointConv_SM, gumbel_softmax, batch_gather
from utils.helper_tool import Plot
import matplotlib.pyplot as plt
import torch.nn.functional as F


def subsampling(fea, knn_idx, scale_factor):
    b, c, n = fea.shape
    sub_fea = batch_gather(fea.squeeze(-1), knn_idx[:, :n//scale_factor, :]).mean(2)
    return sub_fea


def upsampling(fea, knn_idx):
    up_fea = batch_gather(fea, knn_idx).squeeze(2)
    return up_fea


class BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_neighbors, kernel_size=5, bias=False, bn=False, relu=False, sparse=False):
        super(BasicConv, self).__init__()
        self.bn = bn
        self.relu = relu
        self.sparse = sparse
        body = [PointConv_SM(in_channels, out_channels, n_neighbors, kernel_size, bias, sparse=sparse)]

        if bn:
            body.append(nn.BatchNorm1d(out_channels))

        if relu:
            body.append(nn.ReLU(True))

        self.body = nn.Sequential(*body)

    def _prepare(self, ch_mask_in, ch_mask_out):    # 1 for sparse, 0 for dense
        if self.sparse:
            self.conv = self.body[0]
            self.conv._prepare(ch_mask_in, ch_mask_out)
            d_num = int((ch_mask_out==0).sum())
            s_num = int((ch_mask_out==1).sum())

            if self.bn:
                if d_num > 0:
                    self.bn_d = nn.BatchNorm1d(d_num).to(ch_mask_out.device)
                    self.bn_d.running_mean.data = self.body[1].running_mean[ch_mask_out==0]
                    self.bn_d.running_var.data = self.body[1].running_var[ch_mask_out==0]
                    self.bn_d.weight.data = self.body[1].weight[ch_mask_out==0]
                    self.bn_d.bias.data = self.body[1].bias[ch_mask_out==0]
                    self.bn_d.eval()

                if s_num > 0:
                    self.bn_s = nn.BatchNorm1d(s_num).to(ch_mask_out.device)
                    self.bn_s.running_mean.data = self.body[1].running_mean[ch_mask_out == 1]
                    self.bn_s.running_var.data = self.body[1].running_var[ch_mask_out == 1]
                    self.bn_s.weight.data = self.body[1].weight[ch_mask_out == 1]
                    self.bn_s.bias.data = self.body[1].bias[ch_mask_out == 1]
                    self.bn_s.eval()

    def forward(self, input):
        if self.training:
            if not self.sparse:
                out = self.body(input)

                return out
            else:
                sample_xyz, rel_xyz, fea, knn_idx, pt_mask, ch_mask = input     # x: B * C * N | knn_idx: B * N * K
                out = self.body([sample_xyz, rel_xyz, fea, knn_idx])

                # mask
                if pt_mask is not None:
                    out = out * ch_mask[..., 1:] * pt_mask + out * ch_mask[..., :1]

                return out

        else:
            if not self.sparse:
                out = self.body(input)

                return out
            else:
                fea_d, fea_s = self.conv(input)

                if fea_d is not None:
                    if self.bn:
                        fea_d = self.bn_d(fea_d)
                    if self.relu:
                        fea_d = F.relu(fea_d, True)
                if fea_s is not None:
                    if self.bn:
                        fea_s = self.bn_s(fea_s)
                    if self.relu:
                        fea_s = F.relu(fea_s, True)

                return fea_d, fea_s


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_neighbors, n_layers=2, radius=1.0):
        super(BasicBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_neighbors = n_neighbors
        self.n_layers = n_layers
        self.radius = radius
        self.relu = nn.ReLU(True)
        self.tau = 1

        # body
        body = []
        for i in range(n_layers):
            if i == 0:
                body.append(BasicConv(in_channels, out_channels, n_neighbors, bn=True, relu=True, sparse=True))
            else:
                body.append(BasicConv(out_channels, out_channels, n_neighbors, bn=True, relu=True, sparse=True))
        self.body = nn.Sequential(*body)

        # mask
        self.pt_mask = nn.Sequential(
            BasicConv(in_channels, in_channels // 4, 1, bn=True, relu=True),
            BasicConv(in_channels // 4, in_channels // 4, n_neighbors, bn=True, relu=True),
            BasicConv(in_channels // 4, 2, n_neighbors, bn=False, relu=False),
        )
        self.ch_mask = nn.Parameter(torch.rand(n_layers, out_channels, 2))

        # tail
        self.tail = nn.Sequential(
            PointConv_SM(out_channels * n_layers, out_channels, 1, bias=True, sparse=True),
            nn.BatchNorm1d(out_channels)
        )

        # shortcut
        shortcut = []
        if in_channels != out_channels:
            shortcut.append(nn.Conv1d(in_channels, out_channels, 1, bias=False))
            shortcut.append(nn.BatchNorm1d(out_channels))
        self.shortcut = nn.Sequential(*shortcut)

    def _update_tau(self, tau):
        self.tau = tau

    def _generate_indices(self, pt_mask, knn_idx_d):
        idx_mask = torch.nonzero(pt_mask[0, 0, :]).view(-1)

        indices = torch.arange(float(idx_mask.size(0))).view(-1).to(pt_mask.device) + 1
        pt_mask.zero_()
        pt_mask[0, 0, idx_mask] = indices

        knn_idx_s = pt_mask[0, 0, knn_idx_d[:, idx_mask, :]]

        return idx_mask, knn_idx_s.long()

    def _prepare(self):
        ch_mask = (self.ch_mask / 0.01).softmax(-1).round()[..., 1]

        for i in range(self.n_layers):
            if i == 0:
                self.body[i]._prepare(torch.zeros(self.in_channels).to(ch_mask.device), ch_mask[i, ...])
            else:
                self.body[i]._prepare(ch_mask[i-1, ...], ch_mask[i, ...])

        self.tail[0]._prepare(ch_mask.view(-1), torch.zeros(self.out_channels).to(ch_mask.device))

    def forward(self, input):
        xyz, fea, knn_idx = input
        xyz = xyz[..., :fea.shape[2]]

        # resolution 1
        neighbor_xyz = batch_gather(xyz, knn_idx)             # B, 3, K, N
        rel_xyz = neighbor_xyz - xyz.unsqueeze(2)             # B, 3, K, N
        sample_xyz = rel_xyz / self.radius
        sample_xyz = sample_xyz.permute(0, 2, 3, 1).unsqueeze(-2)

        # mask
        if self.training:
            # generate masks
            pt_mask = self.pt_mask[0](fea)
            pt_mask = self.pt_mask[1]([sample_xyz, rel_xyz, pt_mask, knn_idx])
            pt_mask = self.pt_mask[2]([sample_xyz, rel_xyz, pt_mask, knn_idx])
            pt_mask = gumbel_softmax(pt_mask, 1, self.tau)[:, 1:, :]
            ch_mask = gumbel_softmax(self.ch_mask, -1, self.tau)

            # body
            buffer = []
            for i in range(self.n_layers):
                if i == 0:
                    buffer.append(self.body[i]([sample_xyz, rel_xyz, fea, knn_idx,
                                                pt_mask, ch_mask[i:i + 1, ...]]))
                else:
                    buffer.append(self.body[i]([sample_xyz, rel_xyz, buffer[-1], knn_idx,
                                                pt_mask, ch_mask[i:i + 1, ...]]))

            # tail
            out = self.relu(self.tail(torch.cat(buffer, 1)) + self.shortcut(fea))

            # sparsity
            flops = []
            total_flops = 0
            for i in range(self.n_layers):
                sparsity = (pt_mask * ch_mask[i:i + 1, :, 1].view(1, -1, 1) +
                            torch.ones_like(pt_mask) * ch_mask[i:i + 1, :, 0].view(1, -1, 1)).view(-1)
                if i == 0:
                    flops.append(sparsity * self.n_neighbors * (self.in_channels + 1))
                    total_flops += xyz.shape[0] * xyz.shape[-1] * self.n_neighbors * self.out_channels * (
                            self.in_channels + 1)
                else:
                    flops.append(sparsity * self.n_neighbors * (self.out_channels + 1))
                    total_flops += xyz.shape[0] * xyz.shape[-1] * self.n_neighbors * self.out_channels * (
                            self.out_channels + 1)

            return out, torch.cat(flops, 0), total_flops

        else:
            # generate masks
            pt_mask = self.pt_mask[0](fea)
            pt_mask = self.pt_mask[1]([sample_xyz, rel_xyz, pt_mask, knn_idx])
            pt_mask = self.pt_mask[2]([sample_xyz, rel_xyz, pt_mask, knn_idx])
            pt_mask = (pt_mask / self.tau).softmax(1)[:, 1:, :]
            ch_mask = (self.ch_mask / self.tau).softmax(-1)
            pt_mask_clone = pt_mask.clone()

            idx_mask, knn_idx_s = self._generate_indices(pt_mask, knn_idx)

            # body
            buffer_d = []
            buffer_s = []
            fea_d = fea
            fea_s = None
            knn_idx_d = knn_idx
            for i in range(self.n_layers):
                fea_d, fea_s = self.body[i]([sample_xyz, rel_xyz, fea_d, fea_s, idx_mask, knn_idx_d, knn_idx_s,
                                             pt_mask_clone, ch_mask[i:i + 1, ...]])
                buffer_d.append(fea_d)
                buffer_s.append(fea_s)

            # tail
            if idx_mask.shape[0] > 0:
                out = self.tail([torch.cat(buffer_d, 1), torch.cat(buffer_s, 1), idx_mask])
            else:
                out = self.tail([torch.cat(buffer_d, 1), None, idx_mask])
            out = self.relu(out + self.shortcut(fea))

            return out


class SMPointSeg(nn.Module):
    def __init__(self, d_in, num_classes,
                 channels=[32, 64, 128, 256], layers=[2, 2, 2, 2],  ratio=[4, 4, 4, 4],
                 n_neighbors=16, radius=0.05):
        super(SMPointSeg, self).__init__()
        self.n_neighbors = n_neighbors
        self.n_blocks = len(channels)
        self.ratio = ratio

        # initial 1x1 conv
        self.init = BasicConv(d_in, channels[0], 1, bn=True, relu=True)

        # encoder
        encoder = []
        for i in range(self.n_blocks):
            if i == 0:
                encoder.append(BasicBlock(channels[0], channels[i], n_neighbors, n_layers=layers[i], radius=radius*(2**i)))
            else:
                encoder.append(BasicBlock(channels[i-1], channels[i], n_neighbors, n_layers=layers[i], radius=radius*(2**i)))
        self.encoder = nn.Sequential(*encoder)

        # decoder
        decoder = []
        for i in range(self.n_blocks-1):
            in_channels = channels[self.n_blocks - 1 - i] + channels[self.n_blocks - 2 - i]
            out_channels = channels[self.n_blocks - 2 - i]
            decoder.append(BasicConv(in_channels, out_channels, 1, bn=True, relu=True))
        self.decoder = nn.Sequential(*decoder)

        # tail
        self.tail = nn.Sequential(
            BasicConv(channels[0], 32, 1, bn=True, relu=True),
            nn.Dropout(0.5),
            BasicConv(32, num_classes, 1),
        )

    def forward(self, input):
        x, knn_idx = input
        xyz = x[:, :3, :]

        if self.training:
            fea_list = []
            flops_list = []
            total_flops = 0

            # inital
            fea0 = self.init(x)

            # encoder
            for i in range(self.n_blocks):
                if i == 0:
                    fea, flops, module_flops = self.encoder[i]([xyz, fea0, knn_idx[i]])
                else:
                    fea = subsampling(fea_list[-1], knn_idx[i-1], self.ratio[i-1])
                    fea, flops, module_flops = self.encoder[i]([xyz, fea, knn_idx[i]])
                fea_list.append(fea)
                flops_list.append(flops)
                total_flops += module_flops

            # decoder
            for i in range(self.n_blocks-1):
                fea = torch.cat([upsampling(fea_list[-1], knn_idx[self.n_blocks+i]), fea_list[self.n_blocks-2-i]], 1)
                fea = self.decoder[i](fea)
                fea_list.append(fea)

            # tail
            out = self.tail(fea_list[-1])

            # sparsity
            sparsity = torch.cat(flops_list, 0).sum() / total_flops

            return out, sparsity

        else:
            fea_list = []

            # inital
            fea0 = self.init(x)

            # encoder
            for i in range(self.n_blocks):
                if i == 0:
                    fea = self.encoder[i]([xyz, fea0, knn_idx[i]])
                else:
                    fea = subsampling(fea_list[-1], knn_idx[i - 1], self.ratio[i - 1])
                    fea = self.encoder[i]([xyz, fea, knn_idx[i]])
                fea_list.append(fea)

            # decoder
            for i in range(self.n_blocks - 1):
                fea = torch.cat([upsampling(fea_list[-1], knn_idx[self.n_blocks + i]), fea_list[self.n_blocks - 2 - i]], 1)
                fea = self.decoder[i](fea)
                fea_list.append(fea)

            # tail
            out = self.tail(fea_list[-1])

            return out
