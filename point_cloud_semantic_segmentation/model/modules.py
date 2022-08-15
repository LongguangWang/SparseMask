import torch.nn.functional as F
import torch.nn as nn
import torch
import time
import math
import torch.nn.init as init


def gumbel_softmax(x, dim, tau):
    gumbels = torch.rand_like(x)
    while bool((gumbels == 0).sum() > 0):
        gumbels = torch.rand_like(x)
    gumbels = -(-gumbels.log()).log()
    gumbels = (x + gumbels) / tau
    x = gumbels.softmax(dim)

    return x


class PointConv_SM(nn.Module):
    def __init__(self, in_channels, out_channels, n_neighbor, kernel_size=5, bias=True, sparse=True):
        super(PointConv_SM, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_neighbor = n_neighbor
        self.use_bias = bias
        self.sparse = sparse

        if n_neighbor > 1:
            self.conv_1x1 = nn.Conv1d(in_channels+3, out_channels, 1, bias=bias)
            self.conv_dw = nn.Parameter(torch.randn(1, out_channels, kernel_size, kernel_size, kernel_size))

            # initialization
            fan = n_neighbor
            gain = init.calculate_gain('relu')
            std = gain / math.sqrt(fan)
            bound = math.sqrt(3.0) * std
            self.conv_dw.data.uniform_(-bound, bound)
        else:
            self.conv_1x1 = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

    def _prepare(self, ch_mask_in, ch_mask_out):    # 1 for sparse, 0 for dense
        if self.n_neighbor > 1:
            ch_mask_in = F.pad(ch_mask_in, [0, 3])
        self.d_in_num = int((ch_mask_in==0).sum())
        self.s_in_num = int((ch_mask_in==1).sum())
        self.d_out_num = int((ch_mask_out==0).sum())
        self.s_out_num = int((ch_mask_out==1).sum())

        if self.d_in_num > 0 and self.d_out_num > 0:
            self.conv_1x1_d2d = nn.Conv1d(self.d_in_num, self.d_out_num, 1, bias=self.use_bias).to(ch_mask_out.device)
            self.conv_1x1_d2d.weight.data = self.conv_1x1.weight[ch_mask_out==0, ...][:, ch_mask_in==0, ...]
            if self.use_bias:
                self.conv_1x1_d2d.bias.data = self.conv_1x1.bias[ch_mask_out==0]
        if self.d_in_num > 0 and self.s_out_num > 0:
            self.conv_1x1_d2s = nn.Conv1d(self.d_in_num, self.s_out_num, 1, bias=self.use_bias).to(ch_mask_out.device)
            self.conv_1x1_d2s.weight.data = self.conv_1x1.weight[ch_mask_out==1, ...][:, ch_mask_in==0, ...]
            if self.use_bias:
                self.conv_1x1_d2s.bias.data = self.conv_1x1.bias[ch_mask_out==1]
        if self.s_in_num > 0:
            self.conv_1x1_s2ds = nn.Conv1d(self.s_in_num, self.d_out_num + self.s_out_num, 1, bias=self.use_bias).to(ch_mask_out.device)
            self.conv_1x1_s2ds.weight.data = torch.cat([
                    self.conv_1x1.weight[ch_mask_out==0, ...][:, ch_mask_in==1, ...],
                    self.conv_1x1.weight[ch_mask_out==1, ...][:, ch_mask_in==1, ...]
                ], 0)
            if self.use_bias:
                self.conv_1x1_s2ds.bias.data = torch.cat([
                    self.conv_1x1.bias[ch_mask_out==0],
                    self.conv_1x1.bias[ch_mask_out==1]
                ])

        if self.n_neighbor > 1:
            self.conv_dw_new = nn.Parameter(torch.zeros_like(self.conv_dw))
            self.conv_dw_new.data = torch.cat([
                self.conv_dw[:, ch_mask_out==0, ...],
                self.conv_dw[:, ch_mask_out==1, ...]
            ], 1)

    def _sparse_conv(self, fea_dense, fea_sparse, rel_xyz, idx_mask, knn_idx_d, knn_idx_s):
        # dense branch
        if self.d_in_num > 0:
            if self.n_neighbor > 1:
                neighbor_fea_d2d = batch_gather(fea_dense, knn_idx_d, 'residual')
                neighbor_fea_d2d = torch.cat([neighbor_fea_d2d, rel_xyz], 1)            # b, c, k, n
            else:
                neighbor_fea_d2d = fea_dense.unsqueeze(-2)

            if self.d_out_num > 0:
                b, c, k, n = neighbor_fea_d2d.shape
                fea_d2d = self.conv_1x1_d2d(neighbor_fea_d2d.view(b, c, -1)).view(b, -1, k, n)

            if self.s_out_num > 0 and idx_mask.shape[0] > 0:
                neighbor_fea_d2s = neighbor_fea_d2d[:, :, :, idx_mask]                  # b, c, k, n_s
                b, c, k, n = neighbor_fea_d2s.shape
                fea_d2s = self.conv_1x1_d2s(neighbor_fea_d2s.view(b, c, -1)).view(b, -1, k, n)

        # sparse branch
        if self.s_in_num > 0 and idx_mask.shape[0] > 0:
            if self.n_neighbor > 1:
                neighbor_fea_s = batch_gather(F.pad(fea_sparse, [1, 0]), knn_idx_s, 'residual')
            else:
                neighbor_fea_s = fea_sparse.unsqueeze(-2)

            b, c, k, n = neighbor_fea_s.shape
            fea_s2ds = self.conv_1x1_s2ds(neighbor_fea_s.view(b, c, -1)).view(b, -1, k, n)

        # fusion
        if self.d_out_num > 0:
            if self.d_in_num > 0:
                if self.s_in_num > 0 and idx_mask.shape[0] > 0:
                    fea_d2d[:, :, :, idx_mask] += fea_s2ds[:, :self.d_out_num, :, :]
                    fea_d = fea_d2d
                else:
                    fea_d = fea_d2d
            else:
                fea_d = torch.zeros_like(self.spa_mask).repeat([1, self.d_out_num, 1, 1])
                fea_d[:, :, :, idx_mask] = fea_s2ds[:, :self.d_out_num, :, :]
        else:
            fea_d = None

        if self.s_out_num > 0 and idx_mask.shape[0] > 0:
            if self.d_in_num > 0:
                if self.s_in_num > 0:
                    fea_s = fea_d2s + fea_s2ds[:, -self.s_out_num:, :, :]
                else:
                    fea_s = fea_d2s
            else:
                fea_s = fea_s2ds[:, -self.s_out_num:, :, :]
        else:
            fea_s = None

        return fea_d, fea_s

    def forward(self, input):
        if self.training or not self.sparse:
            if self.n_neighbor > 1:
                sample_xyz, rel_xyz, fea, knn_idx = input  # x: B * C * N | knn_idx: B * N * K
                b, n, k = knn_idx.shape

                # 1x1 conv
                neighbor_fea = batch_gather(fea, knn_idx, 'residual')
                neighbor_fea = self.conv_1x1(torch.cat([neighbor_fea, rel_xyz], 1).view(b, -1, k * n)).view(b, -1, k, n)

                # aggregation
                coord_xyz = (sample_xyz.clamp(-0.99999, 0.99999) * 5 / 2).long() + 2
                kernel = self.conv_dw[:, :, coord_xyz[..., 2:3], coord_xyz[..., 1:2], coord_xyz[..., 0:1]]
                kernel = kernel.view(b, self.out_channels, -1, n)  # B * C_out * K * N

                out = (kernel * neighbor_fea).sum(2)

                return out

            else:
                out = self.conv_1x1(input)

                return out

        else:
            if self.n_neighbor > 1:
                sample_xyz, rel_xyz, fea_d, fea_s, idx_mask, knn_idx_d, knn_idx_s = input
                b, n, k = knn_idx_d.shape

                # 1x1 conv
                fea_d1, fea_s1 = self._sparse_conv(fea_d, fea_s, rel_xyz, idx_mask, knn_idx_d, knn_idx_s)

                # aggregation
                coord_xyz = (sample_xyz.clamp(-0.99999, 0.99999) * 5 / 2).round().long() + 2
                kernel = self.conv_dw_new[:, :, coord_xyz[..., 2:3], coord_xyz[..., 1:2], coord_xyz[..., 0:1]]

                kernel = kernel.view(b, self.out_channels, -1, n)  # B * C_out * K * N
                kernel_d = kernel[:, :self.d_out_num, ...]
                kernel_s = kernel[:, self.d_out_num:, ...][..., idx_mask]

                fea_d2 = (kernel_d * fea_d1).sum(2) if fea_d1 is not None else fea_d1
                fea_s2 = (kernel_s * fea_s1).sum(2) if fea_s1 is not None else fea_s1

                return fea_d2, fea_s2

            else:
                fea_d, fea_s, idx_mask = input

                fea_d, fea_s = self._sparse_conv(fea_d, fea_s, None, idx_mask, None, None)

                return fea_d.squeeze(2)


def batch_gather(x, knn_idx, mode='plain'):
    b, n, k = knn_idx.shape

    if mode == 'plain':
        idx = torch.arange(b).to(x.device).view(-1, 1, 1).expand(-1, n, k)
        out = x[idx, :, knn_idx].permute(0, 3, 2, 1)

    if mode == 'residual':
        idx = torch.arange(b).to(x.device).view(-1, 1, 1).expand(-1, n, k-1)
        center = x[..., :n].unsqueeze(2)
        out = torch.cat([center, x[idx, :, knn_idx[..., 1:]].permute(0, 3, 2, 1)-center], 2)

    return out