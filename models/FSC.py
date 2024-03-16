#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.model_utils import *
from pointnet2_ops import pointnet2_utils


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    # fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    # new_xyz = index_points(xyz, fps_idx)
    new_xyz = xyz
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class SuperDecoder(nn.Module):
    def __init__(self, num_dense=16384, latent_dim=2 * (1024 + 256), grid_size=4):
        super(SuperDecoder, self).__init__()
        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size

        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = self.num_dense // (self.grid_size ** 2)

        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, 2048),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(2048, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(1024, 2048),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(2048, 3 * self.num_coarse)
        )
        self.sa = PointNetSetAbstraction(npoint=1024, radius=0.2, nsample=32, in_channel=6, mlp=[64, 128, 256], group_all=False)

        self.final_conv = nn.Sequential(
            nn.Conv1d(self.latent_dim + 256 + 2, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 3, 1)
        )
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(
            self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(
            self.grid_size, self.grid_size).reshape(1, -1)

        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, S)

    def forward(self, feature_global):
        B, _ = feature_global.shape

        coarse = self.mlp(feature_global).reshape(-1, self.num_coarse, 3)  # (B, num_coarse, 3), coarse point cloud
        point_feat = coarse.unsqueeze(2).expand(-1, -1, int((self.grid_size ** 2)), -1)  # (B, num_coarse, S, 3)
        point_feat = point_feat.reshape(-1, self.num_dense, 3).transpose(2, 1)  # (B, 3, num_fine)

        L = self.sa(coarse.transpose(2, 1), coarse.transpose(2, 1))[1]          # [B, 256, num_coarse]
        L = L.unsqueeze(2).expand(-1, -1, 16, -1)                               # (B, 256, 16, num_coarse)
        L = L.reshape(B, 256, self.num_dense)
        
        seed = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1)  # (B, 2, num_coarse, S)
        seed = seed.reshape(B, -1, self.num_dense)  # (B, 2, num_fine)            

        feature_global = feature_global.unsqueeze(2).expand(-1, -1, self.num_dense)  # (B, latent_dim, num_fine)
        
        # feat = torch.cat([feature_global, seed, point_feat], dim=1)  # (B, 1024+2+3, num_fine)
        feat = torch.cat([feature_global, seed, L], dim=1)  # (B, latent_dim+2+256, num_fine)

        fine = self.final_conv(feat) + point_feat  # (B, 3, num_fine), fine point cloud

        return coarse.contiguous(), fine.transpose(1, 2).contiguous()
    

class ConvLayer(nn.Module):
    def __init__(self, num_dense=16384, latent_dim=1024, grid_size=4):
        super(ConvLayer, self).__init__()
        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size

        assert self.num_dense % self.grid_size ** 2 == 0

        self.num_coarse = self.num_dense // (self.grid_size ** 2)

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(128, 256, 1),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2),
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(512, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Conv1d(1024, self.latent_dim, 1),
            nn.BatchNorm1d(self.latent_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )

    def forward(self, xyz):
        B, N, _ = xyz.shape

        feature = self.first_conv(xyz.transpose(2, 1))  # (B,  256, N)
        low_feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # (B,  256, 1)
        feature = torch.cat([low_feature_global.expand(-1, -1, N), feature], dim=1)  # (B,  512, N)
        feature = self.second_conv(feature)  # (B, 1024, N)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # (B, 1024)
        multi_level_global_feature = torch.cat([torch.squeeze(low_feature_global, dim=2), feature_global], dim=1)
        return multi_level_global_feature


class offset_attention(nn.Module):
    def __init__(self, channels):
        super(offset_attention, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x = x + xyz
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x)  # b, c, n
        x_v = self.v_conv(x)
        energy = torch.bmm(x_q, x_k)  # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        x_r = torch.bmm(x_v, attention)  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


class Multi_head_ExternalAttention(nn.Module):
    """
    # Input : x , an array with shape [B, N, C_in]
    # (batchsize, pixels, channels)
    # Parameter: M_K, a linearlayer
    # Parameter: M_V, a linearlayer
    # Parameter: heads number of heads
    # Output : out , an array with shape [B, N, C_in]
    """
    def __init__(self, channels, external_memory, point_num, heads):
        super(Multi_head_ExternalAttention, self).__init__()
        self.channels = channels
        self.external_memory = external_memory
        self.point_num = point_num
        self.heads = heads
        self.Q_linear = nn.Linear(self.channels + 3, self.channels)
        self.K_linear = nn.Linear(self.channels // self.heads, self.external_memory)
        self.softmax = nn.Softmax(dim=2)
        self.LNorm = nn.LayerNorm([self.heads, self.point_num, self.external_memory])
        self.V_Linear = nn.Linear(self.external_memory, self.channels // self.heads)
        self.O_Linear = nn.Linear(self.channels, self.channels)

    def forward(self, x, xyz):
        B, N, C = x.transpose(2, 1).shape
        x = torch.cat([x.transpose(2, 1), xyz], dim=2)
        x = self.Q_linear(x).contiguous().view(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        attention = self.LNorm(self.softmax(self.K_linear(x)))
        output = self.O_Linear(self.V_Linear(attention).permute(0, 2, 1, 3).contiguous().view(B, N, C))
        return output.transpose(2, 1)


class ConvLayer_Transformer(nn.Module):
    def __init__(self, num_dense=16384, latent_dim=1024, grid_size=4, point_scales=2048):
        super(ConvLayer_Transformer, self).__init__()
        self.num_dense = num_dense
        self.latent_dim = latent_dim
        self.grid_size = grid_size
        self.point_scales = point_scales

        assert self.num_dense % self.grid_size ** 2 == 0

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        # self.EA = Multi_head_ExternalAttention(64, 64, self.point_scales, 4)
        self.OA = offset_attention(64)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 128, 1)
        self.CA1 = offset_attention(128)
        self.CA2 = offset_attention(128)
        # self.CEA1 = Multi_head_ExternalAttention(128, 128, self.point_scales, 4)
        # self.CEA2 = Multi_head_ExternalAttention(128, 128, self.point_scales, 4)
        self.conv4 = nn.Conv1d(256, 256, 1)
        self.conv5 = nn.Conv1d(512, 512, 1)
        self.CA3 = offset_attention(512)
        self.CA4 = offset_attention(512)
        # self.CEA3 = Multi_head_ExternalAttention(512, 512, self.point_scales, 4)
        # self.CEA4 = Multi_head_ExternalAttention(512, 512, self.point_scales, 4)
        self.conv6 = nn.Conv1d(1024, 1024, 1)
        self.conv7 = nn.Conv1d(1024, self.latent_dim, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(512)
        self.bn6 = nn.BatchNorm1d(1024)
        self.bn7 = nn.BatchNorm1d(self.latent_dim)

    def forward(self, x):
        B, N, _ = x.shape
        self.xyz = x
        x = F.leaky_relu(self.bn1(self.conv1(x.transpose(2, 1))), negative_slope=0.2)
        # x = self.EA(x, self.xyz)
        x = self.OA(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        # temp00_x_128 = self.CEA1(x, self.xyz)
        # temp01_x_128 = self.CEA2(temp00_x_128, self.xyz)
        temp00_x_128 = self.CA1(x)
        temp01_x_128 = self.CA2(temp00_x_128)
        x = torch.cat((temp00_x_128, temp01_x_128), dim=1)
        x_256 = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        low_feature_global = torch.max(x_256, dim=2, keepdim=True)[0]  # (B,  256, 1)
        x = torch.cat([low_feature_global.expand(-1, -1, N), x_256], dim=1)  # (B,  512, N)
        x_512 = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.2)
        # temp00_x_256 = self.CEA3(x_512, self.xyz)
        # temp01_x_256 = self.CEA4(temp00_x_256, self.xyz)
        temp00_x_256 = self.CA3(x_512)
        temp01_x_256 = self.CA4(temp00_x_256)
        x = torch.cat((temp00_x_256, temp01_x_256), dim=1)
        x = F.leaky_relu(self.bn6(self.conv6(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn7(self.conv7(x)), negative_slope=0.2)
        feature_global = torch.max(x, dim=2, keepdim=False)[0]  # (B, 1024)
        multi_level_global_feature = torch.cat([torch.squeeze(low_feature_global, dim=2), feature_global], dim=1)
        return multi_level_global_feature


class AutoEncoder(nn.Module):
    def __init__(self, multi_global_size=2 * (1024 + 256), num_coarse=1024):
        super(AutoEncoder, self).__init__()
        self.multi_global_size = multi_global_size
        self.num_coarse = num_coarse
        self.Encoder = ConvLayer()
        # self.FC = nn.Sequential(
        #     nn.Linear(self.multi_global_size, 1024),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Linear(1024, 1024),
        #     nn.LeakyReLU(negative_slope=0.2),
        #     nn.Linear(1024, 3 * self.num_coarse)
        # )
        self.Encoder1 = ConvLayer_Transformer()
        # self.FC = nn.Sequential(
        #     nn.Linear(self.multi_global_size, self.multi_global_size, bias=True),
        #     nn.BatchNorm1d(self.multi_global_size),
        #     nn.LeakyReLU(negative_slope=0.2),
        # )
        self.fc = nn.Linear(self.multi_global_size, self.multi_global_size, bias=True)
        self.bn = nn.BatchNorm1d(self.multi_global_size)

    def forward(self, x):
        output = self.Encoder(x)
        # x = self.FC(x).reshape(-1, self.num_coarse, 3)
        output1 = self.Encoder1(x)
        # output = F.relu(self.bn(self.fc(x)))
        output = torch.cat([output, output1], dim=1)
        output = F.relu(self.bn(self.fc(output)))
        # output = self.FC(output)
        return output


class Model(nn.Module):
    def __init__(self):
        super(PreModel, self).__init__()
        self.Encoder = AutoEncoder()
        self.Decoder = SuperDecoder(num_dense=512)
        self.Decoder1 = SuperDecoder(num_dense=16384)

    def forward(self, x, gt=None, is_training=True):
        x = self.Encoder(x)
        coarse, fine = self.Decoder(x)
        coarse, fine1 = self.Decoder(x)


        coarse = coarse.transpose(1, 2).contiguous()
        fine = fine.transpose(1, 2).contiguous()
        fine1 = fine1.transpose(1, 2).contiguous()


        if is_training:
            loss3, _ = calc_cd(fine1, gt)
            gt_fine1 = pointnet2_utils.gather_operation(gt.transpose(1, 2).contiguous(), pointnet2_utils.furthest_point_sample(gt, fine.shape[1])).transpose(1, 2).contiguous()

            loss2, _ = calc_cd(fine, gt_fine1)
            gt_coarse = pointnet2_utils.gather_operation(gt_fine1.transpose(1, 2).contiguous(), pointnet2_utils.furthest_point_sample(gt_fine1, coarse.shape[1])).transpose(1, 2).contiguous()

            loss1, _ = calc_cd(coarse, gt_coarse)

            total_train_loss = loss1.mean() + loss2.mean() + loss3.mean()

            return fine, loss2, total_train_loss
        else:
            cd_p, cd_t = calc_cd(fine1, gt)
            cd_p_coarse, cd_t_coarse = calc_cd(coarse, gt)

            return {'out1': coarse, 'out2': fine1, 'cd_t_coarse': cd_t_coarse, 'cd_p_coarse': cd_p_coarse, 'cd_p': cd_p, 'cd_t': cd_t}



if __name__ == '__main__':

    sim_data = Variable(torch.rand(8, 2560))
    sim_data = sim_data.cuda()
    sim_data2 = Variable(torch.rand(8, 2048, 3))
    sim_data2 = sim_data2.cuda()
    # extract = SuperDecoder()
    # extract = extract.cuda()
    # coarse_output, fine_output = extract(sim_data)
    model = PreModel()
    model = model.cuda()
    coarse, fine = model(sim_data2)
    print('input', sim_data2.size())
    print('Output_coarse:', coarse.size())
    print('Output_fine', fine.size())