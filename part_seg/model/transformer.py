import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from mmdet3d.ops import (PointFPModule, Points_Sampler, QueryAndGroup,
                         gather_points)
from mmdet.models import BACKBONES


class TransformerEncoderLayerPreNorm(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):

        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout, inplace=True)
        self.dropout2 = nn.Dropout(dropout, inplace=True)

        self.activation = nn.ReLU(inplace=True)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask = None, src_key_padding_mask = None):

        src = self.norm1(src)
        src2, mask = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)

        src = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)

        return src

class LocalTransformer(nn.Module):

    def __init__(self, npoint, radius, nsample, dim_feature, dim_out, nhead=4, num_layers=2, norm_cfg=dict(type='BN2d'), ratio=1, drop=0.0, prenorm=True):
        super().__init__()

        self.npoint = npoint
        self.nsample = nsample
        self.radius = radius
        self.nc_in = dim_feature
        self.nc_out = dim_out

        self.sampler = Points_Sampler([self.npoint], ['D-FPS'])
        self.grouper = QueryAndGroup(self.radius, self.nsample, use_xyz=False, return_grouped_xyz=True, normalize_xyz=False)

        self.pe = nn.Sequential(
            ConvModule(3, self.nc_in // 2, 1, norm_cfg=norm_cfg),
            ConvModule(self.nc_in // 2, self.nc_in, 1, act_cfg=None, norm_cfg=None)
            )

        BSC_Encoder = TransformerEncoderLayerPreNorm if prenorm else nn.TransformerEncoderLayer

        self.chunk = nn.TransformerEncoder(
            BSC_Encoder(d_model=self.nc_in, dim_feedforward=2 * self.nc_in, dropout=drop, nhead=nhead) if ratio == 1 else
            LinformerEncoderLayer(src_len=nsample, ratio=ratio, d_model=self.nc_in, nhead=nhead, dropout=drop, dim_feedforward=2 * self.nc_in),
            num_layers=num_layers)

        self.fc = ConvModule(self.nc_in, self.nc_out, 1, norm_cfg=None, act_cfg=None)

    def forward(self, xyz, features):

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        fps_idx = self.sampler(xyz, features)
        new_xyz = gather_points(xyz_flipped, fps_idx).transpose(1, 2)
        group_features, group_xyz = self.grouper(xyz.contiguous(), new_xyz.contiguous(), features.contiguous()) # (B, 3, npoint, nsample) (B, C, npoint, nsample)
        B = group_xyz.shape[0]
        position_encoding = self.pe(group_xyz)
        input_features = group_features + position_encoding
        B, D, np, ns = input_features.shape

        input_features = input_features.permute(0, 2, 1, 3).reshape(-1, D, ns).permute(2, 0, 1)
        transformed_feats = self.chunk(input_features).permute(1, 2, 0).reshape(B, np, D, ns).transpose(1, 2)
        output_features = F.max_pool2d(transformed_feats, kernel_size=[1, ns])  # (B, C, npoint)
        output_features = self.fc(output_features).squeeze(-1)

        return new_xyz, output_features, fps_idx

if __name__ == "__main__":
    local_chunk = LocalTransformer(npoint = 2048, radius=0.2, nsample=16, dim_feature=64, dim_hid=64, nhead=4, num_layers=2, norm_cfg=dict(type='BN2d'), enc_ratio=1, local_drop=0.0, prenorm=True)
    print(local_chunk)
    #num_points=(2048, 1024, 512, 256),
    #radius=(0.2, 0.4, 0.8, 1.2),
    #num_samples=(16, 16, 16, 16),
    #basic_channels=64,
