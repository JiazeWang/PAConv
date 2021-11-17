from copy import deepcopy
import argparse
import config
import torch.nn as nn
import torch
import torch.nn.functional as F
from modules import (
    PointTransformerBlock,
    TransitionDown,
    TransitionUp
)
def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='s3dis_pointtransformer_repro.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_pointtransformer_repro.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

class PointTransformerSeg(nn.Module):

    def __init__(self, class_num, args=None):
        super(PointTransformerSeg, self).__init__()
        c=3
        k=class_num
        self.nsamples = [8, 16, 16, 16, 16]
        self.strides = [None, 4, 4, 4, 4]
        self.planes = [32, 64, 128, 256, 512]
        self.blocks = [2, 3, 4, 6, 3]
        self.num_part = class_num
        # encoder
        self.in_mlp = nn.Sequential(
            nn.Conv1d(c, self.planes[0], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.planes[0]),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.planes[0], self.planes[0], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.planes[0]),
            nn.ReLU(inplace=True)
        )
        self.enc_layer1 = self._make_layer(self.planes[0], self.blocks[0], nsample=self.nsamples[0])
        self.down1to2 = TransitionDown(self.planes[0], self.planes[1], stride=self.strides[1], num_neighbors=self.nsamples[1])
        self.enc_layer2 = self._make_layer(self.planes[1], self.blocks[1], nsample=self.nsamples[1])
        self.down2to3 = TransitionDown(self.planes[1], self.planes[2], stride=self.strides[2], num_neighbors=self.nsamples[2])
        self.enc_layer3 = self._make_layer(self.planes[2], self.blocks[2], nsample=self.nsamples[2])
        self.down3to4 = TransitionDown(self.planes[2], self.planes[3], stride=self.strides[3], num_neighbors=self.nsamples[3])
        self.enc_layer4 = self._make_layer(self.planes[3], self.blocks[3], nsample=self.nsamples[3])
        self.down4to5 = TransitionDown(self.planes[3], self.planes[4], stride=self.strides[4], num_neighbors=self.nsamples[4])
        self.enc_layer5 = self._make_layer(self.planes[4], self.blocks[4], nsample=self.nsamples[4])

        # decoder
        self.dec_mlp = nn.Sequential(
            nn.Conv1d(self.planes[4], self.planes[4], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.planes[4]),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.planes[4], self.planes[4], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.planes[4]),
            nn.ReLU(inplace=True)
        )
        self.dec_layer5 = self._make_layer(self.planes[4], 2, nsample=self.nsamples[4])
        self.up5to4 = TransitionUp(self.planes[4], self.planes[3], self.planes[3])
        self.dec_layer4 = self._make_layer(self.planes[3], 2, nsample=self.nsamples[3])
        self.up4to3 = TransitionUp(self.planes[3], self.planes[2], self.planes[2])
        self.dec_layer3 = self._make_layer(self.planes[2], 2, nsample=self.nsamples[2])
        self.up3to2 = TransitionUp(self.planes[2], self.planes[1], self.planes[1])
        self.dec_layer2 = self._make_layer(self.planes[1], 2, nsample=self.nsamples[1])
        self.up2to1 = TransitionUp(self.planes[1], self.planes[0], self.planes[0])
        self.dec_layer1 = self._make_layer(self.planes[0], 2, nsample=self.nsamples[0])
        self.out_mlp = nn.Sequential(
            nn.Conv1d(self.planes[0]+16, self.planes[0], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.planes[0]),
            nn.ReLU(inplace=True),
            nn.Conv1d(self.planes[0], k, kernel_size=1)
        )

    def _make_layer(self, planes, blocks, nsample):
        layers = []
        for _ in range(blocks):
            layers.append(PointTransformerBlock(planes, num_neighbors=nsample))
        return nn.Sequential(*layers)

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        #features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else deepcopy(xyz)
        features =xyz
        return xyz, features

    def forward(self, pc, onehot, gt=None):
        pc = torch.transpose(pc, 1, 2)
        p1, x1 = self._break_up_pc(pc)
        x1 = x1.transpose(1, 2).contiguous()
        #print(p1.shape, x1.shape)
        # encoder
        x1 = self.in_mlp(x1)
        p1x1 = self.enc_layer1([p1, x1])
        p2x2 = self.down1to2(p1x1)
        p2x2 = self.enc_layer2(p2x2)
        p3x3 = self.down2to3(p2x2)
        p3x3 = self.enc_layer3(p3x3)
        p4x4 = self.down3to4(p3x3)
        p4x4 = self.enc_layer4(p4x4)
        p5x5 = self.down4to5(p4x4)
        p5, x5 = self.enc_layer5(p5x5)

        # decoder
        y = self.dec_mlp(x5)
        p5y = self.dec_layer5([p5, y])
        p4y = self.up5to4(p5y, p4x4)
        p4y = self.dec_layer4(p4y)
        p3y = self.up4to3(p4y, p3x3)
        p3y = self.dec_layer3(p3y)
        p2y = self.up3to2(p3y, p2x2)
        p2y = self.dec_layer2(p2y)
        p1y = self.up2to1(p2y, p1x1)
        p1, y = self.dec_layer1(p1y)
        vertice_num = pc.shape[1]
        onehot = onehot.unsqueeze(1).repeat(1, vertice_num, 1) #(bs, vertice_num, cat_one_hot)
        y = torch.transpose(y, 1, 2)
        #print(y.shape, onehot.shape)
        fm_fuse = torch.cat([y, onehot], dim= 2)
        fm_fuse = fm_fuse.permute(0, 2, 1)
        #print(fm_fuse.shape, fm_fuse.shape)
        y = self.out_mlp(fm_fuse)
        y = F.log_softmax(y, dim=1)
        y = y.permute(0, 2, 1)
        #print("y:",y.shape)
        #print("gt:",gt.shape)
        if gt is not None:
            return y, F.nll_loss(y.contiguous().view(-1, self.num_part), gt.view(-1, 1)[:, 0])
        else:
            return y


if __name__ == "__main__":
    from time import time

    assert torch.cuda.is_available()
    args = get_parser()
    B, C_in, C_out, N, K = 4, 3, 20, 1024, 16

    #model = PointTransformerSeg(args).cuda()
    model = PointTransformerSeg(class_num=50).cuda()
    pc = torch.randn(B, N, 3 + C_in).cuda()
    onehot = torch.zeros((B, 16, 1024)).cuda()

    s = time()
    y = model(pc, onehot)
    d = time() - s

    print("Elapsed time (sec):", d)
    print(y.shape)
