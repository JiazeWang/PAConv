import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../")
import gcn3d
from pointnet_util import index_points, square_distance
import numpy as np


class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k = 16):
        super(TransformerBlock, self).__init__()
        self.k = k
        channels = d_points
        self.q_conv = nn.Linear(channels, channels)
        self.k_conv = nn.Linear(channels, channels)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Linear(channels, channels)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

        self.fc_delta = nn.Sequential(
            nn.Linear(3, channels),
            nn.ReLU(),
            nn.Linear(channels, channels)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(channels, channels),
            nn.ReLU(),
            nn.Linear(channels, channels)
        )

    def forward(self, xyz, x):

        dists = square_distance(xyz, xyz)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k
        knn_xyz = index_points(xyz, knn_idx)
        knn_features = index_points(x, knn_idx)
        pre = x
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)
        x_q = self.q_conv(knn_features)
        x_k = self.k_conv(knn_features)
        x_v = self.v_conv(knn_features)
        qk = torch.einsum('bmnf,bmnf->bmnf', x_q, x_k)
        #print("qk.shape", qk.shape)
        energy = pos_enc + qk
        #print("energy.shape:", energy.shape)
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        res = torch.einsum('bmnf,bmnf->bmf', attention, x_v)
        #print("res.shape", res.shape)
        x = pre + res
        return x

class GCN3D(nn.Module):
    def __init__(self, class_num, support_num, neighbor_num):
        super().__init__()
        self.neighbor_num = neighbor_num
        self.num_part = class_num

        self.conv_0 = GCN_Fusion_surface(128, support_num= support_num)
        self.conv_1 = GCN_Fusion(128, support_num= support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_2 = GCN_Fusion(128, support_num= support_num)
        self.conv_3 = GCN_Fusion(256, support_num= support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_4 = GCN_Fusion(512, support_num= support_num)

        self.attention0 = TransformerBlock(128, 128)
        self.attention1 = TransformerBlock(128, 128)
        self.attention2 = TransformerBlock(256, 256)
        self.attention3 = TransformerBlock(512, 512)
        self.attention4 = TransformerBlock(512, 512)

        self.conv_down0 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace= True),
        )
        self.conv_down1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace= True),
        )

        self.conv_down2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace= True),
        )
        dim_fuse = sum([128, 128, 256, 512, 512, 512, 16])
        self.conv1d_block = nn.Sequential(
            nn.Conv1d(dim_fuse, 512, 1),
            nn.ReLU(inplace= True),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(inplace= True),
            nn.Conv1d(512, class_num, 1),
        )

    def forward(self,
                vertices: "tensor (bs, vetice_num, 3)",
                onehot: "tensor (bs, cat_num)",
                gt=None):
        """
        Return: (bs, vertice_num, class_num)
        """
        vertices = torch.transpose(vertices, 1, 2)
        #print("vertices.shape: ", vertices.shape)
        bs, vertice_num, _ = vertices.size()
        fm_0 = self.conv_0(vertices)
        fm_0 = self.conv_down0(fm_0)
        fm_0 = self.attention0(vertices, fm_0)
        #print("fm_0.shape: ", vertices.shape)
        #fm_0.shape:  torch.Size([2, 2048, 128])
        #fm_1.shape:  torch.Size([2, 2048, 128])
        #fm_3.shape:  torch.Size([2, 512, 512])
        #fm_4.shape:  torch.Size([2, 128, 512])
        fm_1 = self.conv_1(vertices, fm_0)
        fm_1 = self.conv_down1(fm_1)
        fm_1 = self.attention1(vertices, fm_1)
        #print("fm_1.shape: ", fm_1.shape)
        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)
        #print("v_pool_1.shape: ", v_pool_1.shape)
        fm_2 = self.conv_2(v_pool_1, fm_pool_1)
        fm_2 = self.attention2(v_pool_1, fm_2)
        #print("fm_2.shape: ", fm_2.shape)
        fm_3 = self.conv_3(v_pool_1, fm_2)
        fm_3 = self.attention3(v_pool_1, fm_3)
        #print("fm_3.shape: ", fm_3.shape)
        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        fm_4 = self.conv_4(v_pool_2, fm_pool_2)
        fm_4 = self.conv_down2(fm_4)
        fm_4 = self.attention4(v_pool_2, fm_4)
        #print("fm_4.shape: ", fm_4.shape)
        f_global = fm_4.max(1)[0] #(bs, f)
        nearest_pool_1 = gcn3d.get_nearest_index(vertices, v_pool_1)
        nearest_pool_2 = gcn3d.get_nearest_index(vertices, v_pool_2)
        fm_2 = gcn3d.indexing_neighbor(fm_2, nearest_pool_1).squeeze(2)
        fm_3 = gcn3d.indexing_neighbor(fm_3, nearest_pool_1).squeeze(2)
        fm_4 = gcn3d.indexing_neighbor(fm_4, nearest_pool_2).squeeze(2)
        f_global = f_global.unsqueeze(1).repeat(1, vertice_num, 1)
        onehot = onehot.unsqueeze(1).repeat(1, vertice_num, 1) #(bs, vertice_num, cat_one_hot)
        fm_fuse = torch.cat([fm_0, fm_1, fm_2, fm_3, fm_4, f_global, onehot], dim= 2)
        #print(fm_0.shape, fm_1.shape, fm_2.shape, fm_3.shape, fm_4.shape, f_global.shape, onehot.shape )
        conv1d_input = fm_fuse.permute(0, 2, 1) #(bs, fuse_ch, vertice_num)
        conv1d_out = self.conv1d_block(conv1d_input)
        conv1d_out = F.log_softmax(conv1d_out, dim=1)
        pred = conv1d_out.permute(0, 2, 1) #(bs, vertice_num, ch)

        if gt is not None:
            return pred, F.nll_loss(pred.contiguous().view(-1, self.num_part), gt.view(-1, 1)[:, 0])
        else:
            return pred

class GCN_Fusion_surface(nn.Module):
    def __init__(self, dim_input, support_num, neighbor_num_l = 10, neighbor_num_g = 100,):
        super().__init__()
        self.neighbor_num_l = neighbor_num_l
        self.neighbor_num_g = neighbor_num_g
        self.dim = dim_input
        self.conv_l = gcn3d.Conv_surface(kernel_num = dim_input, support_num = support_num)
        self.conv_g0 = gcn3d.Conv_surface(kernel_num = dim_input, support_num = support_num)
        self.conv_g1 = gcn3d.Conv_layer(dim_input, dim_input, support_num= support_num)
        self.bn_l = nn.BatchNorm1d(dim_input, momentum=0.1)
        self.bn_g0 = nn.BatchNorm1d(dim_input, momentum=0.1)
        self.bn_g1 = nn.BatchNorm1d(dim_input, momentum=0.1)

    def forward(self, vertices):
        neighbor_index_l = gcn3d.get_neighbor_index(vertices, self.neighbor_num_l)
        neighbor_index_g = gcn3d.get_neighbor_index(vertices, self.neighbor_num_g)
        fm_l = F.relu(self.bn_l(self.conv_l(neighbor_index_l, vertices).transpose(2, 1)).transpose(2, 1), inplace= True)
        fm_g = F.relu(self.bn_g0(self.conv_g0(neighbor_index_g, vertices).transpose(2, 1)).transpose(2, 1), inplace= True)
        fm_g = F.relu(self.bn_g1(self.conv_g1(neighbor_index_g, vertices, fm_g).transpose(2, 1)).transpose(2, 1), inplace= True)
        output = torch.cat((fm_l, fm_g), 2)
        return output


class GCN_Fusion(nn.Module):
    def __init__(self, dim_input, support_num, neighbor_num_l = 10, neighbor_num_g = 100,):
        super().__init__()
        self.neighbor_num_l = neighbor_num_l
        self.neighbor_num_g = neighbor_num_g
        self.dim = dim_input
        self.conv_l = gcn3d.Conv_layer(dim_input, dim_input, support_num= support_num)
        self.conv_g0 = gcn3d.Conv_layer(dim_input, dim_input, support_num= support_num)
        self.conv_g1 = gcn3d.Conv_layer(dim_input, dim_input, support_num= support_num)
        self.bn_l = nn.BatchNorm1d(dim_input, momentum=0.1)
        self.bn_g0 = nn.BatchNorm1d(dim_input, momentum=0.1)
        self.bn_g1 = nn.BatchNorm1d(dim_input, momentum=0.1)

    def forward(self, vertices, input):
        neighbor_index_l = gcn3d.get_neighbor_index(vertices, self.neighbor_num_l)
        neighbor_index_g = gcn3d.get_neighbor_index(vertices, self.neighbor_num_g)
        fm_l = F.relu(self.bn_l(self.conv_l(neighbor_index_l, vertices, input).transpose(2, 1)).transpose(2, 1), inplace= True)
        fm_g = F.relu(self.bn_g0(self.conv_g0(neighbor_index_g, vertices, input).transpose(2, 1)).transpose(2, 1), inplace= True)
        fm_g = F.relu(self.bn_g1(self.conv_g1(neighbor_index_g, vertices, fm_g).transpose(2, 1)).transpose(2, 1), inplace= True)
        output = torch.cat((fm_l, fm_g), 2)
        return output



def test():
    from dataset_shapenet import test_model
    dataset = "../../shapenetcore_partanno_segmentation_benchmark_v0"
    model = GCN3D(class_num= 50, support_num= 1, neighbor_num= 50)
    test_model(model, dataset, cuda= "0", bs= 2, point_num= 2048)

if __name__ == "__main__":
    test()
