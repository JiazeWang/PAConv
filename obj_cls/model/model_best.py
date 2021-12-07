import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../")
import gcn3d
from model.transformer0 import Transformer
import numpy as np


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

        self.attention0 = Transformer(128)
        self.attention1 = Transformer(128)
        self.attention2 = Transformer(256)
        self.attention3 = Transformer(512)
        self.attention4 = Transformer(512)

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

        self.conv1d_block = nn.Sequential(
            nn.Conv1d(1024, 512, 1),
            nn.ReLU(inplace= True),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(inplace= True),
            nn.Conv1d(512, class_num, 1),
        )

    def forward(self,
                vertices: "tensor (bs, vetice_num, 3)"):
        """
        Return: (bs, vertice_num, class_num)
        """
        vertices = torch.transpose(vertices, 1, 2)
        bs, vertice_num, _ = vertices.size()
        fm_0 = self.conv_0(vertices)
        fm_0 = self.conv_down0(fm_0)
        fm_0 = self.attention0(vertices, fm_0)
        fm_1 = self.conv_1(vertices, fm_0)
        fm_1 = self.conv_down1(fm_1)
        fm_1 = self.attention1(vertices, fm_1)
        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1)
        fm_2 = self.conv_2(v_pool_1, fm_pool_1)
        fm_2 = self.attention2(v_pool_1, fm_2)
        fm_3 = self.conv_3(v_pool_1, fm_2)
        fm_3 = self.attention3(v_pool_1, fm_3)
        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3)
        fm_4 = self.conv_4(v_pool_2, fm_pool_2)
        fm_4 = self.conv_down2(fm_4)
        fm_4 = self.attention4(v_pool_2, fm_4)
        f_global = fm_4.max(1)[0] #(bs, f)
        return f_global

class GCN_Fusion_surface(nn.Module):
    def __init__(self, dim_input, support_num, neighbor_num_l = 10, neighbor_num_g = 50,):
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
    def __init__(self, dim_input, support_num, neighbor_num_l = 10, neighbor_num_g = 50,):
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
    points = torch.from_numpy(np.load("target.npy")).unsqueeze(0)
    model = GCN3D(class_num= 50, support_num= 1, neighbor_num= 50)
    print(model(points).shape)

if __name__ == "__main__":
    test()
