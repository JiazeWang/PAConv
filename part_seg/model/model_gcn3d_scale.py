import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../")
import gcn3d

class GCN3D(nn.Module):
    def __init__(self, class_num, support_num, neighbor_num):
        super().__init__()
        self.neighbor_num = neighbor_num
        self.num_part = class_num
        self.neighbor_num_local = 10
        self.conv_0 = gcn3d.Conv_surface(kernel_num= 128, support_num= support_num)
        self.conv_1 = gcn3d.Conv_layer(128, 128, support_num= support_num)
        self.conv_down0 = nn.Linear(256, 128)
        self.conv_0_l = gcn3d.Conv_surface(kernel_num= 128, support_num= support_num)
        self.conv_1_l = gcn3d.Conv_layer(128, 128, support_num= support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_2 = gcn3d.Conv_layer(128, 256, support_num= support_num)
        self.conv_3 = gcn3d.Conv_layer(256, 256, support_num= support_num)
        self.conv_down1 = nn.Linear(512, 256)
        self.conv_2_l = gcn3d.Conv_layer(128, 256, support_num= support_num)
        self.conv_3_l = gcn3d.Conv_layer(256, 256, support_num= support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_4 = gcn3d.Conv_layer(256, 512, support_num= support_num)
        self.conv_4_l = gcn3d.Conv_layer(256, 512, support_num= support_num)
        self.conv_down2 = nn.Linear(1024, 512)

        dim_fuse = sum([128, 256, 512, 512, 16])
        self.conv1d_block = nn.Sequential(
            nn.Conv1d(dim_fuse, 512, 1),
            nn.ReLU(inplace= True),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(inplace= True),
            nn.Conv1d(512, class_num, 1),
        )

        self.bn0 = nn.BatchNorm1d(128, momentum=0.1)
        self.bn1 = nn.BatchNorm1d(128, momentum=0.1)
        self.bn0_l = nn.BatchNorm1d(128, momentum=0.1)
        self.bn1_l = nn.BatchNorm1d(128, momentum=0.1)
        self.bn2 = nn.BatchNorm1d(256, momentum=0.1)
        self.bn3 = nn.BatchNorm1d(256, momentum=0.1)
        self.bn2_l = nn.BatchNorm1d(256, momentum=0.1)
        self.bn3_l = nn.BatchNorm1d(256, momentum=0.1)
    def forward(self,
                vertices: "tensor (bs, vetice_num, 3)",
                onehot: "tensor (bs, cat_num)",
                gt=None):
        """
        Return: (bs, vertice_num, class_num)
        """
        vertices = torch.transpose(vertices, 1, 2)

        bs, vertice_num, _ = vertices.size()
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        fm_0 = F.relu(self.bn0(self.conv_0(neighbor_index, vertices).transpose(2, 1)).transpose(2, 1), inplace= True)
        fm_1 = F.relu(self.bn1(self.conv_1(neighbor_index, vertices, fm_0).transpose(2, 1)).transpose(2, 1), inplace= True)
        #print("fm_1.shape", fm_1.shape) fm_1.shape torch.Size([32, 2048, 128])
        neighbor_index_l = gcn3d.get_neighbor_index(vertices, self.neighbor_num_local)
        fm_0_l = F.relu(self.bn0_l(self.conv_0_l(neighbor_index_l, vertices).transpose(2, 1)).transpose(2, 1), inplace= True)
        fm_1_l = F.relu(self.bn1_l(self.conv_1_l(neighbor_index_l, vertices, fm_0_l).transpose(2, 1)).transpose(2, 1), inplace= True)
        fm_1_t = torch.cat((fm_1, fm_1_l), dim = 2)
        fm_1_t = self.conv_down0(fm_1_t)
        #print("fm_1_t.shape:", fm_1_t.shape)

        v_pool_1, fm_pool_1 = self.pool_1(vertices, fm_1_t)
        #print("fm_pool_1.shape", fm_pool_1.shape) fm_pool_1.shape torch.Size([32, 512, 128])

        neighbor_index = gcn3d.get_neighbor_index(v_pool_1, self.neighbor_num)
        fm_2 = F.relu(self.bn2(self.conv_2(neighbor_index, v_pool_1, fm_pool_1).transpose(2, 1)).transpose(2, 1), inplace= True)
        fm_3 = F.relu(self.bn3(self.conv_3(neighbor_index, v_pool_1, fm_2).transpose(2, 1)).transpose(2, 1), inplace= True)

        neighbor_index_l = gcn3d.get_neighbor_index(v_pool_1, self.neighbor_num_local)
        fm_2_l = F.relu(self.bn2_l(self.conv_2_l(neighbor_index_l, v_pool_1, fm_pool_1).transpose(2, 1)).transpose(2, 1), inplace= True)
        fm_3_l = F.relu(self.bn3_l(self.conv_3_l(neighbor_index_l, v_pool_1, fm_2_l).transpose(2, 1)).transpose(2, 1), inplace= True)
        #print("fm_3.shape", fm_3.shape) fm_3.shape torch.Size([32, 512, 256])
        fm_3_t = torch.cat((fm_3, fm_3_l), dim = 2)
        fm_3_t = self.conv_down1(fm_3_t)
        #print("fm_3_t.shape:", fm_3_t.shape)

        v_pool_2, fm_pool_2 = self.pool_2(v_pool_1, fm_3_t)
        #print("fm_pool_2.shape", fm_pool_2.shape) fm_pool_2.shape torch.Size([32, 128, 256])
        neighbor_index = gcn3d.get_neighbor_index(v_pool_2, self.neighbor_num)
        fm_4 = self.conv_4(neighbor_index, v_pool_2, fm_pool_2)

        neighbor_index_l = gcn3d.get_neighbor_index(v_pool_2, self.neighbor_num_local)
        fm_4_l = self.conv_4_l(neighbor_index_l, v_pool_2, fm_pool_2)

        fm_4_t = torch.cat((fm_4, fm_4_l), dim = 2)
        fm_4_t = self.conv_down2(fm_4_t)
        #print("fm_4_t.shape:", fm_4_t.shape)
        f_global = fm_4_t.max(1)[0] #(bs, f)

        nearest_pool_1 = gcn3d.get_nearest_index(vertices, v_pool_1)
        nearest_pool_2 = gcn3d.get_nearest_index(vertices, v_pool_2)
        #fm_2 = gcn3d.indexing_neighbor(fm_2, nearest_pool_1).squeeze(2)
        fm_3_f = gcn3d.indexing_neighbor(fm_3_t, nearest_pool_1).squeeze(2)
        fm_4_f = gcn3d.indexing_neighbor(fm_4_t, nearest_pool_2).squeeze(2)
        f_global = f_global.unsqueeze(1).repeat(1, vertice_num, 1)
        onehot = onehot.unsqueeze(1).repeat(1, vertice_num, 1) #(bs, vertice_num, cat_one_hot)
        fm_fuse = torch.cat([fm_1_t, fm_3_f, fm_4_f, f_global, onehot], dim= 2)

        conv1d_input = fm_fuse.permute(0, 2, 1) #(bs, fuse_ch, vertice_num)
        conv1d_out = self.conv1d_block(conv1d_input)
        conv1d_out = F.log_softmax(conv1d_out, dim=1)
        pred = conv1d_out.permute(0, 2, 1) #(bs, vertice_num, ch)

        if gt is not None:
            return pred, F.nll_loss(pred.contiguous().view(-1, self.num_part), gt.view(-1, 1)[:, 0])
        else:
            return pred

def test():
    from dataset_shapenet import test_model
    dataset = "../../shapenetcore_partanno_segmentation_benchmark_v0"
    model = GCN3D(class_num= 50, support_num= 1, neighbor_num= 50)
    test_model(model, dataset, cuda= "0", bs= 2, point_num= 2048)

if __name__ == "__main__":
    test()
