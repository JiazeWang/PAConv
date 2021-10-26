import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../")
import gcn3d

class SA_Layer(nn.Module):
    def __init__(self, channels):
        super(SA_Layer, self).__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.q_conv.bias = self.k_conv.bias

        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # b, n, c
        x_q = self.q_conv(x).permute(0, 2, 1)
        # b, c, n
        x_k = self.k_conv(x)
        x_v = self.v_conv(x)
        # b, n, n
        energy = torch.bmm(x_q, x_k)

        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdim=True))
        # b, c, n
        x_r = torch.bmm(x_v, attention)
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x

class GCN3D(nn.Module):
    def __init__(self, support_num: int, neighbor_num: int):
        super().__init__()
        self.neighbor_num = neighbor_num

        self.conv_0 = gcn3d.Conv_surface(kernel_num= 32, support_num= support_num)
        self.conv_1 = gcn3d.Conv_layer(32, 64, support_num= support_num)
        self.pool_1 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_2 = gcn3d.Conv_layer(64, 128, support_num= support_num)
        self.conv_3 = gcn3d.Conv_layer(128, 256, support_num= support_num)
        self.pool_2 = gcn3d.Pool_layer(pooling_rate= 4, neighbor_num= 4)
        self.conv_4 = gcn3d.Conv_layer(256, 1024, support_num= support_num)
        self.sa1 = SA_Layer(channels=64)
        self.sa2 = SA_Layer(channels=128)
        self.sa3 = SA_Layer(channels=256)
        self.sa4 = SA_Layer(channels=1024)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace= True),
            nn.Linear(256, 40)
        )

    def forward(self,  vertices: "(bs, vertice_num, 3)"):
        bs, vertice_num, _ = vertices.size()
        print("vertices.size:", vertices.size())
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        fm_0 = self.conv_0(neighbor_index, vertices)
        fm_0 = F.relu(fm_0, inplace= True)
        fm_1 = self.conv_1(neighbor_index, vertices, fm_0)
        fm_1 = F.relu(fm_1, inplace= True)
        fm_1 = torch.transpose(self.sa1(torch.transpose(fm_1, 1, 2)), 1, 2)
        vertices, fm_1 = self.pool_1(vertices, fm_1)
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)

        fm_2 = self.conv_2(neighbor_index, vertices, fm_1)
        fm_2 = F.relu(fm_2, inplace= True)
        fm_2 = torch.transpose(self.sa2(torch.transpose(fm_2, 1, 2)), 1, 2)
        fm_3 = self.conv_3(neighbor_index, vertices, fm_2)
        fm_3 = F.relu(fm_3, inplace= True)
        fm_3 = torch.transpose(self.sa3(torch.transpose(fm_3, 1, 2)), 1, 2)
        vertices, fm_3 = self.pool_2(vertices, fm_3)
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        fm_4 = self.conv_4(neighbor_index, vertices, fm_3)
        fm_4 = torch.transpose(self.sa4(torch.transpose(fm_4, 1, 2)), 1, 2)
        feature_global = fm_4.max(1)[0]
        pred = self.classifier(feature_global)
        return pred

def test():
    import time
    sys.path.append("..")
    from util import parameter_number

    device = torch.device('cuda:0')
    points = torch.zeros(8, 1024, 3).to(device)
    model = GCN3D(support_num= 1, neighbor_num= 20).to(device)
    start = time.time()
    output = model(points)

    print("Inference time: {}".format(time.time() - start))
    print("Parameter #: {}".format(parameter_number(model)))
    print("Inputs size: {}".format(points.size()))
    print("Output size: {}".format(output.size()))

if __name__ == '__main__':
    test()
