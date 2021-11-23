import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append("../")
import gcn3d


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
        self.bn1 = nn.BatchNorm1d(64, affine=False)
        self.bn2 = nn.BatchNorm1d(128, affine=False)
        self.bn3 = nn.BatchNorm1d(256, affine=False)
        self.bn4 = nn.BatchNorm1d(1024, affine=False)
        self.d1 = nn.Conv1d(32, 64, kernel_size=1, bias=False)
        self.d2 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.d3 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.d4 = nn.Conv1d(256, 1024, kernel_size=1, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace= False),
            nn.Linear(256, 40)
        )

    def forward(self,  vertices: "(bs, vertice_num, 3)"):
        vertices = torch.transpose(vertices, 1, 2)
        bs, vertice_num, _ = vertices.size()
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)
        fm_0 = self.conv_0(neighbor_index, vertices)
        fm_0 = F.relu(fm_0, inplace= False)

        residual1 = torch.transpose(self.d1(torch.transpose(fm_0, 1, 2)), 1, 2)
        fm_1 = self.conv_1(neighbor_index, vertices, fm_0)
        fm_1 = self.bn1(fm_1.transpose(2,1)).transpose(2, 1)
        fm_1 = F.relu(fm_1, inplace= False)
        fm_1 += residual1
        #fm_1 = F.relu(fm_1, inplace= False)

        vertices, fm_1 = self.pool_1(vertices, fm_1)
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)

        residual2 = torch.transpose(self.d2(torch.transpose(fm_1, 1, 2)), 1, 2)
        fm_2 = self.conv_2(neighbor_index, vertices, fm_1)
        fm_2 = self.bn2(fm_2.transpose(2,1)).transpose(2, 1)
        fm_2 = F.relu(fm_2, inplace= False)
        fm_2 += residual2
        #fm_2 = F.relu(fm_2, inplace= False)

        residual3 = torch.transpose(self.d3(torch.transpose(fm_2, 1, 2)), 1, 2)
        fm_3 = self.conv_3(neighbor_index, vertices, fm_2)
        fm_3 = self.bn3(fm_3.transpose(2,1)).transpose(2, 1)
        fm_3 = F.relu(fm_3, inplace= False)
        fm_3 += residual3
        #fm_3 = F.relu(fm_3, inplace= False)

        vertices, fm_3 = self.pool_2(vertices, fm_3)
        neighbor_index = gcn3d.get_neighbor_index(vertices, self.neighbor_num)

        residual4 = torch.transpose(self.d4(torch.transpose(fm_3, 1, 2)), 1, 2)
        fm_4 = self.conv_4(neighbor_index, vertices, fm_3)
        fm_4 = self.bn4(fm_4.transpose(2,1)).transpose(2, 1)
        fm_4 = F.relu(fm_4, inplace= False)
        fm_4 += residual4

        feature_global = fm_4.max(1)[0]
        pred = self.classifier(feature_global)
        return pred

def parameter_number(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def test():
    import time
    sys.path.append("..")
    #from util import parameter_number

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
