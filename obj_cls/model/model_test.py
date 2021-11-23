import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import open3d as o3d
sys.path.append("../")
import numpy as np

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

def cluster_points_xyz(input, num_cluster=32, num_cluster_points=32):
    # input: (batch_size, num_point, 3)
    value = torch.mean(input, dim = 2)
    index = torch.argsort(value)
    result = batched_index_select(input, index)
    b, np, c = input.size()
    #print(input.size())
    result = result.reshape(b, num_cluster, num_cluster_points, c)
    return result

def cluster_points(input, cluster_num = 4):
    b, np, c = input.size()
    lable = torch.zeros(b, np)
    divide_x, index_x = torch.sort(input[:,:,0])
    srt_x, rank_x = torch.sort(index_x)
    divide_y, index_y = torch.sort(input[:,:,1])
    srt_y, rank_y = torch.sort(index_y)
    divide_z, index_z = torch.sort(input[:,:,2])
    srt_z, rank_z = torch.sort(index_z)
    index_x = torch.floor(rank_x/(np/cluster_num))
    index_y = torch.floor(rank_y/(np/cluster_num))
    index_z = torch.floor(rank_z/(np/cluster_num))
    index_xyz = index_x*1+index_y*cluster_num+index_z*cluster_num*cluster_num
    selet_index, index_index = torch.sort(index_xyz)
    result = batched_index_select(input, index_index)
    box_points = int(cluster_num*cluster_num*cluster_num)
    box_num = int(np / box_points)
    result = result.reshape(b, box_num, box_points * c)
    return result


class PT(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace= True),
            nn.Linear(256, 40)
        )

    def forward(self,  vertices: "(bs, vertice_num, 3)"):
        #vertices = torch.transpose(vertices, 1, 2)
        #print(vertices.shape)
        input = cluster_points(vertices)
        #print(input.shape)
        return input

def test():
    import time
    sys.path.append("..")
    #from util import parameter_number

    #device = torch.device('cuda:0'
    #points = torch.randn(8, 1024, 3)#.to(device)
    points = torch.from_numpy(np.load("target.npy")).unsqueeze(0)
    print(points.shape)
    model = PT()#.to(device)
    start = time.time()
    output = model(points)

    #print(output.shape)
    #print("Inference time: {}".format(time.time() - start))
    #print("Parameter #: {}".format(parameter_number(model)))
    print("Inputs size: {}".format(points.size()))
    print("Output size: {}".format(output.size()))
    """
    visdata = output[0][-1]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(visdata)
    o3d.visualization.draw_geometries([pcd])
    """

if __name__ == '__main__':
    test()
