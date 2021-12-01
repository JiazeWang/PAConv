import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import open3d as o3d
sys.path.append("../")
import numpy as np
from lib.pointops.functions import pointops
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

def sort_points(input):
    b, np, c = input.size()
    inputnew = input.clone()
    inputnew[:,:,0] = inputnew[:,:,0]- input[:,:,0].min()
    inputnew[:,:,1] = inputnew[:,:,1]- input[:,:,1].min()
    inputnew[:,:,2] = inputnew[:,:,2]- input[:,:,2].min()
    distance = inputnew[:,:,0] * inputnew[:,:,0] + inputnew[:,:,1] * inputnew[:,:,1]+inputnew[:,:,2] * inputnew[:,:,2]
    sort_d, index_d = torch.sort(distance)
    result = batched_index_select(input, index_d)
    return result



class get_new_points(nn.Module):

    def __init__(self,
                 num_clusters = 32,
                 num_neighbors = 32):
        super(get_new_points, self).__init__()

        self.num_clusters = num_clusters
        self.num_neighbors = num_neighbors
        self.grouper = pointops.QueryAndGroup(nsample=num_neighbors, use_xyz=True, return_xyz=True)

    def forward(self, p1):
        # points, p: (B, N, 3)
        # in_features, x: (B, C_in, N)
        #result = pointops.furthestsampling(px, self.num_clusters)
        #result = pointops.gathering(px, pointops.furthestsampling(px, self.num_clusters)).transpose(1, 2).contiguous()
        b, n, c = p1.size()
        p1_trans = p1.transpose(1, 2).contiguous() # (B, 3, N)
        p2 = pointops.gathering(p1_trans, pointops.furthestsampling(p1, self.num_clusters)).transpose(1, 2).contiguous()
        p2 = sort_points(p2)
        print(p2)
        print("p2:", p2.shape)
        n_x = self.grouper(xyz=p1, new_xyz=p2).transpose(1,2).reshape(b, self.num_clusters, c * self.num_neighbors)
        return n_x

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
        self.get_new_points = get_new_points()

    def forward(self,  vertices: "(bs, vertice_num, 3)"):
        #vertices = torch.transpose(vertices, 1, 2)
        #print(vertices.shape)
        input = self.get_new_points(vertices)
        #print(input.shape)
        return input

def test():
    import time
    sys.path.append("..")
    #from util import parameter_number

    #device = torch.device('cuda:0'
    #points = torch.randn(8, 1024, 3).cuda()#.to(device)
    points = torch.from_numpy(np.load("target.npy")).unsqueeze(0).cuda()
    print(points.shape)
    model = PT().cuda()#.to(device)
    start = time.time()
    output = model(points)
    print("output", output.shape)
    output = output.reshape(32, 3, 32).transpose(1,2).reshape(1024, 3)
    print(output.shape)
    #print("Inference time: {}".format(time.time() - start))
    #print("Parameter #: {}".format(parameter_number(model)))
    #print("Inputs size: {}".format(points.size()))
    #print("Output size: {}".format(output.size()))
    """

    visdata = output.cpu().detach().numpy()
    print(visdata[0])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(visdata)
    o3d.visualization.draw_geometries([pcd])
    """



if __name__ == '__main__':
    test()
