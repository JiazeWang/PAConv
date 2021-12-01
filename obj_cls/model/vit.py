import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import sys
import numpy as np
sys.path.append("../")
from lib.pointops.functions import pointops
# helpers


class Encoder(nn.Module):
    def __init__(self, encoder_channel):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.encoder_channel, 1)
        )
    def forward(self, point_groups):
        '''
            point_groups : B G N 3
            -----------------
            feature_global : B G C
        '''
        bs, g, n , _ = point_groups.shape
        point_groups = point_groups.reshape(bs * g, n, 3)
        # encoder
        feature = self.first_conv(point_groups.transpose(2,1))  # BG 256 n
        feature_global = torch.max(feature,dim=2,keepdim=True)[0]  # BG 256 1
        feature = torch.cat([feature_global.expand(-1,-1,n), feature], dim=1)# BG 512 n
        feature = self.second_conv(feature) # BG 1024 n
        feature_global = torch.max(feature, dim=2, keepdim=False)[0] # BG 1024
        return feature_global.reshape(bs, g, self.encoder_channel)

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

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
    #print("index_d.shape", index_d.shape)
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
        b, n, c = p1.size()
        p1 = p1.contiguous()
        p1_trans = p1.transpose(1, 2).contiguous() # (B, 3, N)
        p2 = pointops.gathering(p1_trans, (pointops.furthestsampling(p1, self.num_clusters)).contiguous()).transpose(1, 2).contiguous()
        p2 = sort_points(p2)
        n_x = self.grouper(xyz=p1, new_xyz=p2).reshape(b, self.num_clusters, c * self.num_neighbors)
        return n_x


class get_new_points_v2(nn.Module):

    def __init__(self,
                 num_clusters = 32,
                 num_neighbors = 32):
        super(get_new_points_v2, self).__init__()

        self.num_clusters = num_clusters
        self.num_neighbors = num_neighbors
        self.grouper = pointops.QueryAndGroup(nsample=num_neighbors, use_xyz=True, return_xyz=True)

    def forward(self, p1):
        b, n, c = p1.size()
        p1 = p1.contiguous()
        p1_trans = p1.transpose(1, 2).contiguous() # (B, 3, N)
        p2 = pointops.gathering(p1_trans, (pointops.furthestsampling(p1, self.num_clusters)).contiguous()).transpose(1, 2).contiguous()
        p2 = sort_points(p2)
        n_x = self.grouper(xyz=p1, new_xyz=p2)
        n_x = n_x - p2.transpose(1, 2).unsqueeze(-1)
        return n_x.permute(0, 2, 3, 1)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, patch_point_num, num_patches, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 256, dropout = 0., emb_dropout = 0.):
        super().__init__()
        patch_dim = channels * patch_point_num
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.emb = nn.Linear(patch_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        #self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.Encoder = Encoder(encoder_channel = dim)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        """
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        """

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        self.get_new_points = get_new_points_v2(num_clusters = num_patches, num_neighbors = patch_point_num)

    def forward(self, data, label=None, criterion=None):
        data = data.transpose(1,2)
        x = self.get_new_points(data)
        #print("x.shape",x.shape)
        x = self.Encoder(x)
        #print("x.after_encoder",x.shape)
        #x = self.emb(x)
        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        x = self.to_latent(x)
        x = self.mlp_head(x)
        if criterion is not None:
            return x, criterion(x, label)
        else:
            return x

def parameter_number(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':

    model = ViT(patch_point_num=32, num_patches=64, num_classes=40, dim=256, depth=12, heads=6, mlp_dim=512).cuda()
    print(parameter_number(model)/1000000)
    points = torch.from_numpy(np.load("target.npy")).unsqueeze(0).contiguous().cuda()#.to(device)
    points = points.transpose(1,2)
    result = model(points)
    print("final_result:", result.shape)
