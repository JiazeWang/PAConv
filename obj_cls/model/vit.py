import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers


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
    def __init__(self, *, point_size, cluster_num, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        patch_point_num = point_size // (cluster_num * cluster_num)
        num_patches = cluster_num * cluster_num
        patch_dim = channels * patch_point_num
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.emb = nn.Linear(patch_dim, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, data, label=None, criterion=None):
        data = data.transpose(1,2)
        x = cluster_points(data)
        x = self.emb(x)
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
    model = ViT(point_size=1024, cluster_num=4, num_classes=10, dim=512, depth=4, heads=16, mlp_dim=1024, emb_dropout = 0.1)
    print(parameter_number(model)/1000000)
    points = torch.randn(8, 1024, 3)#.to(device)
    result = model(points)
    print("final_result:", result.shape)
