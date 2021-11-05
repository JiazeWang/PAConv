"""
@Author: Zhi-Hao Lin
@Contact: r08942062@ntu.edu.tw
@Time: 2020/03/06
@Document: Basic operation/blocks of 3D-GCN
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

def get_neighbor_index(vertices: "(bs, vertice_num, 3)",  neighbor_num: int):
    """
    Return: (bs, vertice_num, neighbor_num)
    """
    bs, v, _ = vertices.size()
    device = vertices.device
    inner = torch.bmm(vertices, vertices.transpose(1, 2)) #(bs, v, v)
    quadratic = torch.sum(vertices**2, dim= 2) #(bs, v)
    distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)
    neighbor_index = torch.topk(distance, k= neighbor_num + 1, dim= -1, largest= False)[1]
    neighbor_index = neighbor_index[:, :, 1:]
    return neighbor_index

def get_nearest_index(target: "(bs, v1, 3)", source: "(bs, v2, 3)"):
    """
    Return: (bs, v1, 1)
    """
    inner = torch.bmm(target, source.transpose(1, 2)) #(bs, v1, v2)
    s_norm_2 = torch.sum(source ** 2, dim= 2) #(bs, v2)
    t_norm_2 = torch.sum(target ** 2, dim= 2) #(bs, v1)
    d_norm_2 = s_norm_2.unsqueeze(1) + t_norm_2.unsqueeze(2) - 2 * inner
    nearest_index = torch.topk(d_norm_2, k= 1, dim= -1, largest= False)[1]
    return nearest_index

def indexing_neighbor(tensor: "(bs, vertice_num, dim)", index: "(bs, vertice_num, neighbor_num)" ):
    """
    Return: (bs, vertice_num, neighbor_num, dim)
    """
    bs, v, n = index.size()
    id_0 = torch.arange(bs).view(-1, 1, 1)
    tensor_indexed = tensor[id_0, index]
    return tensor_indexed

def get_neighbor_direction_norm(vertices: "(bs, vertice_num, 3)", neighbor_index: "(bs, vertice_num, neighbor_num)"):
    """
    Return: (bs, vertice_num, neighobr_num, 3)
    """
    neighbors = indexing_neighbor(vertices, neighbor_index) # (bs, v, n, 3)
    neighbor_direction = neighbors - vertices.unsqueeze(2)
    neighbor_direction_norm = F.normalize(neighbor_direction, dim= -1)
    return neighbor_direction_norm

class Conv_surface(nn.Module):
    """Extract structure feafure from surface, independent from vertice coordinates"""
    def __init__(self, kernel_num, support_num):
        super().__init__()
        self.kernel_num = kernel_num
        self.support_num = support_num

        self.relu = nn.ReLU(inplace= True)
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * kernel_num))
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.support_num * self.kernel_num)
        self.directions.data.uniform_(-stdv, stdv)

    def forward(self,
                neighbor_index: "(bs, vertice_num, neighbor_num)",
                vertices: "(bs, vertice_num, 3)"):
        """
        Return vertices with local feature: (bs, vertice_num, kernel_num)
        """
        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)

        support_direction_norm = F.normalize(self.directions, dim= 0) #(3, s * k)
        theta = neighbor_direction_norm @ support_direction_norm # (bs, vertice_num, neighbor_num, s*k)

        theta = self.relu(theta)
        theta = theta.contiguous().view(bs, vertice_num, neighbor_num, self.support_num, self.kernel_num)
        theta = torch.max(theta, dim= 2)[0] # (bs, vertice_num, support_num, kernel_num)
        feature = torch.sum(theta, dim= 2) # (bs, vertice_num, kernel_num)
        return feature

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# simple self attention and simple RPE in PointFormer
# position_embedding (b,n,n)
# query / key / value (b,h,n,d)
def attention(query, key, value, mask=None, dropout=None, position_embedding=None):
    d_k = query.size(-1)

    # scores (b,h,n,n)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    if position_embedding is not None:
        position_embedding = position_embedding.unsqueeze(1)
        scores = scores + position_embedding

    p_attn = F.softmax(scores, dim=-1)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, fn_attention=attention, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k

        self.d_model = d_model
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.fn_attention = fn_attention
        self.attn = None
        self.dropout = None
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.d_model * self.d_model)
        for linear in self.linears:
            linear.weight.data.uniform_(-stdv, stdv)

    def forward(self, query, key, value):
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        #x, self.attn = attention(query, key, value, mask=mask,
        #                         dropout=self.dropout, position_embedding=position_embedding)
        x, self.attn = self.fn_attention(query, key, value, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class Attention_Conv_surface(nn.Module):
    def __init__(self, kernel_num, support_num):
        super().__init__()
        self.kernel_num = kernel_num
        self.support_num = support_num

        self.relu = nn.ReLU(inplace= True)
        self.query_directions = nn.Parameter(torch.FloatTensor(3, support_num * kernel_num))
        self.value_directions = nn.Parameter(torch.FloatTensor(3, support_num * kernel_num))
        self.key_directions = nn.Parameter(torch.FloatTensor(3, support_num * kernel_num))
        self.multihead_attention = MultiHeadedAttention(4, kernel_num, fn_attention=attention)
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.support_num * self.kernel_num)
        self.query_directions.data.uniform_(-stdv, stdv)
        self.value_directions.data.uniform_(-stdv, stdv)
        self.key_directions.data.uniform_(-stdv, stdv)

    def forward(self,
                neighbor_index: "(bs, vertice_num, neighbor_num)",
                vertices: "(bs, vertice_num, 3)"):
        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)

        support_query_direction_norm = F.normalize(self.query_directions, dim= 0) #(3, s * k)
        support_key_direction_norm = F.normalize(self.key_directions, dim= 0) #(3, s * k)
        support_value_direction_norm = F.normalize(self.value_directions, dim= 0) #(3, s * k)

        query_theta = neighbor_direction_norm @ support_query_direction_norm # (bs, vertice_num, neighbor_num, s*k)
        query_theta = self.relu(query_theta)
        query_theta = query_theta.contiguous().view(bs, vertice_num, neighbor_num, self.support_num, self.kernel_num)
        query_theta = torch.max(query_theta, dim= 2)[0] # (bs, vertice_num, support_num, kernel_num)
        query_theta = torch.sum(query_theta, dim= 2) # (bs, vertice_num, kernel_num)

        key_theta = neighbor_direction_norm @ support_key_direction_norm # (bs, vertice_num, neighbor_num, s*k)
        key_theta = self.relu(key_theta)
        key_theta = key_theta.contiguous().view(bs, vertice_num, neighbor_num, self.support_num, self.kernel_num)
        key_theta = torch.max(key_theta, dim= 2)[0] # (bs, vertice_num, support_num, kernel_num)
        key_theta = torch.sum(key_theta, dim= 2) # (bs, vertice_num, kernel_num)

        value_theta = neighbor_direction_norm @ support_value_direction_norm # (bs, vertice_num, neighbor_num, s*k)
        value_theta = self.relu(value_theta)
        value_theta = value_theta.contiguous().view(bs, vertice_num, neighbor_num, self.support_num, self.kernel_num)
        value_theta = torch.max(value_theta, dim= 2)[0] # (bs, vertice_num, support_num, kernel_num)
        value_theta = torch.sum(value_theta, dim= 2) # (bs, vertice_num, kernel_num)

        feature = self.multihead_attention(query_theta, key_theta, value_theta)

        return feature

class Attention_Conv_surface2(nn.Module):
    def __init__(self, kernel_num, support_num):
        super().__init__()
        self.kernel_num = kernel_num
        self.support_num = support_num

        self.relu = nn.ReLU(inplace= True)
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * kernel_num))
        self.multihead_attention = MultiHeadedAttention(4, kernel_num, fn_attention=attention)
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.support_num * self.kernel_num)
        self.directions.data.uniform_(-stdv, stdv)

    def forward(self,
                neighbor_index: "(bs, vertice_num, neighbor_num)",
                vertices: "(bs, vertice_num, 3)"):
        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)

        support_direction_norm = F.normalize(self.directions, dim= 0) #(3, s * k)

        theta = neighbor_direction_norm @ support_direction_norm
        theta = self.relu(theta).contiguous().view(bs, vertice_num, neighbor_num, self.support_num, self.kernel_num)
        theta = torch.max(theta, dim= 2)[0] # (bs, vertice_num, support_num, kernel_num)
        theta = torch.sum(theta, dim= 2) # (bs, vertice_num, kernel_num)

        attention_theta = self.multihead_attention(theta, theta, theta)

        return attention_theta

class Conv_layer(nn.Module):
    def __init__(self, in_channel, out_channel, support_num):
        super().__init__()
        # arguments:
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.support_num = support_num

        # parameters:
        self.relu = nn.ReLU(inplace= True)
        self.weights = nn.Parameter(torch.FloatTensor(in_channel, (support_num + 1) * out_channel))
        self.bias = nn.Parameter(torch.FloatTensor((support_num + 1) * out_channel))
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * out_channel))
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.out_channel * (self.support_num + 1))
        self.weights.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.directions.data.uniform_(-stdv, stdv)

    def forward(self,
                neighbor_index: "(bs, vertice_num, neighbor_index)",
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, in_channel)"):
        """
        Return: output feature map: (bs, vertice_num, out_channel)
        """
        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)
        support_direction_norm = F.normalize(self.directions, dim= 0)
        theta = neighbor_direction_norm @ support_direction_norm # (bs, vertice_num, neighbor_num, support_num * out_channel)
        theta = self.relu(theta)
        theta = theta.contiguous().view(bs, vertice_num, neighbor_num, -1)
        # (bs, vertice_num, neighbor_num, support_num * out_channel)

        feature_out = feature_map @ self.weights + self.bias # (bs, vertice_num, (support_num + 1) * out_channel)
        feature_center = feature_out[:, :, :self.out_channel] # (bs, vertice_num, out_channel)
        feature_support = feature_out[:, :, self.out_channel:] #(bs, vertice_num, support_num * out_channel)

        # Fuse together - max among product
        feature_support = indexing_neighbor(feature_support, neighbor_index) # (bs, vertice_num, neighbor_num, support_num * out_channel)
        activation_support = theta * feature_support # (bs, vertice_num, neighbor_num, support_num * out_channel)
        activation_support = activation_support.view(bs,vertice_num, neighbor_num, self.support_num, self.out_channel)
        activation_support = torch.max(activation_support, dim= 2)[0] # (bs, vertice_num, support_num, out_channel)
        activation_support = torch.sum(activation_support, dim= 2)    # (bs, vertice_num, out_channel)
        feature_fuse = feature_center + activation_support # (bs, vertice_num, out_channel)
        return feature_fuse


class Attention_Conv_layer_V2(nn.Module):
    def __init__(self, in_channel, out_channel, support_num):
        super().__init__()
        # arguments:
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.support_num = support_num

        # parameters:
        self.relu = nn.ReLU(inplace= True)
        self.weights = nn.Parameter(torch.FloatTensor(in_channel, (support_num + 1) * out_channel))
        self.bias = nn.Parameter(torch.FloatTensor((support_num + 1) * out_channel))
        self.directions = nn.Parameter(torch.FloatTensor(3, support_num * out_channel))
        self.multihead_attention = MultiHeadedAttention(4, out_channel, fn_attention=attention)
        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.out_channel * (self.support_num + 1))
        self.weights.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)
        self.directions.data.uniform_(-stdv, stdv)

    def forward(self,
                neighbor_index: "(bs, vertice_num, neighbor_index)",
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, in_channel)"):
        """
        Return: output feature map: (bs, vertice_num, out_channel)
        """
        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)
        support_direction_norm = F.normalize(self.directions, dim= 0)
        theta = neighbor_direction_norm @ support_direction_norm # (bs, vertice_num, neighbor_num, support_num * out_channel)
        theta = self.relu(theta)
        theta = theta.contiguous().view(bs, vertice_num, neighbor_num, -1)
        # (bs, vertice_num, neighbor_num, support_num * out_channel)

        feature_out = feature_map @ self.weights + self.bias # (bs, vertice_num, (support_num + 1) * out_channel)
        feature_center = feature_out[:, :, :self.out_channel] # (bs, vertice_num, out_channel)
        feature_support = feature_out[:, :, self.out_channel:] #(bs, vertice_num, support_num * out_channel)

        # Fuse together - max among product
        feature_support = indexing_neighbor(feature_support, neighbor_index) # (bs, vertice_num, neighbor_num, support_num * out_channel)
        activation_support = theta * feature_support # (bs, vertice_num, neighbor_num, support_num * out_channel)
        activation_support = activation_support.view(bs,vertice_num, neighbor_num, self.support_num, self.out_channel)
        activation_support = torch.max(activation_support, dim= 2)[0] # (bs, vertice_num, support_num, out_channel)
        activation_support = torch.sum(activation_support, dim= 2)    # (bs, vertice_num, out_channel)
        feature_fuse = feature_center + activation_support # (bs, vertice_num, out_channel)
        feature_fuse = self.multihead_attention(feature_fuse, feature_fuse, feature_fuse)
        return feature_fuse

class Attention_Conv_layer(nn.Module):
    def __init__(self, in_channel, out_channel, support_num):
        super().__init__()
        # arguments:
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.support_num = support_num

        # parameters:
        self.relu = nn.ReLU(inplace= True)
        self.query_weights = nn.Parameter(torch.FloatTensor(in_channel, (support_num + 1) * out_channel))
        self.query_bias = nn.Parameter(torch.FloatTensor((support_num + 1) * out_channel))
        self.query_directions = nn.Parameter(torch.FloatTensor(3, support_num * out_channel))

        self.key_weights = nn.Parameter(torch.FloatTensor(in_channel, (support_num + 1) * out_channel))
        self.key_bias = nn.Parameter(torch.FloatTensor((support_num + 1) * out_channel))
        self.key_directions = nn.Parameter(torch.FloatTensor(3, support_num * out_channel))

        self.value_weights = nn.Parameter(torch.FloatTensor(in_channel, (support_num + 1) * out_channel))
        self.value_bias = nn.Parameter(torch.FloatTensor((support_num + 1) * out_channel))
        self.value_directions = nn.Parameter(torch.FloatTensor(3, support_num * out_channel))

        self.multihead_attention = MultiHeadedAttention(4, out_channel, fn_attention=attention)

        self.initialize()

    def initialize(self):
        stdv = 1. / math.sqrt(self.out_channel * (self.support_num + 1))
        self.query_weights.data.uniform_(-stdv, stdv)
        self.query_bias.data.uniform_(-stdv, stdv)
        self.query_directions.data.uniform_(-stdv, stdv)

        self.key_weights.data.uniform_(-stdv, stdv)
        self.key_bias.data.uniform_(-stdv, stdv)
        self.key_directions.data.uniform_(-stdv, stdv)

        self.value_weights.data.uniform_(-stdv, stdv)
        self.value_bias.data.uniform_(-stdv, stdv)
        self.value_directions.data.uniform_(-stdv, stdv)

    def forward(self,
                neighbor_index: "(bs, vertice_num, neighbor_index)",
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, in_channel)"):
        bs, vertice_num, neighbor_num = neighbor_index.size()
        neighbor_direction_norm = get_neighbor_direction_norm(vertices, neighbor_index)

        # query
        support_query_direction_norm = F.normalize(self.query_directions, dim= 0)
        query_theta = neighbor_direction_norm @ support_query_direction_norm # (bs, vertice_num, neighbor_num, support_num * out_channel)
        query_theta = self.relu(query_theta)
        query_theta = query_theta.contiguous().view(bs, vertice_num, neighbor_num, -1)
        # (bs, vertice_num, neighbor_num, support_num * out_channel)

        query_feature_out = feature_map @ self.query_weights + self.query_bias # (bs, vertice_num, (support_num + 1) * out_channel)
        query_feature_center = query_feature_out[:, :, :self.out_channel] # (bs, vertice_num, out_channel)
        query_feature_support = query_feature_out[:, :, self.out_channel:] #(bs, vertice_num, support_num * out_channel)

        # Fuse together - max among product
        query_feature_support = indexing_neighbor(query_feature_support, neighbor_index) # (bs, vertice_num, neighbor_num, support_num * out_channel)
        query_activation_support = query_theta * query_feature_support # (bs, vertice_num, neighbor_num, support_num * out_channel)
        query_activation_support = query_activation_support.view(bs,vertice_num, neighbor_num, self.support_num, self.out_channel)
        query_activation_support = torch.max(query_activation_support, dim= 2)[0] # (bs, vertice_num, support_num, out_channel)
        query_activation_support = torch.sum(query_activation_support, dim= 2)    # (bs, vertice_num, out_channel)
        query_feature_fuse = query_feature_center + query_activation_support # (bs, vertice_num, out_channel)

        # key
        support_key_direction_norm = F.normalize(self.key_directions, dim= 0)
        key_theta = neighbor_direction_norm @ support_key_direction_norm # (bs, vertice_num, neighbor_num, support_num * out_channel)
        key_theta = self.relu(key_theta)
        key_theta = key_theta.contiguous().view(bs, vertice_num, neighbor_num, -1)
        # (bs, vertice_num, neighbor_num, support_num * out_channel)

        key_feature_out = feature_map @ self.key_weights + self.key_bias # (bs, vertice_num, (support_num + 1) * out_channel)
        key_feature_center = key_feature_out[:, :, :self.out_channel] # (bs, vertice_num, out_channel)
        key_feature_support = key_feature_out[:, :, self.out_channel:] #(bs, vertice_num, support_num * out_channel)

        # Fuse together - max among product
        key_feature_support = indexing_neighbor(key_feature_support, neighbor_index) # (bs, vertice_num, neighbor_num, support_num * out_channel)
        key_activation_support = key_theta * key_feature_support # (bs, vertice_num, neighbor_num, support_num * out_channel)
        key_activation_support = key_activation_support.view(bs,vertice_num, neighbor_num, self.support_num, self.out_channel)
        key_activation_support = torch.max(key_activation_support, dim= 2)[0] # (bs, vertice_num, support_num, out_channel)
        key_activation_support = torch.sum(key_activation_support, dim= 2)    # (bs, vertice_num, out_channel)
        key_feature_fuse = key_feature_center + key_activation_support # (bs, vertice_num, out_channel)

        # value
        support_value_direction_norm = F.normalize(self.value_directions, dim= 0)
        value_theta = neighbor_direction_norm @ support_value_direction_norm # (bs, vertice_num, neighbor_num, support_num * out_channel)
        value_theta = self.relu(value_theta)
        value_theta = value_theta.contiguous().view(bs, vertice_num, neighbor_num, -1)
        # (bs, vertice_num, neighbor_num, support_num * out_channel)

        value_feature_out = feature_map @ self.value_weights + self.value_bias # (bs, vertice_num, (support_num + 1) * out_channel)
        value_feature_center = value_feature_out[:, :, :self.out_channel] # (bs, vertice_num, out_channel)
        value_feature_support = value_feature_out[:, :, self.out_channel:] #(bs, vertice_num, support_num * out_channel)

        # Fuse together - max among product
        value_feature_support = indexing_neighbor(value_feature_support, neighbor_index) # (bs, vertice_num, neighbor_num, support_num * out_channel)
        value_activation_support = value_theta * value_feature_support # (bs, vertice_num, neighbor_num, support_num * out_channel)
        value_activation_support = value_activation_support.view(bs,vertice_num, neighbor_num, self.support_num, self.out_channel)
        value_activation_support = torch.max(value_activation_support, dim= 2)[0] # (bs, vertice_num, support_num, out_channel)
        value_activation_support = torch.sum(value_activation_support, dim= 2)    # (bs, vertice_num, out_channel)
        value_feature_fuse = value_feature_center + value_activation_support # (bs, vertice_num, out_channel)

        attention_feature = self.multihead_attention(query_feature_fuse, key_feature_fuse, value_feature_fuse)

        return attention_feature

class Pool_layer(nn.Module):
    def __init__(self, pooling_rate: int= 4, neighbor_num: int=  4):
        super().__init__()
        self.pooling_rate = pooling_rate
        self.neighbor_num = neighbor_num

    def forward(self,
                vertices: "(bs, vertice_num, 3)",
                feature_map: "(bs, vertice_num, channel_num)"):
        """
        Return:
            vertices_pool: (bs, pool_vertice_num, 3),
            feature_map_pool: (bs, pool_vertice_num, channel_num)
        """
        bs, vertice_num, _ = vertices.size()
        neighbor_index = get_neighbor_index(vertices, self.neighbor_num)
        neighbor_feature = indexing_neighbor(feature_map, neighbor_index) #(bs, vertice_num, neighbor_num, channel_num)
        pooled_feature = torch.max(neighbor_feature, dim= 2)[0] #(bs, vertice_num, channel_num)

        pool_num = int(vertice_num / self.pooling_rate)
        sample_idx = torch.randperm(vertice_num)[:pool_num]
        vertices_pool = vertices[:, sample_idx, :] # (bs, pool_num, 3)
        feature_map_pool = pooled_feature[:, sample_idx, :] #(bs, pool_num, channel_num)
        return vertices_pool, feature_map_pool

def test():
    import time
    bs = 8
    v = 1024
    dim = 3
    n = 20
    vertices = torch.randn(bs, v, dim)
    neighbor_index = get_neighbor_index(vertices, n)

    s = 3
    # conv_1 = Conv_surface(kernel_num= 32, support_num= s)
    conv_1 = Attention_Conv_surface2(kernel_num=32, support_num=s)
    # conv_2 = Conv_layer(in_channel= 32, out_channel= 64, support_num= s)
    conv_2 = Attention_Conv_layer(in_channel= 32, out_channel= 64, support_num= s)
    pool = Pool_layer(pooling_rate= 4, neighbor_num= 4)

    print("Input size: {}".format(vertices.size()))
    start = time.time()
    f1 = conv_1(neighbor_index, vertices)
    print("\n[1] Time: {}".format(time.time() - start))
    print("[1] Out shape: {}".format(f1.size()))
    start = time.time()
    f2 = conv_2(neighbor_index, vertices, f1)
    print("\n[2] Time: {}".format(time.time() - start))
    print("[2] Out shape: {}".format(f2.size()))
    start = time.time()
    v_pool, f_pool = pool(vertices, f2)
    print("\n[3] Time: {}".format(time.time() - start))
    print("[3] v shape: {}, f shape: {}".format(v_pool.size(), f_pool.size()))


if __name__ == "__main__":
    test()
