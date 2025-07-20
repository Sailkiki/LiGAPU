import torch
from scipy.spatial import cKDTree
import torch.nn as nn
from models.utils import get_knn_pts, index_points
from einops import repeat, rearrange
from models.pointops.functions import pointops
from torch_geometric.nn import knn_graph
from .deltatools.geometry.grad_div_mls import build_grad_div, build_tangent_basis, estimate_basis
import torch.nn.functional as F
from torch_geometric.utils import scatter
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_scatter import scatter


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=32, use_dropout=True, use_bn=True, activation_type='relu'):
        super(SEBlock, self).__init__()
        self.use_dropout = use_dropout
        self.use_bn = use_bn
        self.activation_type = activation_type
        self.fc1 = nn.Conv1d(in_channels, in_channels // reduction, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(in_channels // reduction) if use_bn else nn.Identity()
        self.activation1 = self._get_activation(activation_type)
        self.fc2 = nn.Conv1d(in_channels // reduction, in_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(in_channels) if use_bn else nn.Identity()
        self.dropout = nn.Dropout(p=0.5) if use_dropout else nn.Identity()
        self.sigmoid = nn.Sigmoid()

    def _get_activation(self, activation_type):
        if activation_type == 'relu':
            return nn.ReLU(inplace=True)
        elif activation_type == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation_type == 'gelu':
            return nn.GELU()
        elif activation_type == 'elu':
            return nn.ELU(inplace=True)
        else:
            return nn.ReLU(inplace=True)  # Default to ReLU if unknown
    def forward(self, x):
        se_weight = x.mean(-1, keepdim=True)
        se_weight = self.fc1(se_weight)
        se_weight = self.bn1(se_weight)
        se_weight = self.activation1(se_weight)
        se_weight = self.fc2(se_weight)
        se_weight = self.bn2(se_weight)
        se_weight = self.sigmoid(se_weight)
        se_weight = self.dropout(se_weight)
        return x * se_weight


class DeltaNetSample(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.alpha = args.alpha
        self.k = args.k
        self.grad_regularizer = args.grad_regularizer
        self.grad_kernel_width = args.grad_kernel_width
        in_channels = args.in_channels
        conv_channels = [in_channels] + args.conv_channels
        mlp_depth = args.mlp_depth
        self.convs = torch.nn.ModuleList()  # 初始化空的ModuleList
        for i in range(len(conv_channels) - 1):
            is_last_layer = i == (len(conv_channels) - 2)
            current_in_channels = conv_channels[i]
            current_out_channels = conv_channels[i + 1]
            centralized = i == 0
            vector = not is_last_layer
            layer_depth = mlp_depth
            conv_layer = DGAConv(
                in_channels=current_in_channels,
                out_channels=current_out_channels,
                depth=layer_depth,
                centralized=centralized,
                vector=vector
            )
            self.convs.append(conv_layer)
        self.se_block = SEBlock(args.growth_rate)
        self.conv_delta = nn.Sequential(
            nn.Conv2d(3, args.growth_rate, 1),
            nn.BatchNorm2d(args.growth_rate),
            nn.ReLU(inplace=True)
        )
        self.post_conv = nn.Sequential(
            nn.Conv2d(args.growth_rate, args.growth_rate, 1),
            nn.BatchNorm2d(args.growth_rate),
            nn.ReLU(inplace=True)
        )

    def forward(self, feats, pts, knn_idx=None):
        if knn_idx is None:
            knn_pts, knn_idx = get_knn_pts(self.k, pts, pts, return_idx=True)
        else:
            knn_pts = index_points(pts, knn_idx)
        knn_delta = knn_pts - pts[..., None]
        temp_knn_delta = knn_delta + torch.sin(knn_delta) * torch.cos(knn_delta)
        knn_delta = temp_knn_delta
        knn_delta = self.conv_delta(knn_delta)
        tras_pts = pts.transpose(1, 2)
        b, n, _ = tras_pts.shape
        batch = torch.arange(b).repeat_interleave(n).to(tras_pts.device)
        pts_flat = tras_pts.reshape(-1, 3)
        edge_index = knn_graph(pts_flat, self.k, batch, loop=True, flow='target_to_source')
        edge_index_normal = knn_graph(pts_flat, 10, batch, loop=True, flow='target_to_source')
        normal, x_basis, y_basis = estimate_basis(pts_flat, edge_index_normal, orientation=pts_flat)
        grad, div = build_grad_div(pts_flat, normal, x_basis, y_basis, edge_index, batch,
                                   kernel_width=self.grad_kernel_width, regularizer=self.grad_regularizer)
        x = pts_flat
        v = grad @ x
        out = []
        for conv in self.convs:
            """
<<<<<<< HEAD
            The vector stream v does not participate in the output, but the v 
=======
            Here, the vector stream v does not participate in the output, but the v 
>>>>>>> e0abcb023347bbdfe55e132c642981932b703c97
            obtained in the previous loop will be used as a parameter for the next loop,
            participating in the feature operation of the scalar stream x. So cannot 
            ignore v in the code.
            """
            x, v = conv(x, v, grad, div, edge_index)
            out.append(x)
        f_out = out[-1]
        f_out = f_out.view(b, n, -1).transpose(1, 2)
        f_out = self.se_block(f_out)
        final_batch_output = f_out.unsqueeze(-1).expand(-1, -1, -1, self.k)
        resfeats = final_batch_output * knn_delta
        resfeats = self.post_conv(resfeats)
        resfeats = resfeats.sum(dim=-1)
        return resfeats

class DenseLayer(nn.Module):
    def __init__(self, args, input_dim):
        super(DenseLayer, self).__init__()
        self.conv_bottle = nn.Sequential(
            nn.Conv1d(input_dim, args.bn_size * args.growth_rate, 1),
            nn.BatchNorm1d(args.bn_size * args.growth_rate),
            nn.ReLU(inplace=True)
        )
        self.point_conv = DeltaNetSample(args)

    def forward(self, feats, pts, knn_idx=None):
        new_feats = self.conv_bottle(feats)
        new_feats = self.point_conv(new_feats, pts, knn_idx)
        return torch.cat((feats, new_feats), dim=1)


class DenseUnit(nn.Module):
    def __init__(self, args):
        super(DenseUnit, self).__init__()
        self.dense_layers = nn.ModuleList([])
        for i in range(args.layer_num):
            self.dense_layers.append(DenseLayer(args, args.feat_dim + i * args.growth_rate))

    def forward(self, feats, pts, knn_idx=None):
        for dense_layer in self.dense_layers:
            new_feats = dense_layer(feats, pts, knn_idx)
            feats = new_feats
        return feats

class Transition(nn.Module):
    def __init__(self, args):
        super(Transition, self).__init__()
        input_dim = args.feat_dim + args.layer_num * args.growth_rate
        self.trans = nn.Sequential(
            nn.Conv1d(input_dim, args.feat_dim, 1),
            nn.BatchNorm1d(args.feat_dim),
            nn.ReLU(inplace=True)
        )
    def forward(self, feats):
        new_feats = self.trans(feats)
        return new_feats

class FeatureExtractor(nn.Module):
    def __init__(self, args):
        super(FeatureExtractor, self).__init__()
        self.k = args.k
        self.conv_init = nn.Sequential(
            nn.Conv1d(3, args.feat_dim, 1),
            nn.BatchNorm1d(args.feat_dim),
            nn.ReLU(inplace=True)
        )
        self.dense_blocks = nn.ModuleList([])
        for i in range(args.block_num):
            self.dense_blocks.append(nn.ModuleList([
                DenseUnit(args),
                Transition(args)
            ]))

    def forward(self, pts):
        pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
        knn_idx = pointops.knnquery_heap(self.k, pts_trans, pts_trans).long()
        init_feats = self.conv_init(pts)
        local_feats = []
        local_feats.append(init_feats)
        for dense_block, trans in self.dense_blocks:
            new_feats = dense_block(init_feats, pts, knn_idx)
            new_feats = trans(new_feats)
            init_feats = new_feats
            local_feats.append(init_feats)
        global_feats = init_feats.max(dim=-1)[0]
        return global_feats, local_feats



### =========================== DGAConv ==============================

# """Original code""":
# def MLP(channels, bias=False, nonlin=LeakyReLU(negative_slope=0.2)):
#     return Seq(*[
#         Seq(Lin(channels[i - 1], channels[i], bias=bias), BatchNorm1d(channels[i]), nonlin)
#         for i in range(1, len(channels))
#     ])
#
# def VectorMLP(channels, batchnorm=True):
#     return Seq(*[
#         Seq(Lin(channels[i - 1], channels[i], bias=False), VectorNonLin(channels[i], batchnorm=BatchNorm1d(channels[i]) if batchnorm else None))
#         for i in range(1, len(channels))
#     ])


# Ours
class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x


class VectorMLP(nn.Module):
    def __init__(self, layers):
        super(VectorMLP, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return x


class SelfAttention(nn.Module):
    def __init__(self, in_channels, heads=1):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.query = nn.Linear(in_channels, in_channels * heads)
        self.key = nn.Linear(in_channels, in_channels * heads)
        self.value = nn.Linear(in_channels, in_channels * heads)
        self.out_proj = nn.Linear(in_channels * heads, in_channels)
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        Q = self.query(x).view(batch_size, seq_len, self.heads, -1).transpose(1, 2)
        K = self.key(x).view(batch_size, seq_len, self.heads, -1).transpose(1, 2)
        V = self.value(x).view(batch_size, seq_len, self.heads, -1).transpose(1, 2)
        attn_weights = torch.matmul(Q, K.transpose(-1, -2)) / (Q.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        out = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(out)
        return out

class DGAConv(nn.Module):
    def __init__(self, in_channels, out_channels, depth=1, centralized=False, vector=True, aggr='max',
                 attention_heads=1, num_channels=4, activations=None):
        super(DGAConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.centralized = centralized
        self.aggr = aggr
        self.attn_scalar = SelfAttention(out_channels, heads=attention_heads)
        self.attn_vector = SelfAttention(out_channels, heads=attention_heads)
        self.s_mlp_max = MLP([in_channels] + [out_channels] * depth)
        self.s_mlp = MLP([in_channels * 4] + [out_channels] * depth)
        if vector:
            self.v_mlp = VectorMLP([in_channels * 4 + out_channels * 2] + [out_channels] * depth)
        else:
            self.v_mlp = None
        self.glu_scalar = MultiChannelGLU(out_channels, out_channels, num_channels, activations=activations, learnable_gate=True)
        self.glu_vector = MultiChannelGLU(out_channels, out_channels, num_channels, activations=activations, learnable_gate=True)
        self.maxpool = nn.AdaptiveMaxPool1d(1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.layer_norm = nn.LayerNorm(out_channels)


    def forward(self, x, v, grad, div, edge_index):
        if self.centralized:
            x_edge = x[edge_index[1]] - x[edge_index[0]]
            x_max = scatter(self.s_mlp_max(x_edge), edge_index[0], dim=0, reduce=self.aggr)
        else:
            x_max = scatter(self.s_mlp_max(x)[edge_index[1]], edge_index[0], dim=0, reduce=self.aggr)
        x_cat = torch.cat([x, div @ v, curl(v, div), norm(v)], dim=1)
        x = x_max + self.s_mlp(x_cat)
        x = self.attn_scalar(x.unsqueeze(1))
        x = x.squeeze(1)
        x = self.glu_scalar(x)
        x_pool = self.maxpool(x.unsqueeze(0))
        x_pool = x_pool.squeeze(0)
        x = self.layer_norm(x + x_pool)
        if self.v_mlp is not None:
            v_cat = torch.cat([v, hodge_laplacian(v, grad, div), grad @ x], dim=1)
            v = self.v_mlp(I_J(v_cat))
        v = self.attn_vector(v.unsqueeze(1))
        v = v.squeeze(1)
        v = self.glu_vector(v)
        v_pool = self.avgpool(v.unsqueeze(0))
        v_pool = v_pool.squeeze(0)
        v = v + v_pool
        return x, v
