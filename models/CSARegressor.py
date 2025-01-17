import torch.nn as nn
import torch
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv1d(in_channels, in_channels // reduction, 1)
        self.fc2 = nn.Conv1d(in_channels // reduction, in_channels, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=-1, keepdim=True)
        max_pool, _ = torch.max(x, dim=-1, keepdim=True)
        x = avg_pool + max_pool
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv1d(1, 1, kernel_size, padding=(kernel_size // 2))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        x = avg_pool + max_pool
        x = self.conv1(x)
        return self.sigmoid(x)



class CSARegressor(nn.Module):
    def __init__(self, args):
        super(CSARegressor, self).__init__()
        input_dim = ((args.feat_dim * (args.block_num + 2)) - 2) * 2
        self.mlp_0 = nn.Conv1d(input_dim, args.feat_dim * 2, 1)
        self.mlp_1 = nn.Conv1d(args.feat_dim * 2, args.feat_dim, 1)
        self.mlp_2 = nn.Conv1d(args.feat_dim, args.feat_dim // 2, 1)
        self.mlp_3 = nn.Conv1d(args.feat_dim // 2, 1, 1)
        self.actvn = nn.ReLU()
        self.channel_attention = ChannelAttention(args.feat_dim // 2)
        self.spatial_attention = SpatialAttention(kernel_size=7)

    def forward(self, feats):
        x = self.actvn(self.mlp_0(feats))
        x = self.actvn(self.mlp_1(x))
        x = self.actvn(self.mlp_2(x))
        channel_attention_map = self.channel_attention(x)
        x = x * channel_attention_map
        spatial_attention_map = self.spatial_attention(x)
        x = x * spatial_attention_map
        x = self.mlp_3(x)
        return x
