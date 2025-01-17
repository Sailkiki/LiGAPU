import torch
import torch.nn as nn
from einops import repeat
from models.FeatureExtractor import FeatureExtractor
from models.CSARegressor import CSARegressor
from models.utils import get_knn_pts, index_points
from scipy.spatial import cKDTree
import torch.nn.functional as F

class P2PNet(nn.Module):
    def __init__(self, args):
        super(P2PNet, self).__init__()
        self.args = args
        self.feature_extractor = FeatureExtractor(args)
        self.csa_regressor = CSARegressor(args)
        # self.gat_module = GATLayer(args)

    def extract_feature(self, original_pts):
        global_feats, local_feats = self.feature_extractor(original_pts)
        return global_feats, local_feats

    def Gaussian_interpolation(self, original_pts, query_pts, local_feat):
        k = 3
        sigma = 1.0
        knn_pts, knn_idx = get_knn_pts(k, original_pts, query_pts, return_idx=True)
        repeat_query_pts = repeat(query_pts, 'b c n -> b c n k', k=k)
        dist = torch.norm(knn_pts - repeat_query_pts, p=2, dim=1)
        dist_sq = dist ** 2
        weight = torch.exp(-dist_sq / (2 * sigma ** 2))
        norm = torch.sum(weight, dim=2, keepdim=True)
        weight = weight / norm
        knn_feat = index_points(local_feat, knn_idx)
        interpolated_feat = knn_feat * weight.unsqueeze(1)
        interpolated_feat = torch.sum(interpolated_feat, dim=-1)
        return interpolated_feat


    def regress_distance(self, original_pts, query_pts, global_feats, local_feats):
        device = original_pts.device
        global_feats = global_feats.to(device)
        query_pts = query_pts.to(device)
        b, c, n = original_pts.shape
        adj_matrices = []
        interpolated_local_feats = []
        for i in range(b):
            ori_pts_batch = original_pts[i].transpose(0, 1)  # (n, 3)
            local_feats_batch = []
            for feat in local_feats:
                interpolated_feat = self.Gaussian_interpolation(original_pts[i:i + 1], query_pts[i:i + 1], feat[i:i + 1])
                interpolated_feat = interpolated_feat.squeeze(0)
                interpolated_feat = interpolated_feat.transpose(0, 1)
                local_feats_batch.append(interpolated_feat.unsqueeze(0))
            interpolated_local_feats.append(torch.cat(local_feats_batch, dim=1))  # (1, c*n, n)
        agg_local_feats = torch.cat(interpolated_local_feats, dim=0)  # (b, c*(block_num+1), n)
        global_feats = repeat(global_feats, 'b c -> b c n', n=query_pts.shape[-1])
        agg_local_feats_resized = F.interpolate(agg_local_feats, size=query_pts.shape[-1], mode='linear', align_corners=False)
        agg_feats = torch.cat((query_pts, agg_local_feats_resized, global_feats), dim=1)
        agg_feats = agg_feats.permute(0, 2, 1)
        agg_feats = F.adaptive_max_pool1d(agg_feats, output_size=query_pts.shape[-1])
        agg_feats = agg_feats.permute(0, 2, 1)
        p2p = self.csa_regressor(agg_feats)
        return p2p

    def forward(self, original_pts, query_pts):
        global_feats, local_feats = self.extract_feature(original_pts)
        p2p = self.regress_distance(original_pts, query_pts, global_feats, local_feats)
        return p2p

def compute_radius_adj_matrix(points, radius):
    points_np = points.detach().cpu().numpy()
    tree = cKDTree(points_np)
    pairs = tree.query_pairs(r=radius, output_type='ndarray')
    n_points = points.size(0)
    adj_matrix = torch.sparse_coo_tensor(
        indices=torch.from_numpy(pairs.T).long(),
        values=torch.ones(len(pairs)),
        size=(n_points, n_points)
    )
    adj_matrix = adj_matrix.to_dense() + adj_matrix.t().to_dense()
    adj_matrix = (adj_matrix > 0).float()
    return adj_matrix
