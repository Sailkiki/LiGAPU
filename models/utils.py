import torch
import math
from einops import rearrange
from models.pointops.functions import pointops
import logging
import os
import numpy as np
import random
from torch.autograd import grad
from einops import rearrange, repeat
from sklearn.neighbors import NearestNeighbors
from models.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
<<<<<<< HEAD
import torch.nn.functional as F
from einops import repeat, rearrange

chamfer_dist = chamfer_3DDist()

=======

chamfer_dist = chamfer_3DDist()



>>>>>>> e0abcb023347bbdfe55e132c642981932b703c97
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def index_points(pts, idx):
    """
    Input:
        pts: input points data, [B, C, N]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, C, S, [K]]
    """
    batch_size = idx.shape[0]
    sample_num = idx.shape[1]
    fdim = pts.shape[1]
    reshape = False
    if len(idx.shape) == 3:
        reshape = True
        idx = idx.reshape(batch_size, -1)
    # (b, c, (s k))
    res = torch.gather(pts, 2, idx[:, None].repeat(1, fdim, 1))
    if reshape:
        res = rearrange(res, 'b c (s k) -> b c s k', s=sample_num)

    return res

<<<<<<< HEAD
=======

>>>>>>> e0abcb023347bbdfe55e132c642981932b703c97
def FPS(pts, fps_pts_num):
    # input: (b, 3, n)

    # (b, n, 3)
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    # (b, fps_pts_num)
    sample_idx = pointops.furthestsampling(pts_trans, fps_pts_num).long()
    # (b, 3, fps_pts_num)
    sample_pts = index_points(pts, sample_idx)

    return sample_pts


def get_knn_pts(k, pts, center_pts, return_idx=False):
    # input: (b, 3, n)

    # (b, n, 3)
    pts_trans = rearrange(pts, 'b c n -> b n c').contiguous()
    # (b, m, 3)
    center_pts_trans = rearrange(center_pts, 'b c m -> b m c').contiguous()
<<<<<<< HEAD
    # (b, m, k)
    knn_idx = pointops.knnquery_heap(k, pts_trans, center_pts_trans).long()
    # (b, 3, m, k)
=======
    # (b, m, k) , 作用是为 center_pts_trans 找到在  pts_trans 中的最临近 k 个点索引
    knn_idx = pointops.knnquery_heap(k, pts_trans, center_pts_trans).long()
    # (b, 3, m, k)
    # b 批次 3 通道特征 每个批次m个点 每个点 k 个邻近点
>>>>>>> e0abcb023347bbdfe55e132c642981932b703c97
    knn_pts = index_points(pts, knn_idx)

    if return_idx == False:
        return knn_pts
    else:
        return knn_pts, knn_idx


def midpoint_interpolate(args, sparse_pts):
    pts_num = sparse_pts.shape[-1] # 256
    up_pts_num = int(pts_num * args.up_rate) # 1024
<<<<<<< HEAD
    k = int(2 * args.up_rate) #
=======
    k = int(2 * args.up_rate) # 默认是 2 * 4 = 8
>>>>>>> e0abcb023347bbdfe55e132c642981932b703c97
    knn_pts = get_knn_pts(k, sparse_pts, sparse_pts)
    repeat_pts = repeat(sparse_pts, 'b c n -> b c n k', k=k)
    mid_pts = (knn_pts + repeat_pts) / 2.0
    mid_pts = rearrange(mid_pts, 'b c n k -> b c (n k)')
    interpolated_pts = mid_pts
    interpolated_pts = FPS(interpolated_pts, up_pts_num)
    return interpolated_pts

<<<<<<< HEAD
def get_combined_loss(args, pred_p2p, sample_pts, gt_pts, alpha=0.6, beta=0.4):
    knn_pts = get_knn_pts(1, gt_pts, sample_pts).squeeze(-1)
    gt_p2p = torch.norm(knn_pts - sample_pts, p=2, dim=1, keepdim=True)  # (b, 1, n)
    p2p_loss = torch.nn.L1Loss(reduction='none')(pred_p2p, gt_p2p)  # (b, 1, n)
    p2p_loss = p2p_loss.squeeze(1).sum(dim=-1).mean()  # (b, n) -> (b) -> scalar
    dist1, _ = get_knn_pts(1, gt_pts, sample_pts, return_idx=True)
    dist1 = dist1.squeeze(-1)
    dist2, _ = get_knn_pts(1, sample_pts, gt_pts, return_idx=True)  #
=======



import torch
import torch.nn.functional as F
from einops import repeat, rearrange


def get_combined_loss(args, pred_p2p, sample_pts, gt_pts, alpha=0.6, beta=0.4):
    knn_pts = get_knn_pts(1, gt_pts, sample_pts).squeeze(-1)  # 最近邻点 (b, 3, n)
    gt_p2p = torch.norm(knn_pts - sample_pts, p=2, dim=1, keepdim=True)  # (b, 1, n)
    p2p_loss = torch.nn.L1Loss(reduction='none')(pred_p2p, gt_p2p)  # (b, 1, n)
    p2p_loss = p2p_loss.squeeze(1).sum(dim=-1).mean()  # (b, n) -> (b) -> scalar
    dist1, _ = get_knn_pts(1, gt_pts, sample_pts, return_idx=True)  # sample_pts 到 gt_pts 的距离 (b, n, 1)
    dist1 = dist1.squeeze(-1)
    dist2, _ = get_knn_pts(1, sample_pts, gt_pts, return_idx=True)  # gt_pts 到 sample_pts 的距离 (b, n, 1)
>>>>>>> e0abcb023347bbdfe55e132c642981932b703c97
    dist2 = dist2.squeeze(-1)
    if args.truncate_distance:
        dist1 = torch.clamp(dist1, max=args.max_dist)
        dist2 = torch.clamp(dist2, max=args.max_dist)
    chamfer_loss = dist1.mean() + dist2.mean()
    total_loss = alpha * p2p_loss + beta * chamfer_loss
    return total_loss

<<<<<<< HEAD
# Local Geometric Integrity Loss
def local_geometric_integrity_loss(args, pred_p2p, sample_pts, gt_pts):
    # input: (b, c, n)
    knn_pts = get_knn_pts(args.k, gt_pts, sample_pts)
    closest_knn_pts = knn_pts[..., 0]
    gt_p2p = torch.norm(closest_knn_pts - sample_pts, p=2, dim=1, keepdim=True)
    for i in range(1, args.k):
        knn_distance = torch.norm(knn_pts[..., i] - sample_pts, p=2, dim=1, keepdim=True)
        gt_p2p += knn_distance

    gt_p2p /= args.k
=======


def get_p2p_loss(args, pred_p2p, sample_pts, gt_pts):
    # input: (b, c, n)
    # (b, 3, n)
    knn_pts = get_knn_pts(1, gt_pts, sample_pts).squeeze(-1)
    # (b, 1, n)
    gt_p2p = torch.norm(knn_pts - sample_pts, p=2, dim=1, keepdim=True)

    if args.use_smooth_loss == True:
        if args.truncate_distance == True:
            loss = torch.nn.SmoothL1Loss(reduction='none', beta=args.beta)(torch.clamp(pred_p2p, max=args.max_dist), torch.clamp(gt_p2p, max=args.max_dist))
        else:
            loss = torch.nn.SmoothL1Loss(reduction='none', beta=args.beta)(pred_p2p, gt_p2p)
    else:
        if args.truncate_distance == True:
            loss = torch.nn.L1Loss(reduction='none')(torch.clamp(pred_p2p, max=args.max_dist), torch.clamp(gt_p2p, max=args.max_dist))
        else:
            loss = torch.nn.L1Loss(reduction='none')(pred_p2p, gt_p2p)
    # (b, 1, n) -> (b, n) -> (b) -> scalar
    loss = loss.squeeze(1).sum(dim=-1).mean()
    return loss



    # if args.use_smooth_loss == True:
    #     if args.truncate_distance == True:
    #         loss = torch.nn.SmoothL1Loss(reduction='none', beta=args.beta)(torch.clamp(pred_p2p, max=args.max_dist), torch.clamp(gt_p2p, max=args.max_dist))
    #     else:
    #         loss = torch.nn.SmoothL1Loss(reduction='none', beta=args.beta)(pred_p2p, gt_p2p)
    # else:
    #     if args.truncate_distance == True:
    #         loss = torch.nn.L1Loss(reduction='none')(torch.clamp(pred_p2p, max=args.max_dist), torch.clamp(gt_p2p, max=args.max_dist))
    #     else:






# Local Geometric Integrity Loss
def local_geometric_integrity_loss(args, pred_p2p, sample_pts, gt_pts):
    # input: (b, c, n)

    # 获取每个样本点的局部邻域（K近邻），引入拓扑约束 (b, 3, n, k)
    knn_pts = get_knn_pts(args.k, gt_pts, sample_pts)

    # 获取每个点的最近邻 (b, 3, n)
    closest_knn_pts = knn_pts[..., 0]  # 假设第一个邻居为最近的点
    # 计算最近邻的欧几里得距离 (b, 1, n)
    gt_p2p = torch.norm(closest_knn_pts - sample_pts, p=2, dim=1, keepdim=True)

    # 添加拓扑约束的损失，通过考虑点和其邻居的损失 (拓扑感知损失)
    for i in range(1, args.k):
        knn_distance = torch.norm(knn_pts[..., i] - sample_pts, p=2, dim=1, keepdim=True)
        gt_p2p += knn_distance  # 将每个邻域点的距离累加（拓扑约束）

    gt_p2p /= args.k  # 取平均，使得损失值合理化
>>>>>>> e0abcb023347bbdfe55e132c642981932b703c97

    # (b, 1, n)
    if args.use_smooth_loss == True:
        if args.truncate_distance == True:
            loss = torch.nn.SmoothL1Loss(reduction='none', beta=args.beta)(
                torch.clamp(pred_p2p, max=args.max_dist),
                torch.clamp(gt_p2p, max=args.max_dist)
            )
        else:
            loss = torch.nn.SmoothL1Loss(reduction='none', beta=args.beta)(pred_p2p, gt_p2p)
    else:
        if args.truncate_distance == True:
            loss = torch.nn.L1Loss(reduction='none')(
                torch.clamp(pred_p2p, max=args.max_dist),
                torch.clamp(gt_p2p, max=args.max_dist)
            )
        else:
            loss = torch.nn.L1Loss(reduction='none')(pred_p2p, gt_p2p)

    # (b, 1, n) -> (b, n) -> (b) -> scalar
    loss = loss.squeeze(1).sum(dim=-1).mean()

    return loss

def normalize_point_cloud(input, centroid=None, furthest_distance=None):
    # input: (b, 3, n) tensor

    if centroid is None:
        # (b, 3, 1)
        centroid = torch.mean(input, dim=-1, keepdim=True)
    # (b, 3, n)
    input = input - centroid
    if furthest_distance is None:
        # (b, 3, n) -> (b, 1, n) -> (b, 1, 1)
        furthest_distance = torch.max(torch.norm(input, p=2, dim=1, keepdim=True), dim=-1, keepdim=True)[0]
    input = input / furthest_distance

    return input, centroid, furthest_distance


def add_noise(pts, sigma, clamp):
    # input: (b, 3, n)
<<<<<<< HEAD
=======

>>>>>>> e0abcb023347bbdfe55e132c642981932b703c97
    assert (clamp > 0)
    jittered_data = torch.clamp(sigma * torch.randn_like(pts), -1 * clamp, clamp).cuda()
    jittered_data += pts

    return jittered_data

<<<<<<< HEAD
=======

>>>>>>> e0abcb023347bbdfe55e132c642981932b703c97
# generate patch for test
def extract_knn_patch(k, pts, center_pts):
    # input : (b, 3, n)

    # (n, 3)
    pts_trans = rearrange(pts.squeeze(0), 'c n -> n c').contiguous()
    pts_np = pts_trans.detach().cpu().numpy()
    # (m, 3)
    center_pts_trans = rearrange(center_pts.squeeze(0), 'c m -> m c').contiguous()
    center_pts_np = center_pts_trans.detach().cpu().numpy()
    knn_search = NearestNeighbors(n_neighbors=k, algorithm='auto')
    knn_search.fit(pts_np)
    # (m, k)
    knn_idx = knn_search.kneighbors(center_pts_np, return_distance=False)
    # (m, k, 3)
    patches = np.take(pts_np, knn_idx, axis=0)
    patches = torch.from_numpy(patches).float().cuda()
    # (m, 3, k)
    patches = rearrange(patches, 'm k c -> m c k').contiguous()

    return patches

<<<<<<< HEAD
=======

>>>>>>> e0abcb023347bbdfe55e132c642981932b703c97
def get_logger(name, log_dir):
    # 创建一个名为name的日志记录器
    logger = logging.getLogger(name)
    # 设置级别为DEBUG
    logger.setLevel(logging.DEBUG)
    # 用于定义日志消息的格式
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')
    # 输出到控制台
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    # output to log file
    log_name = name + '_log.txt'
    file_handler = logging.FileHandler(os.path.join(log_dir, log_name))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

<<<<<<< HEAD
def get_query_points(input_pts, args):
    query_pts = input_pts + (torch.randn_like(input_pts) * args.local_sigma)
=======

def get_query_points(input_pts, args):
    # 添加随机扰动
    query_pts = input_pts + (torch.randn_like(input_pts) * args.local_sigma)

>>>>>>> e0abcb023347bbdfe55e132c642981932b703c97
    return query_pts


def reset_model_args(train_args, model_args):
    for arg in vars(train_args):
<<<<<<< HEAD
=======
        #把train_args中的所有属性复制到model_args中  arg相当于键 getatter获取对应的值
>>>>>>> e0abcb023347bbdfe55e132c642981932b703c97
        setattr(model_args, arg, getattr(train_args, arg))



