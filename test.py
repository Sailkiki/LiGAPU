import torch
import numpy as np
from glob import glob
import os
import open3d as o3d
from models.utils import *
from models.LiGAPU import LiGAPU
from einops import rearrange
from time import time
from args.pu1k_args import parse_pu1k_args
# from args.pugan_args import parse_pugan_args
from args.utils import str2bool
from tqdm import tqdm
import argparse


def normalize_colors_for_ply(colors):
    if isinstance(colors, torch.Tensor):
        colors = colors.detach().cpu().numpy()

    # Clip values to [0, 1] range
    colors_normalized = np.clip(colors, 0, 1)

    return colors_normalized


def pcd_update(args, model, interpolated_pcd):
    # interpolated_pcd: (b, 3, n)

    pcd_pts_num = interpolated_pcd.shape[-1]
    # 1024
    patch_pts_num = args.num_points * 4
    # extract patch
    sample_num = int(pcd_pts_num / patch_pts_num * args.patch_rate)
    # FPS: (b, 3, fps_pts_num), ensure seeds have a good coverage
    seed = FPS(interpolated_pcd, sample_num)
    # (b*fps_pts_num, 3, patch_pts_num)
    patches = extract_knn_patch(patch_pts_num, interpolated_pcd, seed)

    # normalize each patch
    patches, centroid, furthest_distance = normalize_point_cloud(patches)

    # fix the parameters of model while updating the patches
    for param in model.parameters():
        param.requires_grad = False

    # initialize updated_patch
    updated_patch = patches.clone()
    updated_patch.requires_grad = True

    # extract the global and local features and fix them
    global_feats, local_feats = model.extract_feature(patches)

    for i in range(args.num_iterations):
        # predict point-to-point distance: (b, 1, n)
        pred_p2p = model.regress_distance(patches, updated_patch, global_feats, local_feats)
        if args.truncate_distance == True:
            pred_p2p = torch.clamp(pred_p2p, max=args.max_dist)
        # back-propagation
        loss = pred_p2p.mean()
        loss.backward()

        # update patch
        gradient = updated_patch.grad.detach()
        updated_patch = updated_patch.detach()
        updated_patch = updated_patch - args.test_step_size * gradient

        # enable the gradient calculation
        updated_patch.requires_grad = True

    # transform to original scale and merge patches
    updated_patch = centroid + updated_patch * furthest_distance
    # (3, m)
    updated_pcd = rearrange(updated_patch, 'b c n -> c (b n)').contiguous()
    # post process: (1, 3, n)
    output_pts_num = interpolated_pcd.shape[-1]
    updated_pcd = FPS(updated_pcd.unsqueeze(0), output_pts_num)

    return updated_pcd


def pcd_upsample(args, model, input_pcd):
    # input: (b, 3, n)

    # interpolate: (b, 3, m)
    interpolated_pcd = midpoint_interpolate(args, input_pcd)
    # interpolated_color = midpoint_interpolate(args, pcd_color)
    # update: (b, 3, m)
    updated_pcd = pcd_update(args, model, interpolated_pcd)
    # updated_color = pcd_update(args, model, interpolated_color)

    return updated_pcd


def test(args):
    # load model
    model = P2PNet(args).cuda()
    model.load_state_dict(torch.load(args.ckpt_path))
    model.eval()

    # test input data path list
    test_input_path = glob(os.path.join(args.test_input_path, '*.xyz'))
    # print(test_input_path)
    # conduct 4X twice to get 16X
    if args.up_rate == 16:
        args.up_rate = 4
        args.double_4X = True

    # log
    output_dir = os.path.join(args.ckpt_path, '../..')
    output_dir = os.path.abspath(output_dir)
    test_dir = os.path.join(output_dir, 'test')
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    logger = get_logger('test', test_dir)


    pcd_dir = os.path.join(test_dir, args.save_dir)
    if not os.path.exists(pcd_dir):
        os.makedirs(pcd_dir)
    # record time
    total_pcd_time = 0.0

    # test
    for i, path in tqdm(enumerate(test_input_path), desc='Processing'):
        start = time()
        # each time upsample one point cloud
        pcd = o3d.io.read_point_cloud(path)

        file_name = os.path.basename(path)

        # 打印文件名称
        print("File name:", file_name)

        pcd_name = path.split('/')[-1]
        # 坐标信息
        input_pcd = np.array(pcd.points)
        # 颜色信息
        # pcd_color = np.array(pcd.colors)
        input_pcd = torch.from_numpy(input_pcd).float().cuda()
        # pcd_color = torch.from_numpy(pcd_color).float().cuda()

        # (n, 3) -> (3, n)
        input_pcd = rearrange(input_pcd, 'n c -> c n').contiguous()
        # pcd_color = rearrange(pcd_color, 'n c -> c n').contiguous()
        # (3, n) -> (1, 3, n)
        input_pcd = input_pcd.unsqueeze(0)
        # pcd_color = pcd_color.unsqueeze(0)
        # normalize input
        input_pcd, centroid, furthest_distance = normalize_point_cloud(input_pcd)
        # pcd_color, centroid1, furthest_distance1 = normalize_point_cloud(pcd_color)


        upsampled_pcd = pcd_upsample(args, model, input_pcd)
        # print("\nupsampled_pcd.shape:", upsampled_pcd.shape,
        #       "\ninput_pcd.shape:", input_pcd.shape,
        #       "\npcd_color.shape:", pcd_color.shape,
        #       "\ncentroid.shape:", centroid.shape,
        #       "\nfurthest_distance.shape:", furthest_distance.shape)
        # if pcd.has_colors():
        #     print("11")
        upsampled_pcd = centroid + upsampled_pcd * furthest_distance
        # upsampled_color = centroid1 + upsampled_color * furthest_distance1

        # upsample 16X, conduct 4X twice
        if args.double_4X == True:
            upsampled_pcd, centroid, furthest_distance = normalize_point_cloud(upsampled_pcd)
            upsampled_pcd = pcd_upsample(args, model, upsampled_pcd)
            upsampled_pcd = centroid + upsampled_pcd * furthest_distance

        # (b, 3, n) -> (n, 3)
        upsampled_pcd = rearrange(upsampled_pcd.squeeze(0), 'c n -> n c').contiguous()
        # upsampled_color = rearrange(upsampled_color.squeeze(0), 'c n -> n c').contiguous()

        upsampled_pcd = upsampled_pcd.detach().cpu().numpy()
        # upsampled_color = upsampled_color.detach().cpu().numpy()
        # save path
        # save_path = os.path.join(pcd_dir, pcd_name)
        # np.savetxt(save_path, upsampled_pcd, fmt='%.6f')

        save_path = os.path.join(pcd_dir, pcd_name)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(upsampled_pcd)
        # upsampled_color = normalize_colors_for_ply(upsampled_color)
        # pcd.colors = o3d.utility.Vector3dVector(upsampled_color)


        o3d.io.write_point_cloud(save_path, pcd, write_ascii=False)

        # time
        end = time()
        total_pcd_time += end - start

    logger.info('Average pcd time: {}s'.format(total_pcd_time / len(test_input_path)))


def parse_test_args():
    parser = argparse.ArgumentParser(description='Test Arguments')

    parser.add_argument('--dataset', default='pu1k', type=str, help='pu1k or pugan')
    parser.add_argument('--test_input_path', default='/home/smz/Code/Grad-PU_ori/data/PU1K/test/input_2048/input_2048', type=str,
                        help='the test input data path')
    parser.add_argument('--ckpt_path', default='/home/smz/Code/Grad-PU_ori/output/1216/ckpt/ckpt-epoch-3.pth', type=str,
                        help='the pretrained model path')
    parser.add_argument('--save_dir', default='pcd', type=str, help='save upsampled point cloud')
    parser.add_argument('--truncate_distance', default=False, type=str2bool, help='whether truncate distance')
    parser.add_argument('--up_rate', default=4, type=int, help='upsampling rate')
    parser.add_argument('--double_4X', default=False, type=str2bool, help='conduct 4X twice to get 16X')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    test_args = parse_test_args()
    assert test_args.dataset in ['pu1k', 'pugan']

    if test_args.dataset == 'pu1k':
        model_args = parse_pu1k_args()
    else:
        model_args = parse_pugan_args()

    reset_model_args(test_args, model_args)

    test(model_args)