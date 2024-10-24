import torch
import numpy as np
import h5py


# load and normalize data
def load_h5_data(args):
    num_points = args.num_points
    # 默认 1024
    num_4X_points = int(args.num_points * 4)
    # 默认 1024   256 * 4
    num_out_points = int(args.num_points * args.up_rate)
    skip_rate = args.skip_rate
    use_random_input = args.use_random_input
    h5_file_path = args.h5_file_path

    if use_random_input:
        with h5py.File(h5_file_path, 'r') as f:
            # (b, n, 3)
            input = f['poisson_%d' % num_4X_points][:]
            # (b, n, 3)
            gt = f['poisson_%d' % num_out_points][:]
    else:
        with h5py.File(h5_file_path, 'r') as f:
            input = f['poisson_%d' % num_points][:]
            gt = f['poisson_%d' % num_out_points][:]
    # (b, n, c) 判断两者是否相等
    assert input.shape[0] == gt.shape[0]

    # (b, 1) 初始化全为1
    data_radius = np.ones(shape=(input.shape[0], 1))
    # 沿着第二维点数为轴，计算第三位坐标的均值，得到每一个点云的质心
    input_centroid = np.mean(input, axis=1, keepdims=True)
    # 使质心位于原点
    input = input - input_centroid
    # 计算每个点云中点到原点的最大距离，具体步骤
    # 1. 对每个点的 x, y, z 坐标平方后求和，得到每个点到原点的平方距离，形状为 (b, n)
    # 2. 对平方距离取平方根，得到每个点到原点的实际距离
    # 3. 沿 n 轴找到每个点云中最大距离，结果形状为 (b, 1)
    input_furthest_distance = np.amax(np.sqrt(np.sum(input ** 2, axis=-1)), axis=1, keepdims=True)
    # 将每个点云的点坐标除以对应的最大距离，进行归一化。
    # 将 input_furthest_distance 的形状从 (b, 1) 扩展为 (b, 1, 1)，以便广播到 input 的形状 (b, n, 3)
    input = input / np.expand_dims(input_furthest_distance, axis=-1)
    # gt进行同样的操作
    gt = gt - input_centroid
    gt = gt / np.expand_dims(input_furthest_distance, axis=-1)
    # 进行下采样处理，sr默认是1，不会有影响
    input = input[::skip_rate]
    gt = gt[::skip_rate]
    data_radius = data_radius[::skip_rate]

    return input, gt, data_radius


# nonuniform sample point cloud to get input data
def nonuniform_sampling(num, sample_num):
    sample = set()
    loc = np.random.rand() * 0.8 + 0.1
    while len(sample) < sample_num:
        # 生成一个均值为 loc、标准差为 0.3 的随机数
        a = int(np.random.normal(loc=loc, scale=0.3) * num)
        # 检查
        if a < 0 or a >= num:
            continue
        sample.add(a)
    return list(sample)


# data augmentation
def jitter_perturbation_point_cloud(input, sigma=0.005, clip=0.02):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, jittered batch of point clouds
    """
    N, C = input.shape
    assert(clip > 0)
    # 生成随机噪声
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    jittered_data += input
    return jittered_data


def rotate_point_cloud_and_gt(input, gt=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, rotated batch of point clouds
    """
    # 生成三个随机角度，代表在xyz轴上，范围为 0 - 2π
    angles = np.random.uniform(size=(3)) * 2 * np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))
    input = np.dot(input, rotation_matrix)
    if gt is not None:
        gt = np.dot(gt, rotation_matrix)
    return input, gt


def random_scale_point_cloud_and_gt(input, gt=None, scale_low=0.5, scale_high=2):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            Nx3 array, original batch of point clouds
        Return:
            Nx3 array, scaled batch of point clouds
    """
    scale = np.random.uniform(scale_low, scale_high)
    input = np.multiply(input, scale)
    if gt is not None:
        gt = np.multiply(gt, scale)
    return input, gt, scale