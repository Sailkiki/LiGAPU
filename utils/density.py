import open3d as o3d
import numpy as np
import os
from glob import glob

# 定义函数来补充点云到 8192 个点
def pad_point_cloud(point_cloud, colors, target_num_points=8192):
    current_num_points = point_cloud.shape[0]
    if current_num_points >= target_num_points:
        return point_cloud[:target_num_points], colors[:target_num_points]
    else:
        extra_points_needed = target_num_points - current_num_points
        extra_indices = np.random.choice(current_num_points, extra_points_needed, replace=True)
        extra_points = point_cloud[extra_indices]
        extra_colors = colors[extra_indices]
        padded_point_cloud = np.vstack((point_cloud, extra_points))
        padded_colors = np.vstack((colors, extra_colors))
        return padded_point_cloud, padded_colors

# 加载 .ply 文件并使用 open3d 处理
def load_ply(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    point_cloud = np.asarray(pcd.points)  # 将点云数据转为 numpy 数组
    colors = np.asarray(pcd.colors)  # 读取颜色信息
    return point_cloud, colors

# 保存 .ply 文件
def save_ply(filepath, point_cloud, colors):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)  # 将 numpy 数组转为点云格式
    pcd.colors = o3d.utility.Vector3dVector(colors)  # 将颜色信息添加到点云
    o3d.io.write_point_cloud(filepath, pcd)

# 文件夹路径
input_dir = r'E:/smz/Datasets/HXNW_ALL/test/gt_8192'
output_dir = r'E:/smz/Datasets/HXNW_ALL/test/gt_8192_2'

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 处理所有 .ply 文件
ply_files = glob(os.path.join(input_dir, "*.ply"))
for ply_file in ply_files:
    point_cloud, colors = load_ply(ply_file)
    if point_cloud.shape[0] < 8192:
        padded_point_cloud, padded_colors = pad_point_cloud(point_cloud, colors, target_num_points=8192)
        save_ply(os.path.join(output_dir, os.path.basename(ply_file)), padded_point_cloud, padded_colors)
        print(f"Processed and padded: {ply_file}")
    else:
        print(f"File already has 8192 points: {ply_file}")
