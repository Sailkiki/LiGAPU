import open3d as o3d
import numpy as np
import os


def split_ply_file_by_xy(ply_file, num_splits_x=4, num_splits_y=4, output_dir="output"):
    # 读取原始 .ply 文件
    pcd = o3d.io.read_point_cloud(ply_file)

    # 获取点云的坐标
    points = np.asarray(pcd.points)

    # 获取点云的 XY 范围
    min_x, max_x = points[:, 0].min(), points[:, 0].max()  # X 轴的范围
    min_y, max_y = points[:, 1].min(), points[:, 1].max()  # Y 轴的范围

    # 计算 X 和 Y 方向上的步长
    x_step = (max_x - min_x) / num_splits_x
    y_step = (max_y - min_y) / num_splits_y

    # 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = 0
    # 按 XY 平面划分点云
    for ix in range(num_splits_x):
        for iy in range(num_splits_y):
            sub_pcd = o3d.geometry.PointCloud()

            count += 1
            # 当前分块的 X 和 Y 范围
            x_min = min_x + ix * x_step
            x_max = x_min + x_step
            y_min = min_y + iy * y_step
            y_max = y_min + y_step

            # 找到位于当前分块范围内的点
            mask = (points[:, 0] >= x_min) & (points[:, 0] < x_max) & (points[:, 1] >= y_min) & (points[:, 1] < y_max)

            # 如果当前分块没有点则跳过
            if not np.any(mask):
                continue

            # 获取当前块的点和颜色
            sub_points = points[mask]
            sub_colors = np.asarray(pcd.colors)[mask]

            # 创建新的点云对象，并赋予颜色
            sub_pcd.points = o3d.utility.Vector3dVector(sub_points)
            sub_pcd.colors = o3d.utility.Vector3dVector(sub_colors)

            # 保存分割后的 .ply 文件
            out_file_name = f"HXNW_{count}.ply"
            out_file_path = os.path.join(output_dir, out_file_name)
            o3d.io.write_point_cloud(out_file_path, sub_pcd)
            print(f"Saved {out_file_path}")

    print("Finish")

# 示例调用
split_ply_file_by_xy('E:/smz/Datasets/HXNW/Merged_pc/HXNW_All.ply', num_splits_x=15, num_splits_y=21, output_dir='E:/smz/Datasets/HXNW/split_ply')