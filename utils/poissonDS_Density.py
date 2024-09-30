import open3d as o3d
import os
import numpy as np
from glob import glob

def pad_point_cloud(point_cloud, colors, target_num_points):
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


def poisson_disk_sampling(num_points):

    path = r'E:/smz/Datasets/HXNW_All/HXNW/split_ply'
    out_path = r'E:/smz/Datasets/HXNW_ALL/HXNW_Poisson2048'
    # 遍历文件夹
    files = os.listdir(path)
    for i, file in enumerate(files):
        out_file_name = f"HXNW512_{i + 1}.ply"
        out_file_path = os.path.join(out_path, out_file_name)

        file_path = os.path.join(path, file)
        # 读取PLY文件
        pcd = o3d.io.read_point_cloud(file_path)

        total_points = len(pcd.points)

        if num_points > total_points:
            print(f"Requested number of samples ({num_points}) exceeds point cloud size ({total_points}).")
            point_cloud = np.asarray(pcd.points)
            colors = np.asarray(pcd.colors)
            # 进行加密操作
            padded_point_cloud, padded_colors = pad_point_cloud(point_cloud, colors, target_num_points=num_points)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(padded_point_cloud)  # 将 numpy 数组转为点云格式
            pcd.colors = o3d.utility.Vector3dVector(padded_colors)  # 将颜色信息添加到点云
            print(f"Density has been increased")

        # 执行泊松盘采样
        sampled_pcd = pcd.farthest_point_down_sample(num_points)

        num_points = 2048
        # 保存采样后的点云
        o3d.io.write_point_cloud(out_file_path, sampled_pcd)
        print(f"The {i + 1}th point cloud done.")

    print("Finish")



if __name__ == '__main__':
    target_points = 2048  # 目标点数
    poisson_disk_sampling(target_points)

