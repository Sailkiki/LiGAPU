import os
import numpy as np
import open3d as o3d
import h5py

def read_ply_files(folder_path):
    point_clouds = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.ply'):
            file_path = os.path.join(folder_path, filename)
            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points)
            point_clouds.append(points)
    return np.array(point_clouds)

def create_h5_dataset(poisson512_folder, poisson2048_folder, output_file):
    # 读取两个文件夹中的PLY文件
    poisson1024_data = read_ply_files(poisson512_folder)
    poisson256_data = read_ply_files(poisson2048_folder)

    # 确保两个数据集的样本数量相同
    assert len(poisson1024_data) == len(poisson256_data), "The number of files in both folders should be the same"

    # 创建H5文件并保存数据
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('poisson512', data=poisson1024_data)
        f.create_dataset('poisson2048', data=poisson256_data)

    print(f"H5 dataset created: {output_file}")
    print(f"Shape of poisson512 dataset: {poisson1024_data.shape}")
    print(f"Shape of poisson2048 dataset: {poisson256_data.shape}")

# 使用示例
if __name__ == '__main__':
    poisson512_folder = 'E:/smz/Datasets/HXNW_ALL/HXNW_Poisson512'  # 替换为实际路径
    poisson2048_folder = 'E:/smz/Datasets/HXNW_ALL/HXNW_Poisson2048'  # 替换为实际路径
    output_file = 'E:/smz/Datasets/HXNW_ALL/HXNW_PU_dataset/HXNW_poisson_512_and_poisson_2048.h5'

    create_h5_dataset(poisson512_folder, poisson2048_folder, output_file)