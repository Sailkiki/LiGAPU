import os
import open3d as o3d
import numpy as np

def convert_64bit_to_32bit_ply(input_dir, output_dir):
    # 如果输出文件夹不存在，创建文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历输入文件夹中的所有文件
    files = os.listdir(input_dir)
    for i, file in enumerate(files):
        file_path = os.path.join(input_dir, file)

        if file_path.endswith(".ply"):  # 只处理PLY文件
            print(f"Processing file {i + 1}/{len(files)}: {file}")

            # 读取点云文件
            pcd = o3d.io.read_point_cloud(file_path)

            # 将点云中的浮点数据转换为32位
            pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points).astype(np.float32))

            # 如果有颜色信息，转换颜色为32位
            if pcd.has_colors():
                pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors).astype(np.float32))

            # 如果有法线信息，转换法线为32位
            if pcd.has_normals():
                pcd.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals).astype(np.float32))

            # 构建输出文件路径
            output_file = os.path.join(output_dir, f"converted_{file}")

            # 保存为32位的PLY文件
            o3d.io.write_point_cloud(output_file, pcd)
            print(f"Saved converted file to: {output_file}")

    print("Conversion completed!")

# 使用示例
input_folder = 'E:/smz/Datasets/HXNW_Poisson4096'  # 替换为您的64位PLY文件所在的文件夹路径
output_folder = 'E:/smz/Datasets/HXNW_Poisson4096_32bit'  # 替换为您想保存32位PLY文件的文件夹路径

convert_64bit_to_32bit_ply(input_folder, output_folder)