import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 文件路径
file_path = r'E:/smz/Datasets/PU1K/train/pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5'

# 读取 H5 文件
with h5py.File(file_path, 'r') as file:
    # 打印文件中可用的键
    print("Available datasets:", list(file.keys()))

    # 假设点云数据存储在名为 'data' 的数据集中
    # 根据实际数据集名称进行修改
    data = file['poisson_1024'][:]
    # 检查数据的形状
    print("Shape of the data:", data.shape)

# 选择其中一个点云进行可视化，例如第一个点云
single_cloud = data[10]  # 选择第一个点云
x, y, z = single_cloud[:, 0], single_cloud[:, 1], single_cloud[:, 2]

# 创建一个 3D 图形
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 绘制点云
ax.scatter(x, y, z, c=z, cmap='viridis', marker='o', s=1)

# 设置图形的标题和标签
ax.set_title('3D Point Cloud Visualization of Single Sample')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

# 显示图形
plt.show()