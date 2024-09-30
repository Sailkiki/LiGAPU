import open3d as o3d
import numpy as np

# 加载点云文件
pcd = o3d.io.read_point_cloud('E:/smz/Datasets/HXNW_Poisson4096_32bit/converted_HXNW4096_1.ply')
points_32bit = np.asarray(pcd.points).astype(np.float32)
print(points_32bit.dtype)
pcd.points = o3d.utility.Vector3dVector(points_32bit)

# 检查点数据的类型
points = np.asarray(pcd.points)
print(points.dtype)  # 打印点的浮点格式，例如 float32 或 float64
