import numpy as np

# 读取原始数据
with open('C:/Users/Pumpkins/Desktop/xd.txt', 'r') as f:
    data = [float(line.strip()) for line in f]

# 计算当前平均值
current_avg = np.mean(data)

# 目标平均值
target_avg = 0.000324772

# 计算调整因子
adjustment_factor = target_avg / current_avg

# 调整数据
adjusted_data = [x * adjustment_factor for x in data]

# 验证新的平均值
new_avg = np.mean(adjusted_data)

print(f"Original average: {current_avg:.9f}")
print(f"New average: {new_avg:.9f}")
print(f"Target average: {target_avg:.9f}")

# 输出调整后的数据
for value in adjusted_data:
    print(f"{value:.9f}")