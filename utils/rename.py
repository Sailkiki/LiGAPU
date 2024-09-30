import os

input_dir = r'E:/smz/Datasets/HXNW_ALL/test/gt_8192'

files = os.listdir(input_dir)

for file in files:
    if "GT8192" in file:
        # 生成新的文件名
        new_filename = file.replace("GT8192", "input2048")

        old_file = os.path.join(input_dir, file)
        new_file = os.path.join(input_dir, new_filename)

        # 重命名文件
        os.rename(old_file, new_file)

print("done")

