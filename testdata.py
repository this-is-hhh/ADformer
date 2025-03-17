# import numpy as np

# # 指定单个 .npy 文件的路径
# file_path = '/data0/zxj_data/APAVA/Feature/feature_01.npy'  # 替换为你的文件路径

# # 加载 .npy 文件
# data = np.load(file_path, allow_pickle=True)

# # 打印文件内容和形状
# print(f"文件路径：{file_path}")
# print(f"形状：{data.shape}")
# print(data)
# print(data)
# print(f"内容：\n{data}")

# import numpy as np

# # 加载 merged_label.npy 文件
# file_path = r'/data0/zxj_data/fNIRS/VFT/class4_classhealth_npy/merge2/shuffled_merged_label.npy'
# merged_label = np.load(file_path)

# # 检查原始形状
# print("Original shape:", merged_label.shape)

# # 确保第二维的值是从 1 到 172
# # 假设 merged_label 的形状是 (172, 2)，其中第二列是需要调整的值
# merged_label[:, 1] = np.arange(1, 173)  # 从 1 到 172

# # 检查修改后的值
# print("Modified array:", merged_label)

# # 保存修改后的数组
# np.save(file_path, merged_label)

# print(f"Modified array saved to {file_path}")

# print(merged_label)

import numpy as np
import os

# 加载数组
data_path = r'/data0/zxj_data/APAVA/Feature/feature_01.npy'
data = np.load(data_path,allow_pickle=True)

# 检查形状
print("Original data shape:", data.shape)

# # 确保存储路径存在
# save_dir = r'/data0/zxj_data/fNIRS/VFT/class4_classhealth_npy/Feature'
# os.makedirs(save_dir, exist_ok=True)

# # 拆分并保存
# for i in range(data.shape[0]):
#     # 提取单个样本
#     sample = data[i:i+1]  # 形状为 (1, 2400, 53)
    
#     # 生成文件名
#     file_name = f'feature_{i+1:03d}.npy'  # 格式为 feature_001.npy, feature_002.npy, ..., feature_172.npy
    
#     # 保存路径
#     save_path = os.path.join(save_dir, file_name)
    
#     # 保存文件
#     np.save(save_path, sample)
    
#     print(f"Saved {save_path}")

# print("All files saved successfully.")
print(data[0])