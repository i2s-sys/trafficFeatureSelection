# TensorFlow 2.9.0 compatible second training script for UNSW-IoT AEaddRF
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os
from AEAddRF2 import AE2
import matplotlib.pyplot as plt

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 减少TensorFlow日志输出

# 配置GPU内存增长
def configure_gpu():
    """配置GPU设置"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU配置成功: {len(gpus)} 个GPU")
            return True
        else:
            print("未检测到GPU，将使用CPU")
            return False
    except Exception as e:
        print(f"GPU配置失败: {e}")
        return False

# 执行GPU配置
gpu_available = configure_gpu()

# 获取当前脚本的文件名
file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")

SEED = 25
K = 16   # topk 特征
TRAIN_EPOCH = 30

curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

# 使用预定义的特征选择结果
# sorted_indices = [50,23,18,13,45,40,35,26,16,21,31,12,19,14,20,24,15,27,17,22,25,10,9,68,66,67,38,43,41,36,42,47]
# sorted_indices = [71, 69, 68, 65, 61, 59, 58, 57, 55, 53, 50, 48, 43, 37, 34, 30, 25, 24, 23, 21, 20, 19, 17, 16, 15, 14, 13, 9, 8, 5, 4, 0] # pso
# sorted_indices = [14, 70, 3, 12, 62, 55, 23, 25, 61, 20, 51, 56, 24, 18, 15, 21, 48, 13, 17, 9, 59, 26, 32, 68, 5, 67, 66, 71, 8, 7, 69, 65] # sca
# sorted_indices = [60, 34, 13, 62, 11, 10, 24, 70, 61, 12, 30, 27, 28, 14, 15, 68, 26, 2, 52, 65, 22, 7, 18, 45, 67, 53, 4, 35, 20, 55, 21, 19] # fpa
sorted_indices = [12, 28, 48, 29, 40, 35, 67, 43, 38, 9, 26, 49, 70, 69, 16, 21, 42, 64, 62, 71, 27, 37, 44, 39, 58, 22, 17, 20, 15, 54, 55, 33]


top_k_indices = sorted_indices[:K]
print("K=",K,"top_k_indices",top_k_indices)
selected_features = top_k_indices

ae2 = AE2(dim=len(selected_features), selected_features=selected_features, seed=SEED)
print('start retraining...')

start_time = time.time()
for _ in range(TRAIN_EPOCH):
    ae2.train()
    ae2.epoch_count += 1
end_time = time.time()
total_training_time = end_time - start_time
print("ae2_loss_history", ae2.loss_history)
print('start testing...')
ae2.train_classifier()
