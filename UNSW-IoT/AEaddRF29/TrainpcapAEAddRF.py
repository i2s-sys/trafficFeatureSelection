# TensorFlow 2.9.0 compatible training script for UNSW-IoT AEaddRF
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os
from pcapAEAddRF import AE
import matplotlib.pyplot as plt

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
K = 32 # topk 特征
TRAIN_EPOCH = 30

curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

ae = AE(seed=SEED)
print('start training...')

start_time = time.time()
for _ in range(TRAIN_EPOCH):
    ae.train()
    ae.epoch_count += 1
end_time = time.time()
total_training_time = end_time - start_time
print("ae_loss_history", ae.loss_history)
print('start testing...')
ae.train_classifier()
