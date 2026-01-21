
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os
from AEAddRF2 import AE2
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  


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


gpu_available = configure_gpu()


file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")

SEED = 25
K = 16   
TRAIN_EPOCH = 30

curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())






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
