# TensorFlow 2.9.0 compatible second training script for UNSW-IoT AEaddRF
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os
from AEAddRF2 import AE2
import matplotlib.pyplot as plt

# 设置环境变量
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
K = 72   # 使用全部特征
TRAIN_EPOCH = 30

curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

# 选择缩放因子来源: "pso" / "sca" / "fpa" / "inf" / "file" / "file_top32"
SCALING_METHOD = "file_top32" # "pso"
PSO_TOP32 = [70, 68, 66, 65, 62, 57, 56, 55, 43, 53, 48, 46, 27, 30, 25, 24,
             14, 23, 22, 21, 20, 18, 17, 15, 8, 13, 11, 9, 4, 5, 0, 61]
SCA_TOP32 = [58, 0, 17, 2, 36, 56, 65, 4, 14, 6, 18, 22, 15, 1, 70, 60, 13, 16, 19, 9, 20, 26, 31, 63, 43, 23, 10, 68, 67, 69, 66, 71]
FPA_TOP32 = [71, 60, 38, 55, 11, 14, 12, 26, 8, 69, 25, 19, 56, 68, 48, 53, 3, 30, 65, 47, 24, 54, 6, 15, 2, 43, 59, 16, 9, 63, 5, 1]
INF_TOP32 = [50, 23, 13, 18, 40, 45, 26, 31, 16, 21, 12, 14, 35, 19, 20, 24, 15, 27, 25, 17, 22, 10, 9, 68, 66, 41, 43, 42, 48, 46, 47, 38]

# 全部特征参与，缩放因子由 scaling_factor.npy 控制
scaling_path = os.path.join(os.path.dirname(__file__), f"scaling_factor.npy") # 获取factor的不用加new
if SCALING_METHOD == "pso":
    full_scaling = np.zeros((1, K), dtype=np.float32)
    full_scaling[0, PSO_TOP32] = 1.0
    np.save(scaling_path, full_scaling)
    print(f"[PSO] Saved fixed scaling mask to {scaling_path}")
elif SCALING_METHOD == "sca":
    full_scaling = np.zeros((1, K), dtype=np.float32)
    full_scaling[0, SCA_TOP32] = 1.0
    np.save(scaling_path, full_scaling)
    print(f"[SCA] Saved fixed scaling mask to {scaling_path}")
elif SCALING_METHOD == "fpa":
    full_scaling = np.zeros((1, K), dtype=np.float32)
    full_scaling[0, FPA_TOP32] = 1.0
    np.save(scaling_path, full_scaling)
    print(f"[SCA] Saved fixed scaling mask to {scaling_path}")
elif SCALING_METHOD == "inf":
    full_scaling = np.zeros((1, K), dtype=np.float32)
    full_scaling[0, INF_TOP32] = 1.0
    np.save(scaling_path, full_scaling)
    print(f"[inf] Saved fixed scaling mask to {scaling_path}")
elif SCALING_METHOD == "file_top32":
    if os.path.exists(scaling_path):
        full = np.load(scaling_path)
        if full.ndim == 2 and full.shape[1] >= K:
            masked = np.zeros_like(full, dtype=np.float32)
            top_idx = np.argsort(full.flatten())[::-1][:32]
            masked.reshape(-1)[top_idx] = 1.0
            np.save(scaling_path, masked.astype(np.float32))
            print(f"[FILE_TOP32] Saved top-32 mask to {scaling_path}")
        else:
            print(f"[WARN] scaling_factor shape mismatch for top32: {full.shape}")
    else:
        print(f"[WARN] scaling_factor file not found for top32: {scaling_path}")
elif SCALING_METHOD == "file":
    print(f"[FILE] Use existing scaling factor at {scaling_path}")
else:
    print(f"[WARN] Unknown SCALING_METHOD={SCALING_METHOD}, fallback to file: {scaling_path}")

selected_features = list(range(K))
print("Using all features, K=", K)

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
