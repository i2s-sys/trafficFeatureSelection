# 使用早停策略
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os
from pcapVGG2Seed import VGG2 # 导入DNN类
import matplotlib.pyplot as plt

# 获取当前脚本的文件名
file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")
SEED = 25
K = 32 # topk 特征
WIDTHLITMIT = 1024 # 位宽限制加大
TRAIN_EPOCH = 30
ES_THRESHOLD = 3
selected_features =[20, 19]
# selected_features =[19, 20, 18, 56]
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.compat.v1.Session(config=config)
curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())

if __name__ == '__main__':
    print("selected_features",selected_features)
    model2 = VGG2("cb_focal_loss",dim=len(selected_features), selected_features=selected_features,seed=SEED)
    start_time = time.time()
    for _ in range(TRAIN_EPOCH):
        model2.train()
        model2.epoch_count += 1
    end_time = time.time()
    total_training_time = end_time - start_time
    print("dnn2_loss_history", model2.loss_history)
    model2.test()