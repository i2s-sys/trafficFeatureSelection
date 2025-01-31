# 纯resnet 就检查全部特征的时候 resnet的准确率
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os
from pcapResnetPureSeed import Resnet # 导入DNN类
import matplotlib.pyplot as plt

# 获取当前脚本的文件名
file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.compat.v1.Session(config=config)

SEED = 25
EPOCH_NUM = 30

curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
model = Resnet("ce",SEED,2,0)
model_dir = "./model"
new_folder = "model_" + curr_time
new_folder2 = new_folder + "best_model"
os.mkdir(os.path.join(model_dir, new_folder2))
os.mkdir(os.path.join(model_dir, new_folder))
start_time = time.time()
best_accuracy = 0.0  # 初始化最高accuracy
best_model_path = None  # 初始化最佳模型路径
model.train_start = 0
saver = tf.compat.v1.train.Saver()
for _ in range(EPOCH_NUM):
    accuracy  = model.train()
    model.epoch_count += 1
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_path = os.path.join(model_dir, new_folder2, f"model_best_epoch_{model.epoch_count}.ckpt")
        saver.save(model.sess, best_model_path)
end_time = time.time()
total_training_time = end_time - start_time  # 计算训练总时长

with model.sess as sess:
    saver.restore(sess, best_model_path)
    print('best model test result: ')
    model.test()
