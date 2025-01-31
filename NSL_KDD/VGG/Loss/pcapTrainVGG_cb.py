# 纯resnet 就检查全部特征的时候 resnet的准确率
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os
from pcapVGGSeed import VGG # 导入DNN类
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
BETA = 0.9999 # 类平衡损失的β
GAMMA = 2

curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
model = VGG("ce", SEED, BETA, GAMMA)
start_time = time.time()
for _ in range(EPOCH_NUM):
    model.train()
    model.epoch_count += 1
end_time = time.time()  # 记录训练结束时间
total_training_time = end_time - start_time  # 计算训练总时长
model_dir = "./model"
new_folder = "model_" + curr_time
os.mkdir(os.path.join(model_dir, new_folder))
model_path = os.path.join(model_dir, new_folder, "model.ckpt")
print("loss_history",model.loss_history)
print("macro_F1List",model.macro_F1List)
print("micro_F1List",model.micro_F1List)

saver = tf.compat.v1.train.Saver()
with model.sess as sess:
    saver.save(sess, model_path)
    print('start testing...')
    accuracy = model.test()
