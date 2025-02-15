# 为了得到最好的特征 先不用早停策略 后续对比用了早停策略的精度和所选特征
import argparse
import sys
import time
import tensorflow as tf
import numpy as np
import csv, os
from Resnet_loss import Resnet
from tensorflow.keras.backend import clear_session
import matplotlib.pyplot as plt

# 获取当前脚本的文件名
file_name = os.path.basename(__file__)
print(f"当前脚本的文件名是: {file_name}")

SEED = 25
 
K = 32 # topk 特征
WIDTHLITMIT = 1024 # 位宽限制加大
TRAIN_EPOCH = 30
ES_THRESHOLD = 3

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
sess = tf.compat.v1.Session(config=config)

parser = argparse.ArgumentParser(description='Specify number of features.')
parser.add_argument('--l1F', type=float, default='0.001', help='l1')
parser.add_argument('--Q', type=int, default=3, help='Number of ES_THRESHOLD')
parser.add_argument('--lossType', type=str, default='cb', help='')
parser.add_argument('--beta', type=float, default=0.999, help='')
parser.add_argument('--gamma', type=float, default= 2, help='')
args = parser.parse_args()

curr_time = time.strftime("%Y%m%d%H%M%S", time.localtime())
model = Resnet(K,args.Q,SEED,args.l1F,args.lossType,args.beta, args.gamma)
start_time = time.time()

model_dir = "./model"
new_folder = "model_" + curr_time
os.mkdir(os.path.join(model_dir, new_folder))
best_accuracy = 0.0  # 初始化最高accuracy
best_model_path = None  # 初始化最佳模型路径
new_folder2 = new_folder + "best_model"
os.mkdir(os.path.join(model_dir, new_folder2))
saver = tf.compat.v1.train.Saver()

model.train_start = 0
for _ in range(TRAIN_EPOCH):
    accuracy = model.train()
    model.epoch_count += 1
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_path = os.path.join(model_dir, new_folder2, f"model_best_epoch_{model.epoch_count}.ckpt")
        saver.save(model.sess, best_model_path)
end_time = time.time()
total_training_time = end_time - start_time  # 计算训练总时长
model.test()