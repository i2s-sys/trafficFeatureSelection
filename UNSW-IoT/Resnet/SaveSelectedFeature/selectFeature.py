# import sys
# import time
# import tensorflow as tf
# import numpy as np
# import csv, os
# from pcapResnetPacketSeed import VGG,Resnet2 # 导入DNN类
# import matplotlib.pyplot as plt
#
# # 获取当前脚本的文件名
# file_name = os.path.basename(__file__)
# print(f"当前脚本的文件名是: {file_name}")
#
# SEED = 25
# DATA_DIM = 72 # 特征数
# K = 8 # topk 特征
# WIDTHLITMIT = 1024 # 位宽限制加大
# TRAIN_EPOCH = 30
# ES_THRESHOLD = 2
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# config.allow_soft_placement = True
# sess = tf.compat.v1.Session(config=config)
#
# selected_features = [10, 15, 25, 64, 20, 44, 42, 16, 21, 26, 35, 22, 39, 43, 17, 9, 27, 31, 45, 6, 14, 18, 12, 67, 49, 24, 38, 40, 19, 66,
#  68, 48]
# dnn2 = Resnet2(dim=len(selected_features), selected_features=selected_features,seed=SEED)
#     start_time = time.time()
#     for _ in range(TRAIN_EPOCH):
#         delta_loss, count = dnn2.train()
#         dnn2.epoch_count += 1
#     end_time = time.time()
#     total_training_time = end_time - start_time
#     print("dnn2_loss_history", dnn2.loss_history)
#     print("dnn2_macro_F1List", dnn2.macro_F1List)
#     print("dnn2_micro_F1List", dnn2.micro_F1List)
#     print('start testing...')
#     accuracy2 = dnn2.test()