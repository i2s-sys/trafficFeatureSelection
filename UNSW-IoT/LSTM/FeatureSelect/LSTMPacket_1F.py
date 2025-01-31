import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from sklearn.metrics import f1_score
import numpy as np
import csv, random
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, GlobalMaxPool2D, Reshape, Multiply, Lambda

# 禁用 Eager Execution
tf.compat.v1.disable_eager_execution()

DATA_DIM = 72 # 特征
OUTPUT_DIM = 29  # 0-17类
BETA = 0.999 # 类平衡损失的β
GAMMA = 2
n_units = 128
TRAINDATA = '../../train_data.csv'
TESTDATA = '../../test_data.csv'

# Hyper Parameters
LEARNING_RATE = 0.0001
BATCH_SIZE = 128

MODEL_SAVE_PATH = './model/'
MODEL_SAVE_PATH2 = './model2/'
MODEL_NAME = 'model'
MODEL_NAME2 = 'model2'
KEEP_PROB = 0.5
top_k_values=[]
top_k_indice=[]
NUM_ATTENTION_CHANNELS=1

feature_widths = [
    32, 32, 32, 32,  # fiat_mean, fiat_min, fiat_max, fiat_std
    32, 32, 32, 32,  # biat_mean, biat_min, biat_max, biat_std
    32, 32, 32, 32,  # diat_mean, diat_min, diat_max, diat_std
    32,              # duration 13
    64, 32, 32, 32, 32,  # fwin_total, fwin_mean, fwin_min, fwin_max, fwin_std
    64, 32, 32, 32, 32,  # bwin_total, bwin_mean, bwin_min, bwin_max, bwin_std
    64, 32, 32, 32, 32,  # dwin_total, dwin_mean, dwin_min, dwin_max, dwin_std
    16, 16, 16,         # fpnum, bpnum, dpnum
    32, 32, 32, 32,         # bfpnum_rate, fpnum_s, bpnum_s, dpnum_s 22
    64, 32, 32, 32, 32,  # fpl_total, fpl_mean, fpl_min, fpl_max, fpl_std
    64, 32, 32, 32, 32,  # bpl_total, bpl_mean, bpl_min, bpl_max, bpl_std
    64, 32, 32, 32, 32,  # dpl_total, dpl_mean, dpl_min, dpl_max, dpl_std
    32, 32, 32, 32,         # bfpl_rate, fpl_s, bpl_s, dpl_s  19
    16, 16, 16, 16, 16, 16, 16, 16,  # fin_cnt, syn_cnt, rst_cnt, pst_cnt, ack_cnt, urg_cnt, cwe_cnt, ece_cnt
    16, 16, 16, 16,     # fwd_pst_cnt, fwd_urg_cnt, bwd_pst_cnt, bwd_urg_cnt
    16, 16, 16,         # fp_hdr_len, bp_hdr_len, dp_hdr_len
    32, 32, 32          # f_ht_len, b_ht_len, d_ht_len 18
]

# 增强数据 如果不足10000就 复制直到10000条 否则就筛选出10000条
def augment_data_to_target(data_list, target_length=10000): # 平衡每个类全部都是10000个样本 使用平衡损失时 不使用
    for i in range(len(data_list)):
        # 如果数据长度小于目标长度，则进行增强
        while len(data_list[i]) < target_length:
            shortage = target_length - len(data_list[i])
            # 如果现有数据不足以补充到目标长度，则将整个数据集复制一份
            if shortage > len(data_list[i]):
                data_list[i] += data_list[i]
            else:
                # 否则，从现有数据中随机选择缺少的数据量进行补充
                data_list[i] += random.sample(data_list[i], shortage)
        # 当数据量达到或超过目标长度后，随机采样目标长度的数据
        data_list[i] = random.sample(data_list[i], target_length)
    return data_list

def normalization(self,minibatch):
    data = np.delete(minibatch, -1, axis=1)
    labels = np.array(minibatch,dtype=np.int32)[:, -1]
    mmax = np.max(data, axis=0)
    mmin = np.min(data, axis=0)
    for i in range(len(mmax)):
        if mmax[i] == mmin[i]:
            mmax[i] += 0.000001     # avoid getting devided by 0
    res = (data - mmin) / (mmax - mmin)
    res = np.c_[res,labels]
    return res


class LSTM():
    def __init__(self,lossType,seed,K,ES_THRESHOLD):
        self.lossType = lossType
        self.gamma = GAMMA
        self.K = K
        self.ES_THRESHOLD = ES_THRESHOLD
        self.beta = BETA
        self.seed = seed
        self.nowSet = set()  # 本次和上次的因子集合
        self.lastSet = set()
        self.NowIntersection = set()  # 本次和上次的交集
        self.lastIntersection = set()
        self.maintainCnt = 0  # 阈值为0 的时候集合不变的次数
        self.loss_history = []
        self.intersection_sets = []
        self.TSMRecord = []
        self.earlyStop = False

        self.learning_rate = LEARNING_RATE
        print(f"BETA = {BETA}, GAMMA = {GAMMA}")
        self.batch_size = BATCH_SIZE
        self.epoch_count = 0
        self.x_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, DATA_DIM])
        self.target = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None])
        self.train_step = tf.Variable(0, trainable=False)
        self.loss_history = []
        self.micro_F1List = []
        self.macro_F1List = []
        self.init_data()
        self.total_iterations_per_epoch = self.train_length // BATCH_SIZE
        self.create_LSTM()

        beta = BETA    # cb数据预处理
        ClassNum = len(self.label_status)
        effective_num = {}
        for key, value in self.label_status.items():
            new_value = (1.0 - beta) / (1.0 - np.power(beta, value))
            effective_num[key] = new_value
        # 计算好有效数 之后 使用的是有效数的权重
        total_effective_num = sum(effective_num.values())
        self.weights = {}
        for key, value in effective_num.items():
            new_value = effective_num[key] / total_effective_num * ClassNum
            self.weights[key] = new_value
        self.sess = tf.compat.v1.Session()
        self.build_loss()  # 构建损失

        self.ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        self.saver = tf.compat.v1.train.Saver()
        self.train_start = 0
        if self.ckpt and self.ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)
            self.train_start = self.sess.run(self.train_step)

    def create_LSTM(self):
        # 缩放因子层
        tf.random.set_seed(self.seed)
        np.random.seed(self.seed)
        self.scaling_factor = tf.Variable(
            tf.constant(1, dtype=tf.float32, shape=[1, DATA_DIM]))
        self.scaling_factor_extended = tf.tile(self.scaling_factor, [BATCH_SIZE, 1])
        scaled_input = tf.multiply(self.x_input, self.scaling_factor_extended)
        layer = { 'weights': tf.Variable(tf.random.normal([n_units, OUTPUT_DIM])),
                  'bias': tf.Variable(tf.random.normal([OUTPUT_DIM])) }
        x = tf.split(scaled_input, DATA_DIM, 1)
        lstm_cell = tf.keras.layers.LSTM(128, return_sequences=True, return_state=True)
        x = tf.stack(x, axis=1)
        outputs, state_h, state_c = lstm_cell(x)
        self.y = tf.matmul(outputs[:, -1, :], layer['weights']) + layer['bias']

    def build_loss(self):
        l1_regularizer = tf.keras.regularizers.l1(0.001)
        regularization_penalty = l1_regularizer(self.scaling_factor)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y, labels=self.target)
        class_weights = tf.constant([self.weights[str(i)] for i in range(len(self.weights))], dtype=tf.float32)
        # cb
        sample_weights = tf.gather(class_weights, self.target)
        cbce = tf.multiply(ce, sample_weights)
        # cbfocalLoss
        softmax_probs = tf.nn.softmax(self.y)
        labels_one_hot = tf.one_hot(self.target, depth=len(self.weights))
        pt = tf.reduce_sum(labels_one_hot * softmax_probs, axis=1)
        modulator = tf.pow(1.0 - pt, self.gamma)
        focal_loss = modulator * ce
        cb_focal_loss = tf.multiply(focal_loss, sample_weights)
        if (self.lossType == "ce"):
            self.loss = tf.reduce_sum(ce) + regularization_penalty
        elif (self.lossType == "cb"):
            self.loss = tf.reduce_sum(cbce) + regularization_penalty
        elif (self.lossType == "cb_focal_loss"):
            self.loss = tf.reduce_sum(cb_focal_loss) + regularization_penalty
        self.train_op = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                                      global_step=self.train_step)
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def train(self):
        total_loss = 0
        num_batches = 0
        self.train_start = 0
        epoch_start_time = time.time()  # 记录epoch开始时间
        for _ in range(self.total_iterations_per_epoch):
            step = self.train_start
            batch = self.get_a_train_batch(step)
            data, label = self.get_data_label(batch)
            _, y = self.sess.run([self.train_op, self.y], feed_dict={self.x_input: data, self.target: label})
            curr_loss, y = self.sess.run([self.loss, self.y], feed_dict={self.x_input: data, self.target: label})
            total_loss += curr_loss
            num_batches += 1
            self.train_start += 1
        average_loss = total_loss / num_batches
        self.loss_history.append(average_loss)
        macro_F1, micro_F1 = self.test2()
        self.micro_F1List.append(micro_F1)
        self.macro_F1List.append(macro_F1)
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch {self.epoch_count + 1} completed, average loss: {average_loss}, duration: {epoch_duration:.2f} seconds')
        ''' 早停策略'''
        keyFeatureNums = self.K
        values, indices = tf.math.top_k(self.scaling_factor, k=keyFeatureNums, sorted=True)
        max_indices = self.sess.run(indices)
        max_indices = max_indices[0]
        max_set = set(max_indices)  # 本次迭代得到的因子集合
        self.intersection_sets.append(max_set)  # 将当前epoch的关键特征下标集合添加到列表中
        if len(self.intersection_sets) > self.ES_THRESHOLD:  # 保持列表长度
            self.intersection_sets.pop(0)
        # if len(self.intersection_sets) == self.ES_THRESHOLD:  # 当列表中有self.ES_THRESHOLD个集合时，直接计算交集
            intersection = set.intersection(*self.intersection_sets)
            print(f"Epoch {self.epoch_count + 1}, Intersection size: {len(intersection)}")
            self.TSMRecord.append(len(intersection))
            if len(intersection) == keyFeatureNums:
                print("Early stopping condition met.")
                self.earlyStop = True
        if self.epoch_count == 0:  # 第一个epoch
            # 初始化损失的变化
            delta_loss = 0
            self.count = 0
            self.curr_loss = average_loss
        else:
            # 计算损失的变化
            self.prev_loss = self.curr_loss  # 上一个epoch的损失
            self.curr_loss = average_loss  # 新epoch的损失
            delta_loss = abs(self.curr_loss - self.prev_loss)
            # 判断损失的变化是否小于阈值
            if delta_loss <= 0.03:
                # 次数加一
                self.count += 1
            else:
                # 次数归零
                self.count = 0
        # 返回损失的变化
        return delta_loss, self.count

    def get_a_train_batch(self, step):
        '''从训练数据集中获取一个批次的数据 '''
        min_index = step * self.batch_size
        max_index = min_index + self.batch_size
        if max_index > self.train_length:
            max_index = self.train_length
        return self.train_data[min_index:max_index]  # 返回min到max之间的数据样本

    def get_data_label(self, batch):
        '''划分数据特征 和 最后一列标签'''
        data = np.delete(batch, -1, axis=1) # 从输入的批次数据中删除最后一列，以提取仅包含特征的数据
        label = np.array(batch, dtype=np.int32)[:, -1]
        return data, label

    def init_data(self):
        self.train_data = []
        self.test_data = []  # 初始化训练和测试数据
        self.label_status = {}
        filename = TRAINDATA
        csv_reader = csv.reader(open(filename))
        label_data = {i: [] for i in range(29)}
        for row in csv_reader:
            data = [np.float32(char) if char != 'None' else 0 for char in row]
            label = int(data[-1])
            if label in label_data:
                label_data[label].append(data)
            self.label_status[str(label)] = self.label_status.get(str(label), 0) + 1
        self.train_data = [data for label in label_data for data in label_data[label]]
        filename = TESTDATA
        csv_reader = csv.reader(open(filename))
        for row in csv_reader:
            data = [np.float32(char) if char != 'None' else 0 for char in row]
            self.test_data.append(data)
        self.train_length = len(self.train_data)
        self.test_length = len(self.test_data)
        self.train_data = self.normalization(self.train_data)
        self.test_data = self.normalization(self.test_data)
         
        print('init data completed!')

    def normalization(self, minibatch):
        data = np.delete(minibatch, -1, axis=1)
        labels = np.array(minibatch, dtype=np.int64)[:, -1]
        mmax = np.max(data, axis=0)
        mmin = np.min(data, axis=0)
        for i in range(len(mmax)):
            if mmax[i] == mmin[i]:
                mmax[i] += 0.000001  # avoid getting devided by 0
        res = (data - mmin) / (mmax - mmin)
        res = np.c_[res, labels]
        return res

    def predict_top_k(self, x_feature, k=5):
        predict = self.sess.run(self.y, feed_dict={self.x_input: [x_feature]})[0]
        top_k_indices = np.argsort(predict)[-k:][::-1]  # 获取前k个预测标签的索引
        return top_k_indices

    def test(self):
        label_count = {}  # 标签的总数量
        label_correct = {}  # 标签预测正确的数量
        length = len(self.test_data)
        count = 0
        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        y_true = []
        y_pred = []

        for row in self.test_data:
            feature = row[0:-1]
            label = row[-1]
            count += 1
            x_feature = np.array(feature)
            top_k_labels = self.predict_top_k(x_feature, k=5)  # 获取前5个预测标签
            y_true.append(label)
            y_pred.append(top_k_labels[0])  # 记录TOP-1预测结果

            if str(int(label)) not in label_count:
                label_count[str(int(label))] = 0
                label_correct[str(int(label))] = 0
            if label == top_k_labels[0]:  # TOP-1预测  逐个判断top1-top5
                label_correct[str(int(label))] += 1
                top1_correct += 1
            if label in top_k_labels[:3]:  # TOP-3预测
                top3_correct += 1
            if label in top_k_labels[:5]:  # TOP-5预测
                top5_correct += 1
            label_count[str(int(label))] += 1
            if count % 10000 == 0:
                print(f"Processed {count} rows")

        accuracy1 = {}
        for label in sorted(label_count):
            accuracy1[label] = label_correct[label] / label_count[label]
            print(label, accuracy1[label], label_correct[label], label_count[label])

        top1_accuracy = top1_correct / length
        top3_accuracy = top3_correct / length
        top5_accuracy = top5_correct / length
        print(f"Top-1 accuracy: {top1_accuracy}")
        print(f"Top-3 accuracy: {top3_accuracy}")
        print(f"Top-5 accuracy: {top5_accuracy}")

        # 计算Macro-F1和Micro-F1
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        print(f"Macro-F1: {macro_f1}")
        print(f"Micro-F1: {micro_f1}")
        return top1_accuracy
    
    def test2(self): # 仅返回top1测试结果
        y_true = []
        y_pred = []
        for row in self.test_data:
            feature = row[0:-1]
            label = row[-1]
            x_feature = np.array(feature)
            top_k_labels = self.predict_top_k(x_feature, k=5)  # 获取前5个预测标签
            y_true.append(label)
            y_pred.append(top_k_labels[0])  # 记录TOP-1预测结果
        # 计算Macro-F1和Micro-F1
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        print(f"Macro-F1: {macro_f1}")
        print(f"Micro-F1: {micro_f1}")
        return macro_f1,micro_f1

class LSTM2(): # 第二次训练
    def __init__(self,lossType,seed,dim,selected_features=[]):
        self.lossType = lossType
        self.seed = seed
        self.dim = dim
        self.top_k_indices = selected_features
        print(f"BETA = {BETA}, GAMMA = {GAMMA}")
        self.learning_rate = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.epoch_count = 0
        self.x_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, dim])
        self.target = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None])
        self.train_step = tf.Variable(0, trainable=False)
        self.loss_history = []
        self.micro_F1List = []
        self.macro_F1List = []
        self.init_data()
        self.total_iterations_per_epoch = self.train_length // BATCH_SIZE
        self.create_LSTM(dim)
        beta = BETA  # cb数据预处理
        ClassNum = len(self.label_status)
        effective_num = {}
        for key, value in self.label_status.items():
            new_value = (1.0 - beta) / (1.0 - np.power(beta, value))
            effective_num[key] = new_value
        # 计算好有效数 之后 使用的是有效数的权重
        total_effective_num = sum(effective_num.values())
        self.weights = {}
        for key, value in effective_num.items():
            new_value = effective_num[key] / total_effective_num * ClassNum
            self.weights[key] = new_value
        self.sess = tf.compat.v1.Session()
        self.build_loss()  # 构建损失
        self.ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH2)
        self.saver = tf.compat.v1.train.Saver()
        self.train_start = 0
        if self.ckpt and self.ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, self.ckpt.model_checkpoint_path)
            self.train_start = self.sess.run(self.train_step)

    def create_LSTM(self, NEW_DIM):
            tf.random.set_seed(self.seed)
            np.random.seed(self.seed)
            layer = {'weights': tf.Variable(tf.random.normal([n_units, OUTPUT_DIM])),
                     'bias': tf.Variable(tf.random.normal([OUTPUT_DIM]))}
            x = tf.split(self.x_input, NEW_DIM, 1)
            lstm_cell = tf.keras.layers.LSTM(128, return_sequences=True, return_state=True)
            x = tf.stack(x, axis=1)
            outputs, state_h, state_c = lstm_cell(x)
            self.y = tf.matmul(outputs[:, -1, :], layer['weights']) + layer['bias']

    def build_loss(self):
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y, labels=self.target)
        class_weights = tf.constant([self.weights[str(i)] for i in range(len(self.weights))], dtype=tf.float32)
        # cb
        sample_weights = tf.gather(class_weights, self.target)
        cbce = tf.multiply(ce, sample_weights)
        # cbfocalLoss
        softmax_probs = tf.nn.softmax(self.y)
        labels_one_hot = tf.one_hot(self.target, depth=len(self.weights))
        pt = tf.reduce_sum(labels_one_hot * softmax_probs, axis=1)
        modulator = tf.pow(1.0 - pt, GAMMA)
        focal_loss = modulator * ce
        cb_focal_loss = tf.multiply(focal_loss, sample_weights)
        if (self.lossType == "ce"):
            self.loss = tf.reduce_sum(ce)
        elif (self.lossType == "cb"):
            self.loss = tf.reduce_sum(cbce)
        elif (self.lossType == "cb_focal_loss"):
            self.loss = tf.reduce_sum(cb_focal_loss)
        self.train_op = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss,
                                                                                      global_step=self.train_step)
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def train(self):  # 每次训练都是一个epoch遍历完整个训练集
        total_loss = 0
        num_batches = 0
        self.train_start = 0
        epoch_start_time = time.time()  # 记录epoch开始时间
        for _ in range(self.total_iterations_per_epoch):
            step = self.train_start
            batch = self.get_a_train_batch(step)
            data, label = self.get_data_label(batch)
            _, y = self.sess.run([self.train_op, self.y], feed_dict={self.x_input: data, self.target: label})
            curr_loss, y = self.sess.run([self.loss, self.y], feed_dict={self.x_input: data, self.target: label})
            total_loss += curr_loss
            num_batches += 1
            self.train_start += 1
        average_loss = total_loss / num_batches
        self.loss_history.append(average_loss)
        macro_F1, micro_F1 = self.test2()
        self.micro_F1List.append(micro_F1)
        self.macro_F1List.append(macro_F1)
        epoch_end_time = time.time()  # 记录epoch结束时间
        epoch_duration = epoch_end_time - epoch_start_time  # 计算epoch时长
        print(f'Epoch {self.epoch_count + 1} completed, average loss: {average_loss}, duration: {epoch_duration:.2f} seconds')
        if self.epoch_count == 0:  # 第一个epoch
            # 初始化损失的变化
            delta_loss = 0
            self.count = 0
            self.curr_loss = average_loss
        else:
            # 计算损失的变化
            self.prev_loss = self.curr_loss  # 上一个epoch的损失
            self.curr_loss = average_loss  # 新epoch的损失
            delta_loss = abs(self.curr_loss - self.prev_loss)
            # 判断损失的变化是否小于阈值
            if delta_loss <= 0.03:
                # 次数加一
                self.count += 1
            else:
                # 次数归零
                self.count = 0
        # 返回损失的变化
        return delta_loss, self.count

    def get_a_train_batch(self, step):
        '''从训练数据集中获取一个批次的数据 '''
        min_index = step * self.batch_size
        max_index = min_index + self.batch_size
        if max_index > self.train_length:
            max_index = self.train_length
        return self.train_data[min_index:max_index]  # 返回min到max之间的数据样本

    def get_data_label(self, batch):
        data = np.delete(batch, -1, axis=1)
        label = np.array(batch, dtype=np.int32)[:, -1]
        return data, label

    def init_data(self):  # 此处的init_data和第一个DNN是不一样的，需要按照排序数组筛选
        self.train_data = []
        self.test_data = []  # 初始化训练和测试数据
        self.label_status = {}
        filename = TRAINDATA
        csv_reader = csv.reader(open(filename))
        label_data = {i: [] for i in range(29)}
        for row in csv_reader:
            data = []
            for i, char in enumerate(row):  # 在indices里面才加入data
                if i in self.top_k_indices or i == len(row) - 1:
                    if char == 'None':
                        data.append(0)
                    else:
                        data.append(np.float32(char))  # 将数据从字符串格式转换为 float32 格式
            label = int(data[-1])
            if label in label_data:
                label_data[label].append(data)
            self.label_status[str(label)] = self.label_status.get(str(label), 0) + 1
        self.train_data = [data for label in label_data for data in label_data[label]]
        filename = TESTDATA
        csv_reader = csv.reader(open(filename))
        for row in csv_reader:
            data = []
            for i, char in enumerate(row):
                if i in self.top_k_indices or i == len(row) - 1:
                    if char == 'None':
                        data.append(0)
                    else:
                        data.append(np.float32(char))  # 将数据从字符串格式转换为 float32 格式
            self.test_data.append(data)
        self.train_length = len(self.train_data)
        self.test_length = len(self.test_data)
        self.train_data = self.normalization(self.train_data)
        self.test_data = self.normalization(self.test_data)
         
        print('init data completed!')

    def normalization(self, minibatch):
        data = np.delete(minibatch, -1, axis=1)
        labels = np.array(minibatch, dtype=np.int64)[:, -1]
        mmax = np.max(data, axis=0)
        mmin = np.min(data, axis=0)
        for i in range(len(mmax)):
            if mmax[i] == mmin[i]:
                mmax[i] += 0.000001  # avoid getting devided by 0
        res = (data - mmin) / (mmax - mmin)
        res = np.c_[res, labels]
        return res

    def predict_top_k(self, x_feature, k=5):
        predict = self.sess.run(self.y, feed_dict={self.x_input: [x_feature]})[0]
        top_k_indices = np.argsort(predict)[-k:][::-1]  # 获取前k个预测标签的索引
        return top_k_indices

    def test(self):
        label_count = {}  # 标签的总数量
        label_correct = {}  # 标签预测正确的数量
        length = len(self.test_data)
        count = 0
        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        y_true = []
        y_pred = []

        for row in self.test_data:
            feature = row[0:-1]
            label = row[-1]
            count += 1
            x_feature = np.array(feature)
            top_k_labels = self.predict_top_k(x_feature, k=5)  # 获取前5个预测标签
            y_true.append(label)
            y_pred.append(top_k_labels[0])  # 记录TOP-1预测结果

            if str(int(label)) not in label_count:
                label_count[str(int(label))] = 0
                label_correct[str(int(label))] = 0
            if label == top_k_labels[0]:  # TOP-1预测  逐个判断top1-top5
                label_correct[str(int(label))] += 1
                top1_correct += 1
            if label in top_k_labels[:3]:  # TOP-3预测
                top3_correct += 1
            if label in top_k_labels[:5]:  # TOP-5预测
                top5_correct += 1
            label_count[str(int(label))] += 1
            if count % 10000 == 0:
                print(f"Processed {count} rows")

        accuracy1 = {}
        for label in sorted(label_count):
            accuracy1[label] = label_correct[label] / label_count[label]
            print(label, accuracy1[label], label_correct[label], label_count[label])

        top1_accuracy = top1_correct / length
        top3_accuracy = top3_correct / length
        top5_accuracy = top5_correct / length
        print(f"Top-1 accuracy: {top1_accuracy}")
        print(f"Top-3 accuracy: {top3_accuracy}")
        print(f"Top-5 accuracy: {top5_accuracy}")

        # 计算Macro-F1和Micro-F1
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        # print(f"Macro-F1: {macro_f1}")
        # print(f"Micro-F1: {micro_f1}")
        return top1_accuracy

    def test2(self): # 仅返回top1测试结果
        y_true = []
        y_pred = []
        for row in self.test_data:
            feature = row[0:-1]
            label = row[-1]
            x_feature = np.array(feature)
            top_k_labels = self.predict_top_k(x_feature, k=5)  # 获取前5个预测标签
            y_true.append(label)
            y_pred.append(top_k_labels[0])  # 记录TOP-1预测结果
        # 计算Macro-F1和Micro-F1
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        print(f"Macro-F1: {macro_f1}")
        print(f"Micro-F1: {micro_f1}")
        return macro_f1,micro_f1