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

DATA_DIM = 72
OUTPUT_DIM = 29  # 0-17类
BETA = 0.999 # 类平衡损失的β
GAMMA = 1

# Hyper Parameters
LEARNING_RATE = 0.0001
BATCH_SIZE = 128
TESTFILE = '../test_data.csv'
TRAINFILE = '../train_data.csv'

MODEL_SAVE_PATH = 'model/'
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

class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride=1, seed=None):
        super(BasicBlock, self).__init__()
        self.seed = seed
        self.conv1 = layers.Conv2D(filter_num, (3, 3), strides=stride, padding='same',
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')
        self.conv2 = layers.Conv2D(filter_num, (3, 3), strides=1, padding='same',
                                   kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.bn2 = layers.BatchNormalization()
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(layers.Conv2D(filter_num, (1, 1), strides=stride,
                                              kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed)))
        else:
            self.downsample = lambda x: x
    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        identity = self.downsample(inputs)
        output = layers.add([out, identity])
        output = tf.nn.relu(output)
        return output

class Resnet():
    def __init__(self,K,ES_THRESHOLD,seed):
        self.K = K
        self.ES_THRESHOLD = ES_THRESHOLD
        self.seed = seed
        self.maintainCnt = 0
        self.loss_history = []
        self.micro_F1List = []
        self.macro_F1List = []
        self.intersection_sets = []
        self.TSMRecord = []
        self.earlyStop = False
        print(f"BETA = {BETA}, GAMMA = {GAMMA}")
        self.learning_rate = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.epoch_count = 0
        self.x_input = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, DATA_DIM])
        self.target = tf.compat.v1.placeholder(dtype=tf.int32, shape=[None])
        self.train_step = tf.Variable(0, trainable=False)
        self.init_data()
        self.total_iterations_per_epoch = self.train_length // BATCH_SIZE
        self.conv_layer = layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding='same',
                                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.stm = Sequential([self.conv_layer,
                               layers.BatchNormalization(),
                               layers.Activation('relu'),
                               layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='same')])
        layer_dims = [2, 2, 2, 2]
        # 堆叠4个Block，每个block包含了多个BasicBlock,设置步长不一样
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)

        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(OUTPUT_DIM)
        print("create_ResNet start")
        self.create_ResNet()
        print("create_ResNet end")
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

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, stride, seed=self.seed))
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1, seed=self.seed))
        return res_blocks

    def create_ResNet(self):
        self.scaling_factor = tf.Variable(
            tf.constant(1, dtype=tf.float32, shape=[1, DATA_DIM]))
        self.scaling_factor_extended = tf.tile(self.scaling_factor, [BATCH_SIZE, 1])
        scaled_input = tf.multiply(self.x_input, self.scaling_factor_extended)
        scaled_input = tf.reshape(scaled_input, [BATCH_SIZE, DATA_DIM, 1, 1])
        x = self.stm(scaled_input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        self.y = self.fc(x)

    def build_loss(self):
        l1_regularizer = tf.keras.regularizers.l1(0.001)
        regularization_penalty = l1_regularizer(self.scaling_factor)
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y, labels=self.target)
        class_weights = tf.constant([self.weights[str(i)] for i in range(len(self.weights))], dtype=tf.float32)
        sample_weights = tf.gather(class_weights, self.target)
        ''' 类平衡损失 ce*类权重 
        cbce = tf.multiply(ce, sample_weights)
        self.loss = tf.reduce_sum(cbce)  # + regularization_penalty  # tf.reduce_mean()'''
        '''focal_loss+cb:'''
        # 计算Focal Loss的modulator
        softmax_probs = tf.nn.softmax(self.y)
        labels_one_hot = tf.one_hot(self.target, depth=len(self.weights))
        pt = tf.reduce_sum(labels_one_hot * softmax_probs, axis=1)
        modulator = tf.pow(1.0 - pt, GAMMA)
        focal_loss = modulator * ce
        cb_focal_loss = tf.multiply(focal_loss, sample_weights)
        self.loss = tf.reduce_sum(cb_focal_loss) + regularization_penalty
        self.train_op = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss,global_step=self.train_step)
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
        epoch_end_time = time.time()  # 记录epoch结束时间
        epoch_duration = epoch_end_time - epoch_start_time  #计算epoch时长
        print(f'Epoch {self.epoch_count + 1} completed, average loss: {average_loss}, duration: {epoch_duration:.2f} seconds')
        micro_F1, macro_F1 = self.test2()
        self.micro_F1List.append(micro_F1)
        self.macro_F1List.append(macro_F1)
        ''' 早停策略 每次都要'''
        keyFeatureNums = self.K
        values, indices = tf.math.top_k(self.scaling_factor, k=keyFeatureNums, sorted=True)
        max_indices = self.sess.run(indices)
        max_indices = max_indices[0]
        max_set = set(max_indices)  # 本次迭代得到的因子集合
        self.intersection_sets.append(max_set)  # 将当前epoch的关键特征下标集合添加到列表中
        if len(self.intersection_sets) > self.ES_THRESHOLD:  # 保持列表长度为ES_THRESHOLD
            self.intersection_sets.pop(0)
        if len(self.intersection_sets) >= self.ES_THRESHOLD:  # 仅当元素个数>=ES_THRESHOLD才判定早停
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
        self.test_data = []  # init train and test data
        self.label_status = {}
        filename = TRAINFILE
        csv_reader = csv.reader(open(filename))
        label0_data = label1_data = label2_data = label3_data = label4_data = label5_data = []
        label6_data = label7_data = label8_data = label9_data = label10_data = label11_data = []
        label12_data = label13_data = label14_data = label15_data = label16_data = label17_data = []
        label18_data = label19_data = label20_data = label21_data = label22_data = label23_data = []
        label24_data = label25_data = label26_data = label27_data = label28_data = []
        for row in csv_reader:
            data = []
            for char in row:
                if char == 'None':
                    data.append(0)
                else:
                    data.append(np.float32(char))  # transform data from format of string to float32 将数据从字符串格式转换为float32格式
            if data[-1] == 0:
                label0_data.append(data)
            if data[-1] == 1:
                label1_data.append(data)
            if data[-1] == 2:
                label2_data.append(data)
            if data[-1] == 3:
                label3_data.append(data)
            if data[-1] == 4:
                label4_data.append(data)
            if data[-1] == 5:
                label5_data.append(data)
            if data[-1] == 6:
                label6_data.append(data)
            if data[-1] == 7:
                label7_data.append(data)
            if data[-1] == 8:
                label8_data.append(data)
            if data[-1] == 9:
                label9_data.append(data)
            if data[-1] == 10:
                label10_data.append(data)
            if data[-1] == 11:
                label11_data.append(data)
            if data[-1] == 12:
                label12_data.append(data)
            if data[-1] == 13:
                label13_data.append(data)
            if data[-1] == 14:
                label14_data.append(data)
            if data[-1] == 15:
                label15_data.append(data)
            if data[-1] == 16:
                label16_data.append(data)
            if data[-1] == 17:
                label17_data.append(data)
            if data[-1] == 18:
                label18_data.append(data)
            if data[-1] == 19:
                label19_data.append(data)
            if data[-1] == 20:
                label20_data.append(data)
            if data[-1] == 21:
                label21_data.append(data)
            if data[-1] == 22:
                label22_data.append(data)
            if data[-1] == 23:
                label23_data.append(data)
            if data[-1] == 24:
                label24_data.append(data)
            if data[-1] == 25:
                label25_data.append(data)
            if data[-1] == 26:
                label26_data.append(data)
            if data[-1] == 27:
                label27_data.append(data)
            if data[-1] == 28:
                label28_data.append(data)
            if self.label_status.get(str(int(data[-1])), 0) > 0:
                self.label_status[str(int(data[-1]))] += 1
            else:
                self.label_status[str(int(data[-1]))] = 1
        self.train_data = label0_data + label1_data + label2_data + label3_data + label4_data + label5_data + \
                          label6_data + label7_data + label8_data + label9_data + label10_data + label11_data + \
                          label12_data + label13_data + label14_data + label15_data + label16_data + label17_data + \
                          label18_data + label19_data + label20_data + label21_data + label22_data + label23_data + \
                          label24_data + label25_data + label26_data + label27_data + label28_data

        filename = TESTFILE
        csv_reader = csv.reader(open(filename))
        for row in csv_reader:
            data = []
            for char in row:
                if char == 'None':
                    data.append(0)
                else:
                    data.append(np.float32(char))  # transform data from format of string to float32
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

class Resnet2(): # 第二次训练
    def __init__(self,dim,selected_features=[],seed=25):  # 需输入维度 即当前特征数
        self.dim = dim
        self.top_k_indices = selected_features
        self.seed = seed
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
        self.conv_layer = layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding='same',
                                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.stm = Sequential([self.conv_layer,
                               layers.BatchNormalization(),
                               layers.Activation('relu'),
                               layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='same')])
        layer_dims = [2, 2, 2, 2]
        # 堆叠4个Block，每个block包含了多个BasicBlock,设置步长不一样
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)
        # 通过Pooling层将高宽降低为1x1
        self.avgpool = layers.GlobalAveragePooling2D()
        # 最后连接一个全连接层分类
        self.fc = layers.Dense(OUTPUT_DIM)
        self.create_ResNet(dim)
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

    def create_ResNet(self, NEW_DIM):
        scaled_input = tf.reshape(self.x_input, [-1, NEW_DIM, 1, 1])
        x = self.stm(scaled_input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        self.y = self.fc(x)

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, stride, seed=self.seed))
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1, seed=self.seed))
        return res_blocks

    def build_loss(self):
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y, labels=self.target)
        class_weights = tf.constant([self.weights[str(i)] for i in range(len(self.weights))], dtype=tf.float32)
        sample_weights = tf.gather(class_weights, self.target)
        ''' 类平衡损失 ce*类权重 
        cbce = tf.multiply(ce, sample_weights)
        self.loss = tf.reduce_sum(cbce)  # + regularization_penalty  # tf.reduce_mean()'''
        '''focal_loss+cb:'''
        # 计算Focal Loss的modulator
        softmax_probs = tf.nn.softmax(self.y)
        labels_one_hot = tf.one_hot(self.target, depth=len(self.weights))
        pt = tf.reduce_sum(labels_one_hot * softmax_probs, axis=1)
        modulator = tf.pow(1.0 - pt, GAMMA)
        focal_loss = modulator * ce
        cb_focal_loss = tf.multiply(focal_loss, sample_weights)
        self.loss = tf.reduce_sum(cb_focal_loss)
        self.train_op = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss,global_step=self.train_step)
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
        micro_F1, macro_F1 = self.test2()
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

    def init_data(self): # 此处的init_data和第一个DNN是不一样的，需要按照排序数组筛选
        self.train_data = []
        self.test_data = []  # init train and test data
        self.label_status = {}
        filename = '../train_data.csv'
        csv_reader = csv.reader(open(filename))
        label0_data = label1_data = label2_data = label3_data = label4_data = label5_data = []
        label6_data = label7_data = label8_data = label9_data = label10_data = label11_data = []
        label12_data = label13_data = label14_data = label15_data = label16_data = label17_data = []
        label18_data = label19_data = label20_data = label21_data = label22_data = label23_data = []
        label24_data = label25_data = label26_data = label27_data = label28_data = []
        for row in csv_reader:
            data = []
            for i, char in enumerate(row): # 在indices里面才加入data
                if i in self.top_k_indices or i == len(row) - 1:
                    if char == 'None':
                        data.append(0)
                    else:
                        data.append(np.float32(char))  # 将数据从字符串格式转换为 float32 格式
            if data[-1] == 0:
                label0_data.append(data)
            if data[-1] == 1:
                label1_data.append(data)
            if data[-1] == 2:
                label2_data.append(data)
            if data[-1] == 3:
                label3_data.append(data)
            if data[-1] == 4:
                label4_data.append(data)
            if data[-1] == 5:
                label5_data.append(data)  #
            if data[-1] == 6:
                label6_data.append(data)  #
            if data[-1] == 7:
                label7_data.append(data)  # 没有标签是7的 因为只捕获到1条数据
            if data[-1] == 8:
                label8_data.append(data)
            if data[-1] == 9:
                label9_data.append(data)
            if data[-1] == 10:
                label10_data.append(data)
            if data[-1] == 11:
                label11_data.append(data)
            if data[-1] == 12:
                label12_data.append(data)
            if data[-1] == 13:
                label13_data.append(data)
            if data[-1] == 14:
                label14_data.append(data)
            if data[-1] == 15:
                label15_data.append(data)
            if data[-1] == 16:
                label16_data.append(data)
            if data[-1] == 17:
                label17_data.append(data)
            if data[-1] == 18:
                label18_data.append(data)
            if data[-1] == 19:
                label19_data.append(data)
            if data[-1] == 20:
                label20_data.append(data)
            if data[-1] == 21:
                label21_data.append(data)
            if data[-1] == 22:
                label22_data.append(data)
            if data[-1] == 23:
                label23_data.append(data)
            if data[-1] == 24:
                label24_data.append(data)
            if data[-1] == 25:
                label25_data.append(data)
            if data[-1] == 26:
                label26_data.append(data)
            if data[-1] == 27:
                label27_data.append(data)
            if data[-1] == 28:
                label28_data.append(data)
            if self.label_status.get(str(int(data[-1])), 0) > 0:
                self.label_status[str(int(data[-1]))] += 1
            else:
                self.label_status[str(int(data[-1]))] = 1
        self.train_data = label0_data + label1_data + label2_data + label3_data + label4_data + label5_data + \
                          label6_data + label7_data + label8_data + label9_data + label10_data + label11_data + \
                          label12_data + label13_data + label14_data + label15_data + label16_data + label17_data + \
                          label18_data + label19_data + label20_data + label21_data + label22_data + label23_data + \
                          label24_data + label25_data + label26_data + label27_data + label28_data
        filename = '../test_data.csv'
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