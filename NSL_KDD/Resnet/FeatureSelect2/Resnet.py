# 第一次训练，确定选择因子
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
import numpy as np
import csv, random
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, GlobalMaxPool2D, Reshape, Multiply, Lambda

tf.compat.v1.disable_eager_execution()

DATA_DIM = 41
OUTPUT_DIM = 2
LOSSTYPE = "cb"
BETA = 0.9999 # 类平衡损失的β gamma 使用cb 因为cb效果最好
GAMMA = 1

LEARNING_RATE = 0.0001
BATCH_SIZE = 128
TRAIN_FILE = '../../train_data.csv'
TEST_FILE = '../../test_data.csv'

MODEL_SAVE_PATH = './model/'
MODEL_SAVE_PATH2 = './model2/'
MODEL_NAME = 'model'
MODEL_NAME2 = 'model2'
top_k_values=[]
top_k_indice=[]

def set_seed(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

class BasicBlock(layers.Layer):
    def __init__(self, filter_num, stride, seed):
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
        self.lossType = LOSSTYPE
        set_seed(seed)
        self.maintainCnt = 0
        # self.scaling_factors_history = []
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
        self.create_ResNet()
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
            self.loss = tf.reduce_sum(ce) + regularization_penalty
        elif (self.lossType == "cb"):
            self.loss = tf.reduce_sum(cbce) + regularization_penalty
        elif (self.lossType == "cb_focal_loss"):
            self.loss = tf.reduce_sum(cb_focal_loss) + regularization_penalty
        self.train_op = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss,global_step=self.train_step)
        self.sess.run(tf.compat.v1.global_variables_initializer())

    def train(self):
        total_loss = 0
        num_batches = 0
        self.train_start = 0
        epoch_start_time = time.time()  # 记录epoch开始时间
        self.best_micro_f1 = -1
        self.best_model = None
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
        epoch_duration = epoch_end_time - epoch_start_time  # 计算epoch时长
        print(f'Epoch {self.epoch_count + 1} completed, average loss: {average_loss}, duration: {epoch_duration:.2f} seconds')
        macro_F1, micro_F1 = self.test2()
        self.macro_F1List.append(macro_F1)
        self.micro_F1List.append(micro_F1)
        ''' 早停策略 每次都要 '''
        keyFeatureNums = self.K
        values, indices = tf.math.top_k(self.scaling_factor, k=keyFeatureNums, sorted=True)
        max_indices = self.sess.run(indices)
        max_indices = max_indices[0]
        max_set = set(max_indices)  # 本次迭代得到的因子集合
        self.intersection_sets.append(max_set)  # 将当前epoch的关键特征下标集合添加到列表中
        if len(self.intersection_sets) > self.ES_THRESHOLD:  # 保持列表长度为ES_THRESHOLD
            self.intersection_sets.pop(0)
        if len(self.intersection_sets) > 0:  # 仅当元素个数>=ES_THRESHOLD才判定早停
            intersection = set.intersection(*self.intersection_sets)
            print(f"Epoch {self.epoch_count + 1}, Intersection size: {len(intersection)}")
            if len(intersection) == self.K and len(self.intersection_sets) == self.ES_THRESHOLD:
                self.earlyStop = True
            self.TSMRecord.append(len(intersection))

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
            if delta_loss <= 0.03:
                self.count += 1
            else:
                self.count = 0
        return delta_loss, self.count, micro_F1

    def get_a_train_batch(self, step):
        min_index = step * self.batch_size
        max_index = min_index + self.batch_size
        if max_index > self.train_length:
            max_index = self.train_length
        return self.train_data[min_index:max_index]  # 返回min到max之间的数据样本

    def get_data_label(self, batch):
        data = np.delete(batch, -1, axis=1) # 从输入的批次数据中删除最后一列，以提取仅包含特征的数据
        label = np.array(batch, dtype=np.int32)[:, -1]
        return data, label

    def init_data(self):
        self.train_data = []
        self.test_data = []
        self.valid_data = []
        self.label_status = {}
        filename = TRAIN_FILE
        csv_reader = csv.reader(open(filename))
        label_data = [[] for _ in range(OUTPUT_DIM)]
        for row in csv_reader:
            data = []
            for char in row:
                if char == 'None':
                    data.append(0)
                else:
                    data.append(np.float32(char))  # 将数据从字符串格式转换为float32格式
            label = int(data[-1])
            if label < OUTPUT_DIM:
                label_data[label].append(data)
            if self.label_status.get(str(label), 0) > 0:
                self.label_status[str(label)] += 1
            else:
                self.label_status[str(label)] = 1
        self.train_data = sum(label_data, [])
        filename = TEST_FILE
        csv_reader = csv.reader(open(filename))
        label_data = [[] for _ in range(OUTPUT_DIM)]
        for row in csv_reader:
            data = []
            for char in row:
                if char == 'None':
                    data.append(0)
                else:
                    data.append(np.float32(char))  # 将数据从字符串格式转换为float32格式
            label = int(data[-1])
            if label < OUTPUT_DIM:
                label_data[label].append(data)
        for label in range(OUTPUT_DIM):
            num_samples = len(label_data[label])
            num_valid = max(1, num_samples // 3)
            self.valid_data.extend(label_data[label][:num_valid])
            self.test_data.extend(label_data[label][num_valid:])
        self.train_data = self.normalization(self.train_data)
        self.valid_data = self.normalization(self.valid_data)
        self.test_data = self.normalization(self.test_data)
        np.random.shuffle(self.train_data)
        self.train_length = len(self.train_data)
        self.valid_length = len(self.valid_data)
        self.test_length = len(self.test_data)
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

    def get_a_test_batch(self, step):
        '''从测试数据集中获取一个批次的数据 '''
        min_index = step * self.batch_size
        max_index = min_index + self.batch_size
        if max_index > self.test_length:
            max_index = self.test_length
        return self.test_data[min_index:max_index]  # 返回min到max之间的数据样本

    def get_a_valid_batch(self, step):
        '''从测试数据集中获取一个批次的数据 '''
        min_index = step * self.batch_size
        max_index = min_index + self.batch_size
        if max_index > self.test_length:
            max_index = self.test_length
        return self.valid_data[min_index:max_index]  # 返回min到max之间的数据样本

    def predict_top_k(self, x_features, k=5):
        # 使用批量输入进行预测
        predicts = self.sess.run(self.y, feed_dict={self.x_input: x_features})
        top_k_indices = [np.argsort(predict)[-k:][::-1] for predict in predicts]
        return top_k_indices

    def test(self):
        from sklearn.metrics import f1_score, precision_score, recall_score
        label_count = {}  # 标签的总数量
        label_correct = {}  # 标签预测正确的数量
        top1_correct = 0
        top3_correct = 0
        top5_correct = 0
        y_true = []
        y_pred = []
        num_batches = self.test_length // self.batch_size
        for step in range(num_batches):
            batch = self.get_a_test_batch(step)
            data, labels = self.get_data_label(batch)
            top_k_labels = self.predict_top_k(data, k=5)  # 获取前5个预测标签
            y_true.extend(labels)
            y_pred.extend([labels[0] for labels in top_k_labels])  # 记录TOP-1预测结果
            for label, top_k in zip(labels, top_k_labels):
                label_str = str(int(label))
                if label_str not in label_count:
                    label_count[label_str] = 0
                    label_correct[label_str] = 0
                if label == top_k[0]:  # TOP-1预测 逐个判断top1-top5
                    label_correct[label_str] += 1
                    top1_correct += 1
                if label in top_k[:3]:  # TOP-3预测
                    top3_correct += 1
                if label in top_k[:5]:  # TOP-5预测
                    top5_correct += 1
                label_count[label_str] += 1
        accuracy1 = {}
        for label in sorted(label_count):
            accuracy1[label] = label_correct[label] / label_count[label]
            print(label, accuracy1[label], label_correct[label], label_count[label])
        top1_accuracy = top1_correct / self.test_length
        top3_accuracy = top3_correct / self.test_length
        top5_accuracy = top5_correct / self.test_length
        print(f"Top-1 accuracy: {top1_accuracy}")
        print(f"Top-3 accuracy: {top3_accuracy}")
        print(f"Top-5 accuracy: {top5_accuracy}")
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        print(f"Micro-F1: {micro_f1}")
        print(f"Macro-F1: {macro_f1}")
        print(f"precision: {precision}")
        print(f"recall: {recall}")
        return top1_accuracy

    def test2(self):  # 设置默认的批次大小
        from sklearn.metrics import f1_score
        y_true = []
        y_pred = []
        num_batches = self.valid_length // self.batch_size
        for step in range(num_batches):
            batch = self.get_a_valid_batch(step)
            data, label = self.get_data_label(batch)
            top_k_labels = self.predict_top_k(data, k=5)  # 获取前5个预测标签
            y_true.extend(label)
            y_pred.extend([labels[0] for labels in top_k_labels])  # 记录TOP-1预测结果
        # 计算Macro-F1和Micro-F1
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        print(f"Macro-F1: {macro_f1}")
        print(f"Micro-F1: {micro_f1}")
        return macro_f1, micro_f1
