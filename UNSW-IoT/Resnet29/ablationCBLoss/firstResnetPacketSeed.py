# TensorFlow 2.9.0 compatible ResNet implementation for UNSW-IoT FeatureSelect
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from sklearn.metrics import f1_score
import numpy as np
import csv, random

# Disable eager execution for better performance
tf.config.run_functions_eagerly(False)

# 设置随机种子确保结果可复现
def set_deterministic_seed(seed):
    """设置所有随机种子确保结果可复现"""
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

TRAIN_FILE = '../../OWtrain_data2.csv'  #封闭世界 数据集 25个设备
TEST_FILE = '../../OWtest_data2.csv'
DATA_DIM = 72
OUTPUT_DIM = 25  # 0-25类
BETA = 0.9999 # 类平衡损失的β
GAMMA = 1
LossType = "cb_focal_loss"

# Hyper Parameters
LEARNING_RATE = 0.0001
BATCH_SIZE = 128

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
    64, 32, 32, 32, 32,  # dpl_total, dpl_mean, dpl_min, dpl_max, dwin_std
    32, 32, 32, 32,         # bfpl_rate, fpl_s, bpl_s, dpl_s  19
    16, 16, 16, 16, 16, 16, 16, 16,  # fin_cnt, syn_cnt, rst_cnt, pst_cnt, ack_cnt, urg_cnt, cwe_cnt, ece_cnt
    16, 16, 16, 16,     # fwd_pst_cnt, fwd_urg_cnt, bwd_pst_cnt, bwd_urg_cnt
    16, 16, 16,         # fp_hdr_len, bp_hdr_len, dp_hdr_len
    32, 32, 32          # f_ht_len, b_ht_len, d_ht_len 18
]

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
        out = self.bn1(out, training=training)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)
        identity = self.downsample(inputs)
        output = layers.add([out, identity])
        output = tf.nn.relu(output)
        return output

class ResNetModel(Model):
    def __init__(self, K, ES_THRESHOLD, seed, fixed_scaling=None):
        super(ResNetModel, self).__init__()
        self.K = K
        self.ES_THRESHOLD = ES_THRESHOLD
        self.seed = seed

        # Feature scaling factor for feature selection
        if fixed_scaling is not None:
            # 冻结的缩放因子
            self.scaling_factor = tf.constant(fixed_scaling, dtype=tf.float32, name='scaling_factor_fixed')
            self.scaling_trainable = False
        else:
            self.scaling_factor = tf.Variable(
                tf.constant(1, dtype=tf.float32, shape=[1, DATA_DIM]),
                trainable=True,
                name='scaling_factor'
            )
            self.scaling_trainable = True

        # Build the ResNet architecture
        self.conv_layer = layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding='same',
                                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.stm = Sequential([
            self.conv_layer,
            layers.BatchNormalization(),
            layers.Activation('relu'),
            layers.MaxPool2D(pool_size=(2, 2), strides=1, padding='same')
        ])

        layer_dims = [2, 2, 2, 2]
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(OUTPUT_DIM)

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, stride, seed=self.seed))
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1, seed=self.seed))
        return res_blocks

    def call(self, inputs, training=None):
        # Apply scaling factor
        batch_size = tf.shape(inputs)[0]
        scaling_factor_extended = tf.tile(self.scaling_factor, [batch_size, 1])
        scaled_input = tf.multiply(inputs, scaling_factor_extended)
        scaled_input = tf.reshape(scaled_input, [batch_size, DATA_DIM, 1, 1])

        x = self.stm(scaled_input, training=training)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        return self.fc(x)


class Resnet():
    def __init__(self, K, ES_THRESHOLD, seed, fixed_scaling=None, loss_type=LossType):
        self.K = K
        self.ES_THRESHOLD = ES_THRESHOLD
        self.seed = seed
        self.maintainCnt = 0
        self.loss_history = []
        self.lossType = loss_type
        self.gamma = GAMMA
        self.micro_F1List = []
        self.macro_F1List = []
        self.intersection_sets = []
        self.TSMRecord = []
        self.earlyStop = False
        print(f"BETA = {BETA}, GAMMA = {GAMMA}")
        self.learning_rate = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.epoch_count = 0
        self.eval_batch_size = max(BATCH_SIZE, 512)

        self.init_data()
        self.total_iterations_per_epoch = max(1, (self.train_length + BATCH_SIZE - 1) // BATCH_SIZE)

        # Create the model
        self.model = ResNetModel(K=K, ES_THRESHOLD=ES_THRESHOLD, seed=self.seed, fixed_scaling=fixed_scaling)

        # Setup optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Calculate class weights for CB loss when enabled
        self.weights = None
        if self.lossType in ("cb", "cb_focal_loss"):
            beta = BETA
            ClassNum = len(self.label_status)
            effective_num = {}
            for key, value in self.label_status.items():
                new_value = (1.0 - beta) / (1.0 - np.power(beta, value))
                effective_num[key] = new_value

            total_effective_num = sum(effective_num.values())
            self.weights = {}
            for key, value in effective_num.items():
                new_value = effective_num[key] / total_effective_num * ClassNum
                self.weights[key] = new_value

    def compute_loss(self, y_true, y_pred):
        l1_regularizer = tf.keras.regularizers.l1(0.001)
        regularization_penalty = 0.0
        # 仅当缩放因子可训练时加入正则
        if isinstance(self.model.scaling_factor, tf.Variable) and self.model.scaling_factor.trainable:
            regularization_penalty = l1_regularizer(self.model.scaling_factor)

        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)

        class_weights = None
        sample_weights = None
        if self.weights is not None:
            class_weights = tf.constant([self.weights[str(i)] for i in range(len(self.weights))], dtype=tf.float32)
            sample_weights = tf.gather(class_weights, y_true)

        # 计算Focal Loss的modulator
        softmax_probs = tf.nn.softmax(y_pred)
        labels_one_hot = tf.one_hot(y_true, depth=len(self.weights))
        pt = tf.reduce_sum(labels_one_hot * softmax_probs, axis=1)
        modulator = tf.pow(1.0 - pt, self.gamma)
        focal_loss = modulator * ce
        cb_focal_loss = tf.multiply(focal_loss, sample_weights) if sample_weights is not None else None

        if self.lossType == "ce":
            return tf.reduce_sum(ce) + regularization_penalty
        elif self.lossType == "cb":
            return tf.reduce_sum(tf.multiply(ce, sample_weights)) + regularization_penalty
        elif self.lossType == "cb_focal_loss":
            return tf.reduce_sum(cb_focal_loss) + regularization_penalty

    @tf.function
    def train_step(self, data, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(data, training=True)
            loss = self.compute_loss(labels, predictions)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return loss

    def train(self):
        total_loss = 0
        num_batches = 0
        epoch_start_time = time.time()

        # 随机打乱训练数据
        np.random.shuffle(self.train_data)

        for step in range(self.total_iterations_per_epoch):
            batch = self.get_a_train_batch(step)
            data, label = self.get_data_label(batch)

            loss = self.train_step(data, label)
            total_loss += loss.numpy()
            num_batches += 1

        average_loss = total_loss / num_batches
        self.loss_history.append(average_loss)

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(
            f'Epoch {self.epoch_count + 1} completed, average loss: {average_loss:.6f}, duration: {epoch_duration:.2f} seconds')

        ''' 早停策略 每次都要'''
        keyFeatureNums = self.K
        scaling_factor_value = self.model.scaling_factor.numpy()
        max_indices = np.argsort(scaling_factor_value.flatten())[::-1][:keyFeatureNums]
        max_set = set(max_indices)
        self.intersection_sets.append(max_set)

        if len(self.intersection_sets) > self.ES_THRESHOLD:
            self.intersection_sets.pop(0)
        if len(self.intersection_sets) >= self.ES_THRESHOLD:
            intersection = set.intersection(*self.intersection_sets)
            print(f"Epoch {self.epoch_count + 1}, Intersection size: {len(intersection)}")
            self.TSMRecord.append(len(intersection))
            if len(intersection) == keyFeatureNums:
                print("Early stopping condition met.")
                self.earlyStop = True

        if self.epoch_count == 0:
            delta_loss = 0
            self.count = 0
            self.curr_loss = average_loss
        else:
            self.prev_loss = self.curr_loss
            self.curr_loss = average_loss
            delta_loss = abs(self.curr_loss - self.prev_loss)
            if delta_loss <= 0.03:
                self.count += 1
            else:
                self.count = 0

        return delta_loss, self.count

    def get_a_train_batch(self, step):
        min_index = step * self.batch_size
        max_index = min_index + self.batch_size
        if max_index > self.train_length:
            max_index = self.train_length
        return self.train_data[min_index:max_index]

    def get_data_label(self, batch):
        data = np.delete(batch, -1, axis=1)
        label = np.array(batch, dtype=np.int32)[:, -1]
        return data, label

    def init_data(self):
        self.train_data = []
        self.test_data = []
        self.label_status = {}
        label_data = {i: [] for i in range(29)}

        def process_row(row):
            data = [0 if char in('None','') else np.float32(char) for char in row]
            label = int(data[-1])
            label_data[label].append(data)
            self.label_status[str(label)] = self.label_status.get(str(label), 0) + 1

        with open(TRAIN_FILE, mode="r", encoding="utf-8") as f:
            csv_reader = csv.reader(f)
            next(csv_reader)  # Skip header if exists
            for row in csv_reader:
                process_row(row)

        self.train_data = [data for label in label_data.values() for data in label]

        with open(TEST_FILE, mode="r", encoding="utf-8") as f:
            csv_reader = csv.reader(f)
            self.test_data = [
                [0 if char in('None','')  else np.float32(char) for char in row]
                for row in csv_reader
            ]

        self.train_data = self.normalization(self.train_data)
        self.test_data = self.normalization(self.test_data)
        self.train_data = np.array(self.train_data, dtype=np.float32)
        self.test_data = np.array(self.test_data, dtype=np.float32)
        self.test_features, self.test_labels = self._prepare_test_arrays(self.test_data)
        self.train_length = len(self.train_data)
        self.test_length = len(self.test_data)
        np.random.shuffle(self.train_data)
        print('init data completed!')

    def normalization(self, minibatch):
        data = np.delete(minibatch, -1, axis=1)
        labels = np.array(minibatch, dtype=np.int64)[:, -1]
        mmax = np.max(data, axis=0)
        mmin = np.min(data, axis=0)
        for i in range(len(mmax)):
            if mmax[i] == mmin[i]:
                mmax[i] += 0.000001
        res = (data - mmin) / (mmax - mmin)
        res = np.c_[res, labels]
        return res

    def predict_top_k(self, x_feature, k=5):
        x_tensor = tf.constant([x_feature], dtype=tf.float32)
        predict = self.model(x_tensor, training=False)[0]
        top_k_indices = np.argsort(predict.numpy())[-k:][::-1]
        return top_k_indices
    
    def predict_top_k_batch(self, features, k=5):
        x_tensor = tf.constant(features, dtype=tf.float32)
        logits = self.model(x_tensor, training=False)
        _, top_k_indices = tf.nn.top_k(logits, k=k)
        return top_k_indices.numpy()

    def _prepare_test_arrays(self, data_array):
        features = data_array[:, :-1]
        labels = data_array[:, -1].astype(np.int32)
        return features, labels

    def test2(self):
        num_samples = self.test_features.shape[0]
        top1_preds = []
        for start in range(0, num_samples, self.eval_batch_size):
            end = min(start + self.eval_batch_size, num_samples)
            batch_feats = self.test_features[start:end]
            topk = self.predict_top_k_batch(batch_feats, k=5)
            top1_preds.extend(topk[:, 0])

        y_true = self.test_labels
        y_pred = np.array(top1_preds, dtype=np.int32)

        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        print(f"Macro-F1: {macro_f1}")
        print(f"Micro-F1: {micro_f1}")
        return macro_f1, micro_f1
