# TensorFlow 2.9.0 compatible ResNet implementation for UNSW-IoT FeatureSelect
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from sklearn.metrics import f1_score
import numpy as np
import csv, random

# 默认保持图执行（graph mode），比强制 eager 更快
tf.config.run_functions_eagerly(False)

# 设置随机种子确保结果可复现
def set_deterministic_seed(seed):
    """设置所有随机种子确保结果可复现"""
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

OUTPUT_DIM = 20  # 开放1是20 开放2是25类
TRAINDATA = '../../OWtrain_data1.csv'
TESTDATA = '../../OWtest_data1.csv'
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

class ResNetModel2(Model):
    def __init__(self, dim, selected_features=[], seed=25, fixed_scaling=None):
        super(ResNetModel2, self).__init__()
        self.dim = dim
        self.selected_features = selected_features
        self.seed = seed

        # scaling factor：可训练或冻结
        if fixed_scaling is not None:
            self.scaling_factor = tf.constant(fixed_scaling, dtype=tf.float32, name='scaling_factor_fixed')
        else:
            self.scaling_factor = tf.Variable(tf.ones([1, dim], dtype=tf.float32), trainable=True, name='scaling_factor')
        
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
        batch_size = tf.shape(inputs)[0]
        scaling_factor_extended = tf.tile(self.scaling_factor, [batch_size, 1])
        scaled_input = tf.multiply(inputs, scaling_factor_extended)
        scaled_input = tf.reshape(scaled_input, [batch_size, self.dim, 1, 1])
        x = self.stm(scaled_input, training=training)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        return self.fc(x)

class Resnet2():
    def __init__(self,dim,selected_features=[],seed=25, fixed_scaling=None):
        self.dim = dim
        self.top_k_indices = selected_features
        self.seed = seed
        self.learning_rate = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.epoch_count = 0
        self.loss_history = []
        self.micro_F1List = []
        self.macro_F1List = []
        self.eval_batch_size = max(BATCH_SIZE, 512)
        
        self.init_data()
        # ceil 取整，避免丢弃最后不足一个 batch 的数据
        self.total_iterations_per_epoch = max(1, (self.train_length + BATCH_SIZE - 1) // BATCH_SIZE)
        
        # Create the model
        self.model = ResNetModel2(dim=len(selected_features), selected_features=selected_features, seed=self.seed, fixed_scaling=fixed_scaling)
        
        # Setup optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Calculate class weights for CB loss
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
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
        class_weights = tf.constant([self.weights[str(i)] for i in range(len(self.weights))], dtype=tf.float32)
        sample_weights = tf.gather(class_weights, y_true)

        # 计算Focal Loss的modulator
        softmax_probs = tf.nn.softmax(y_pred)
        labels_one_hot = tf.one_hot(y_true, depth=len(self.weights))
        pt = tf.reduce_sum(labels_one_hot * softmax_probs, axis=1)
        modulator = tf.pow(1.0 - pt, GAMMA)
        focal_loss = modulator * ce
        cb_focal_loss = tf.multiply(focal_loss, sample_weights)

        return tf.reduce_sum(cb_focal_loss)

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
        # micro_F1, macro_F1 = self.test2()
        # self.micro_F1List.append(micro_F1)
        # self.macro_F1List.append(macro_F1)
        #
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch {self.epoch_count + 1} completed, average loss: {average_loss:.6f}, duration: {epoch_duration:.2f} seconds')

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
            data = [0 if char in ('None', '') else np.float32(char) for char in row]
            label = int(data[-1])
            label_data[label].append(data)
            self.label_status[str(label)] = self.label_status.get(str(label), 0) + 1
        
        with open(TRAINDATA, mode="r", encoding="utf-8") as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                process_row(row)
        
        self.train_data = [data for label in label_data.values() for data in label]

        with open(TESTDATA, mode="r", encoding="utf-8") as f:
            csv_reader = csv.reader(f)
            self.test_data = [
                [0 if char in ('None', '') else np.float32(char) for char in row]
                for row in csv_reader
            ]

        self.train_length = len(self.train_data)
        self.test_length = len(self.test_data)
        self.train_data = self.normalization(self.train_data)
        self.test_data = self.normalization(self.test_data)
        # 转成 numpy 数组，便于后续切片与向量化
        self.train_data = np.array(self.train_data, dtype=np.float32)
        self.test_data = np.array(self.test_data, dtype=np.float32)
        self.test_features, self.test_labels = self._prepare_test_arrays()
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
        logits = self.model(x_tensor, training=False)[0]
        _, top_k_indices = tf.nn.top_k(logits, k=k)
        return top_k_indices.numpy()

    def predict_top_k_batch(self, features, k=5):
        """按 batch 进行推理，返回 shape=(N, k) 的 top-k 索引，显著加速评估。"""
        topk_list = []
        total = features.shape[0]
        for start in range(0, total, self.eval_batch_size):
            end = min(start + self.eval_batch_size, total)
            batch = tf.constant(features[start:end], dtype=tf.float32)
            logits = self.model(batch, training=False)
            _, topk = tf.nn.top_k(logits, k=k)
            topk_list.append(topk.numpy())
        return np.vstack(topk_list)

    def _prepare_test_arrays(self):
        """把测试集拆成 features 和 labels 数组，便于向量化计算。"""
        test_np = np.array(self.test_data)
        features = test_np[:, :-1].astype(np.float32)
        labels = test_np[:, -1].astype(np.int32)
        return features, labels

    def test(self):
        features, labels = self._prepare_test_arrays()
        topk_all = self.predict_top_k_batch(features, k=5)

        preds = topk_all[:, 0]
        length = len(labels)

        # top-k 命中统计（向量化）
        top1_hits = (preds == labels)
        top3_hits = (topk_all[:, :3] == labels[:, None]).any(axis=1)
        top5_hits = (topk_all[:, :5] == labels[:, None]).any(axis=1)

        top1_correct = top1_hits.sum()
        top3_correct = top3_hits.sum()
        top5_correct = top5_hits.sum()

        # 按标签的准确率统计
        label_count = {}
        label_correct = {}
        for y, hit in zip(labels, top1_hits):
            key = str(int(y))
            label_count[key] = label_count.get(key, 0) + 1
            if hit:
                label_correct[key] = label_correct.get(key, 0) + 1

        accuracy1 = {}
        for key in sorted(label_count):
            correct = label_correct.get(key, 0)
            cnt = label_count[key]
            accuracy1[key] = correct / cnt if cnt else 0.0
            print(key, accuracy1[key], correct, cnt)

        top1_accuracy = top1_correct / length
        top3_accuracy = top3_correct / length
        top5_accuracy = top5_correct / length
        print(f"Top-1 accuracy: {top1_accuracy}")
        print(f"Top-3 accuracy: {top3_accuracy}")
        print(f"Top-5 accuracy: {top5_accuracy}")

        macro_f1 = f1_score(labels, preds, average='macro')
        micro_f1 = f1_score(labels, preds, average='micro')
        print(f"Macro-F1: {macro_f1}")
        print(f"Micro-F1: {micro_f1}")
        return top1_accuracy
    
    def test2(self):
        features, labels = self._prepare_test_arrays()
        topk_all = self.predict_top_k_batch(features, k=5)
        preds = topk_all[:, 0]

        macro_f1 = f1_score(labels, preds, average='macro')
        micro_f1 = f1_score(labels, preds, average='micro')
        print(f"Macro-F1: {macro_f1}")
        print(f"Micro-F1: {micro_f1}")
        return macro_f1, micro_f1
