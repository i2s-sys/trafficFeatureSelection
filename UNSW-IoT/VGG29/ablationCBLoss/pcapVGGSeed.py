# TensorFlow 2.9.0 compatible VGG implementation for UNSW-IoT FeatureSelect
import gc
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential, Model
from sklearn.metrics import f1_score
import numpy as np
import csv, random

# Disable eager for better performance (graph mode)
tf.config.run_functions_eagerly(False)

# 设置随机种子确保结果可复现
def set_deterministic_seed(seed):
    """设置所有随机种子确保结果可复现"""
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

DATA_DIM = 72 # 特征
OUTPUT_DIM = 25
LEARNING_RATE = 0.0001
BETA = 0.999
GAMMA = 1
BATCH_SIZE = 128
TRAIN_FILE = '../../OWtrain_data2.csv'  #封闭世界 数据集 25个设备
TEST_FILE = '../../OWtest_data2.csv'

MODEL_SAVE_PATH = './model/'
MODEL_SAVE_PATH2 = './model2/'
MODEL_NAME = 'model'
MODEL_NAME2 = 'model2'
top_k_values=[]
top_k_indice=[]

class VGGModel(Model):
    def __init__(self, K, lossType, ES_THRESHOLD, seed):
        super(VGGModel, self).__init__()
        self.K = K
        self.lossType = lossType
        self.ES_THRESHOLD = ES_THRESHOLD
        self.seed = seed
        
        # Feature scaling factor for feature selection
        self.scaling_factor = tf.Variable(
            tf.constant(1, dtype=tf.float32, shape=[1, DATA_DIM]), 
            trainable=True,
            name='scaling_factor'
        )
        
        # Build VGG architecture
        self.conv1_1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                    name='conv_1',
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.conv1_2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                    name='conv_2',
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.pool1 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool_1')
        
        self.conv2_1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                    name='conv_3',
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.conv2_2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                    name='conv_4',
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.pool2 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool_2')
        
        self.conv3_1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                                    name='conv_5',
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.conv3_2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                                    name='conv_6',
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.conv3_3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                                    name='conv_7',
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.pool3 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool_3')
        
        self.conv4_1 = layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                                    name='conv_8',
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(1024, activation='relu', name='fc_1',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.dropout1 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(1024, activation='relu', name='fc_2',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.dropout2 = layers.Dropout(0.5)
        self.fc3 = layers.Dense(OUTPUT_DIM, name='fc_3',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
    
    def call(self, inputs, training=None):
        # Apply scaling factor
        batch_size = tf.shape(inputs)[0]
        scaling_factor_extended = tf.tile(self.scaling_factor, [batch_size, 1])
        scaled_input = tf.multiply(inputs, scaling_factor_extended)
        scaled_input = tf.reshape(scaled_input, [batch_size, DATA_DIM, 1, 1])
        
        # VGG forward pass
        x = self.conv1_1(scaled_input)
        x = self.conv1_2(x)
        x = self.pool1(x)
        
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)
        
        x = self.conv4_1(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x, training=training)
        x = self.fc2(x)
        x = self.dropout2(x, training=training)
        return self.fc3(x)

class VGG():
    def __init__(self, K, lossType, ES_THRESHOLD, seed):  # 不输入维度 默认是DATA_DIM
        self.ES_THRESHOLD = ES_THRESHOLD
        set_deterministic_seed(seed)
        self.K = K
        self.lossType = lossType
        self.seed = seed
        self.gamma = GAMMA
        self.learning_rate = LEARNING_RATE
        self.batch_size = BATCH_SIZE
        self.epoch_count = 0
        self.loss_history = []
        self.micro_F1List = []
        self.macro_F1List = []
        self.intersection_sets = []
        self.TSMRecord = []
        self.earlyStop = False
        self.init_data()
        self.total_iterations_per_epoch = max(1, (self.train_length + BATCH_SIZE - 1) // BATCH_SIZE)
        self.eval_batch_size = max(BATCH_SIZE, 512)
        
        # Create the model
        self.model = VGGModel(K=K, lossType=lossType, ES_THRESHOLD=ES_THRESHOLD, seed=self.seed)
        
        # Setup optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        # Calculate class weights for CB loss when enabled
        self.weights = None
        if self.lossType in ("cb", "cb_focal_loss"):
            ClassNum = len(self.label_status)
            effective_num = {}
            for key, value in self.label_status.items():
                new_value = (1.0 - BETA) / (1.0 - np.power(BETA, value))
                effective_num[key] = new_value

            total_effective_num = sum(effective_num.values())
            self.weights = {}
            for key, value in effective_num.items():
                new_value = effective_num[key] / total_effective_num * ClassNum
                self.weights[key] = new_value

    def compute_loss(self, y_true, y_pred):
        l1_regularizer = tf.keras.regularizers.l1(0.001)
        regularization_penalty = l1_regularizer(self.model.scaling_factor)
        
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
        class_weights = None
        sample_weights = None
        if self.weights is not None:
            class_weights = tf.constant([self.weights[str(i)] for i in range(len(self.weights))], dtype=tf.float32)
            sample_weights = tf.gather(class_weights, y_true)
        
        # 计算Focal Loss的modulator
        softmax_probs = tf.nn.softmax(y_pred)
        labels_one_hot = tf.one_hot(y_true, depth=(len(self.weights) if self.weights is not None else OUTPUT_DIM))
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
        print(f'Epoch {self.epoch_count + 1} completed, average loss: {average_loss:.6f}, duration: {epoch_duration:.2f} seconds')

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

    def get_a_train_batch(self, step):
        '''从训练数据集中获取一个批次的数据 '''
        min_index = step * self.batch_size
        max_index = min_index + self.batch_size
        if max_index > self.train_length:
            max_index = self.train_length
        return self.train_data[min_index:max_index]

    def get_data_label(self, batch):
        '''划分数据特征 和 最后一列标签'''
        data = np.delete(batch, -1, axis=1)
        label = np.array(batch, dtype=np.int32)[:, -1]
        return data, label

    def init_data(self):
        self.train_data = []
        self.test_data = []
        self.label_status = {}
        filename = TRAIN_FILE
        with open(filename, 'r') as file:
            csv_reader = csv.reader(file)
            # 初始化每个类的数据列表
            label_data = [[] for _ in range(OUTPUT_DIM)]
            for row in csv_reader:
                data = []
                for char in row:
                    if char in('None',''):
                        data.append(0)
                    else:
                        data.append(np.float32(char))
                label = int(data[-1])
                if label < OUTPUT_DIM:
                    label_data[label].append(data)
                if self.label_status.get(str(label), 0) > 0:
                    self.label_status[str(label)] += 1
                else:
                    self.label_status[str(label)] = 1
            self.train_data = sum(label_data, [])
        
        filename = TEST_FILE
        with open(filename, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                data = []
                for char in row:
                    if char in('None',''):
                        data.append(0)
                    else:
                        data.append(np.float32(char))
                self.test_data.append(data)
        
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
        logits = self.model(x_tensor, training=False)[0]
        _, top_k_indices = tf.nn.top_k(logits, k=k)
        return top_k_indices.numpy()

    def predict_top_k_batch(self, features, k=5):
        x_tensor = tf.constant(features, dtype=tf.float32)
        logits = self.model(x_tensor, training=False)
        _, top_k_indices = tf.nn.top_k(logits, k=k)
        return top_k_indices.numpy()

    def _prepare_test_arrays(self, data_array):
        features = data_array[:, :-1]
        labels = data_array[:, -1].astype(np.int32)
        return features, labels

    def test(self):
        num_samples = self.test_features.shape[0]
        top1_preds = []
        top3_hits = 0
        top5_hits = 0
        for start in range(0, num_samples, self.eval_batch_size):
            end = min(start + self.eval_batch_size, num_samples)
            batch_feats = self.test_features[start:end]
            topk = self.predict_top_k_batch(batch_feats, k=5)
            top1_preds.extend(topk[:, 0])
            batch_labels = self.test_labels[start:end]
            top3_hits += np.sum(np.any(topk[:, :3] == batch_labels[:, None], axis=1))
            top5_hits += np.sum(np.any(topk[:, :5] == batch_labels[:, None], axis=1))

        y_true = self.test_labels
        y_pred = np.array(top1_preds, dtype=np.int32)

        top1_correct = np.sum(y_true == y_pred)
        top1_accuracy = top1_correct / num_samples
        top3_accuracy = top3_hits / num_samples
        top5_accuracy = top5_hits / num_samples
        print(f"Top-1 accuracy: {top1_accuracy}")
        print(f"Top-3 accuracy: {top3_accuracy}")
        print(f"Top-5 accuracy: {top5_accuracy}")

        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        print(f"Macro-F1: {macro_f1}")
        print(f"Micro-F1: {micro_f1}")
        del y_true, y_pred
        gc.collect()
        return top1_accuracy

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
        del y_true, y_pred
        gc.collect()
        return macro_f1, micro_f1


class VGGModel2(Model):
    def __init__(self, dim, selected_features=[], seed=25):
        super(VGGModel2, self).__init__()
        self.dim = dim
        self.selected_features = selected_features
        self.seed = seed
        
        # Build VGG architecture
        self.conv1_1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                    name='conv_1',
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.conv1_2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                                    name='conv_2',
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.pool1 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool_1')
        
        self.conv2_1 = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                    name='conv_3',
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.conv2_2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                                    name='conv_4',
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.pool2 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool_2')
        
        self.conv3_1 = layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                                    name='conv_5',
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.conv3_2 = layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                                    name='conv_6',
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.conv3_3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                                    name='conv_7',
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.pool3 = layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='pool_3')
        
        self.conv4_1 = layers.Conv2D(512, (3, 3), activation='relu', padding='same',
                                    name='conv_8',
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(1024, activation='relu', name='fc_1',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.dropout1 = layers.Dropout(0.5)
        self.fc2 = layers.Dense(1024, activation='relu', name='fc_2',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.dropout2 = layers.Dropout(0.5)
        self.fc3 = layers.Dense(OUTPUT_DIM, name='fc_3',
                               kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
    
    def call(self, inputs, training=None):
        scaled_input = tf.reshape(inputs, [tf.shape(inputs)[0], self.dim, 1, 1])
        
        # VGG forward pass
        x = self.conv1_1(scaled_input)
        x = self.conv1_2(x)
        x = self.pool1(x)
        
        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.pool2(x)
        
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.pool3(x)
        
        x = self.conv4_1(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout1(x, training=training)
        x = self.fc2(x)
        x = self.dropout2(x, training=training)
        return self.fc3(x)

class VGG2(): # 第二次训练
    def __init__(self,lossType,dim,selected_features=[],seed=25):  # 需输入维度 即当前特征数
        self.lossType = lossType
        set_deterministic_seed(seed)
        self.gamma = GAMMA
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
        self.total_iterations_per_epoch = max(1, (self.train_length + BATCH_SIZE - 1) // BATCH_SIZE)

        # Create the model
        self.model = VGGModel2(dim=dim, selected_features=selected_features, seed=self.seed)
        
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
        modulator = tf.pow(1.0 - pt, self.gamma)
        focal_loss = modulator * ce
        cb_focal_loss = tf.multiply(focal_loss, sample_weights)
        
        if self.lossType == "ce":
            return tf.reduce_sum(ce)
        elif self.lossType == "cb":
            return tf.reduce_sum(tf.multiply(ce, sample_weights))
        elif self.lossType == "cb_focal_loss":
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
        
        micro_F1, macro_F1 = self.test2()
        self.micro_F1List.append(micro_F1)
        self.macro_F1List.append(macro_F1)
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f'Epoch {self.epoch_count + 1} completed, average loss: {average_loss:.6f}, duration: {epoch_duration:.2f} seconds')

    def get_a_train_batch(self, step):
        '''从训练数据集中获取一个批次的数据 '''
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

        # 初始化标签数据字典
        label_data = {i: [] for i in range(29)}

        # 读取训练数据
        filename = TRAIN_FILE
        with open(filename, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                data = []
                for i, char in enumerate(row):
                    if i in self.top_k_indices or i == len(row) - 1:
                        if char == 'None':
                            data.append(0)
                        else:
                            data.append(np.float32(char))

                label = int(data[-1])
                if label in label_data:
                    label_data[label].append(data)

                if str(label) not in self.label_status:
                    self.label_status[str(label)] = 0
                self.label_status[str(label)] += 1

        self.train_data = [item for sublist in label_data.values() for item in sublist]

        # 读取测试数据
        filename = TEST_FILE
        with open(filename, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                data = []
                for i, char in enumerate(row):
                    if i in self.top_k_indices or i == len(row) - 1:
                        if char == 'None':
                            data.append(0)
                        else:
                            data.append(np.float32(char))
                self.test_data.append(data)

        self.train_length = len(self.train_data)
        self.test_length = len(self.test_data)
        self.train_data = self.normalization(self.train_data)
        self.test_data = self.normalization(self.test_data)
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
        x_tensor = tf.constant(features, dtype=tf.float32)
        logits = self.model(x_tensor, training=False)
        _, top_k_indices = tf.nn.top_k(logits, k=k)
        return top_k_indices.numpy()

    def _prepare_test_arrays(self):
        test_np = np.array(self.test_data)
        features = test_np[:, :-1].astype(np.float32)
        labels = test_np[:, -1].astype(np.int32)
        return features, labels

    def test(self):
        num_samples = self.test_features.shape[0]
        top1_preds = []
        top3_hits = 0
        top5_hits = 0
        for start in range(0, num_samples, self.eval_batch_size):
            end = min(start + self.eval_batch_size, num_samples)
            batch_feats = self.test_features[start:end]
            topk = self.predict_top_k_batch(batch_feats, k=5)
            top1_preds.extend(topk[:, 0])
            batch_labels = self.test_labels[start:end]
            top3_hits += np.sum(np.any(topk[:, :3] == batch_labels[:, None], axis=1))
            top5_hits += np.sum(np.any(topk[:, :5] == batch_labels[:, None], axis=1))

        y_true = self.test_labels
        y_pred = np.array(top1_preds, dtype=np.int32)

        top1_correct = np.sum(y_true == y_pred)
        top1_accuracy = top1_correct / num_samples
        top3_accuracy = top3_hits / num_samples
        top5_accuracy = top5_hits / num_samples
        print(f"Top-1 accuracy: {top1_accuracy}")
        print(f"Top-3 accuracy: {top3_accuracy}")
        print(f"Top-5 accuracy: {top5_accuracy}")

        macro_f1 = f1_score(y_true, y_pred, average='macro')
        micro_f1 = f1_score(y_true, y_pred, average='micro')
        print(f"Macro-F1: {macro_f1}")
        print(f"Micro-F1: {micro_f1}")
        return top1_accuracy

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
        del y_true, y_pred
        gc.collect()
        return macro_f1, micro_f1
