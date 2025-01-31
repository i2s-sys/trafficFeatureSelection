import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd
from email.policy import default

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from preprocessor import Preprocessor
from appscanner import AppScanner

TRAINFILE = '../train_data2.csv'
TESTFILE = '../test_data2.csv'

def extract_labels(files):
    result = list()
    for file in files:
        result.append(os.path.split(os.path.dirname(file))[-1])
    return result

def test2(y_true,y_pred):
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    print(f"Macro-F1: {macro_f1}")
    print(f"Micro-F1: {micro_f1}")
    return macro_f1, micro_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AppScanner.')
    parser.add_argument('--files',
                        nargs='+', help='pcap files to run through '
                                                   'AppScanner. We use the '
                                                   'directory of each file as '
                                                   'label')
    parser.add_argument('--save',help='Save preprocessed data to given '
                                       'file.')
    parser.add_argument('--load', default='C_feature50_continue.csv', help='load preprocessed data from given '
                                       'file.')
    parser.add_argument('--test', type=float, default=0.33, help='Portion of '
                                                   'data to be used for '
                                                   'testing. All other data is '
                                                   'used for training. '
                                                   '(default=0.33)')
    parser.add_argument('--threshold', type=float, default=0.9, help=
                                                   'Certainty threshold '
                                                   'used in AppScanner '
                                                   '(default=0.9)')
    args = parser.parse_args()
    if len(sys.argv)==1:
        parser.print_help(sys.stderr)
    preprocessor = Preprocessor(verbose=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train_data = []
    test_data = []
    label_status = {}
    filename = TRAINFILE
    csv_reader = csv.reader(open(filename))
    label_data = {i: [] for i in range(29)}
    for row in csv_reader:
        data = [0 if char == 'None' else np.float32(char) for char in row]
        label = int(data[-1])
        label_data[label].append(data)
        if str(label) in label_status:
            label_status[str(label)] += 1
        else:
            label_status[str(label)] = 1
    train_data = sum(label_data.values(), [])
    filename = TESTFILE
    csv_reader = csv.reader(open(filename))
    for row in csv_reader:
        data = [0 if char == 'None' else np.float32(char) for char in row]
        test_data.append(data)
    train_length = len(train_data)
    test_length = len(test_data)
      
    print('init data completed!')

    def load_data(datas):
        data = np.delete(datas, -1, axis=1)
        label = np.array(datas, dtype=np.int32)[:, -1]
        return data, label

    # 加载数据
    X_train, y_train = load_data(train_data)
    X_test, y_test = load_data(test_data)

    # Create appscanner object
    appscanner = AppScanner(args.threshold)
    # Fit AppScanner 先用 train的x和y进行训练
    appscanner.fit(X_train, y_train)
    # Predict AppScanner  然后使用test的x进行预测
    y_pred = appscanner.predict(X_test)

    test2(y_test, y_pred)
    print(classification_report(y_test, y_pred))
