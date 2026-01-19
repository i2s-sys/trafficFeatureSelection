"""
对比不同特征选择方法在两个开放场景下的表现（PR/PT/ROC）。

场景说明：
- 场景1（open set 20->29）：known_classes=20，使用 OWtest_data1.csv，对应权重 *_OW1_*.weights.h5
- 场景2（open set 25->29）：known_classes=25，使用 OWtest_data2.csv，对应权重 *_OW2_*.weights.h5

特征选择方法与权重：
- factor1: checkpoints/infOW1_resnet2_k32_20260113140827.weights.h5  (20类，OWtest_data1)
- factor2: checkpoints/factorOW2_resnet2_k32_20260113140851.weights.h5 (25类，OWtest_data2)
- pso1   : checkpoints/PSOOW1_resnet2_k32_20260113133425.weights.h5   (20类，OWtest_data1)
- pso2   : checkpoints/PSOOW1_resnet2_k32_20260113133445.weights.h5   (25类，OWtest_data2)
- fpa1   : checkpoints/fpa_resnet2_k32_20260113133007.weights.h5      (20类，OWtest_data1)
- fpa2   : checkpoints/fpaOW2_resnet2_k32_20260113133155.weights.h5   (25类，OWtest_data2)
- inf1   : checkpoints/infOW1_resnet2_k32_20260113133623.weights.h5   (20类，OWtest_data1)
- inf2   : checkpoints/infOW2_resnet2_k32_20260113133550.weights.h5   (25类，OWtest_data2)

输出：
- 在 evaluateRes/compare 目录下分别生成场景1/场景2的 PR/PT/ROC 对比图（多曲线）。
"""

import os
import glob
import argparse
import numpy as np
import tensorflow as tf
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve, auc  # noqa: E402

from pcapResnetPacketSeed import BasicBlock
from tensorflow.keras import layers, Sequential, Model


# ----------------- 特征下标定义 -----------------
indices_factor = [10, 56, 15, 19, 20, 69, 11, 64, 39, 4, 55, 70, 14, 38, 35, 44, 17, 71, 47, 8, 54, 43, 18, 62, 31, 48, 50, 24, 51, 27, 46, 52]
indices_pso = [71, 69, 68, 65, 61, 59, 58, 57, 55, 53, 50, 48, 43, 37, 34, 30, 25, 24, 23, 21, 20, 19, 17, 16, 15, 14, 13, 9, 8, 5, 4, 0]
indices_sca = [14, 70, 3, 12, 62, 55, 23, 25, 61, 20, 51, 56, 24, 18, 15, 21, 48, 13, 17, 9, 59, 26, 32, 68, 5, 67, 66, 71, 8, 7, 69, 65]  # 未用到，但保留
indices_fpa = [60, 34, 13, 62, 11, 10, 24, 70, 61, 12, 30, 27, 28, 14, 15, 68, 26, 2, 52, 65, 22, 7, 18, 45, 67, 53, 4, 35, 20, 55, 21, 19]
indices_inf = [50, 23, 18, 13, 45, 40, 35, 26, 16, 21, 31, 12, 19, 14, 20, 24, 15, 27, 17, 22, 25, 10, 9, 68, 66, 67, 38, 43, 41, 36, 42, 47]


# ----------------- 模型定义（可变输出维度） -----------------
class ResNetModelVar(Model):
    def __init__(self, dim, output_dim, selected_features=None, seed=25):
        super().__init__()
        self.dim = dim
        self.selected_features = selected_features or []
        self.seed = seed

        self.conv_layer = layers.Conv2D(64, kernel_size=(3, 3), strides=1, padding="same",
                                        kernel_initializer=tf.keras.initializers.glorot_uniform(seed=self.seed))
        self.stm = Sequential([
            self.conv_layer,
            layers.BatchNormalization(),
            layers.Activation("relu"),
            layers.MaxPool2D(pool_size=(2, 2), strides=1, padding="same"),
        ])
        layer_dims = [2, 2, 2, 2]
        self.layer1 = self.build_resblock(64, layer_dims[0])
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)
        self.avgpool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(output_dim)

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        res_blocks.add(BasicBlock(filter_num, stride, seed=self.seed))
        for _ in range(1, blocks):
            res_blocks.add(BasicBlock(filter_num, stride=1, seed=self.seed))
        return res_blocks

    def call(self, inputs, training=None):
        x = tf.reshape(inputs, [tf.shape(inputs)[0], self.dim, 1, 1])
        x = self.stm(x, training=training)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        return self.fc(x)


# ----------------- 数据与推理 -----------------
def load_csv(path, feature_indices):
    data, labels = [], []
    with open(path, "r", encoding="utf-8") as f:
        for row in f:
            row = row.strip().split(",")
            if not row or len(row) < 2:
                continue
            values = [
                0.0 if row[i] in ("None", "") else np.float32(row[i])
                for i in range(len(row))
                if i in feature_indices or i == len(row) - 1
            ]
            labels.append(int(values[-1]))
            data.append(values[:-1])
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return data, labels


def minmax_normalize(data):
    mmax = np.max(data, axis=0)
    mmin = np.min(data, axis=0)
    mmax = np.where(mmax == mmin, mmax + 1e-6, mmax)
    return (data - mmin) / (mmax - mmin), (mmin, mmax)


def predict_scores(model, data, batch_size=512):
    scores, preds = [], []
    total = data.shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = tf.constant(data[start:end], dtype=tf.float32)
        logits = model(batch, training=False).numpy()
        prob = tf.nn.softmax(logits, axis=1).numpy()
        scores.append(prob.max(axis=1))
        preds.append(prob.argmax(axis=1))
    return np.concatenate(scores), np.concatenate(preds)


# ----------------- 绘图 -----------------
def plot_curves(no_def_labels, scores_dict, save_dir, prefix):
    """scores_dict: {method: scores_array}"""
    os.makedirs(save_dir, exist_ok=True)

    # PR
    plt.figure(figsize=(8, 4.5))
    for name, scores in scores_dict.items():
        precision, recall, _ = precision_recall_curve(no_def_labels, scores)
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, linewidth=2, label=f"{name} (PR-AUC={pr_auc:.4f})")
    plt.grid()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="lower left")
    plt.title(f"{prefix} PR")
    plt.savefig(os.path.join(save_dir, f"{prefix}_PR.png"), dpi=400, bbox_inches="tight", pad_inches=0)
    plt.close()

    # PT
    plt.figure(figsize=(8, 4.5))
    for name, scores in scores_dict.items():
        precision, _, thresholds = precision_recall_curve(no_def_labels, scores)
        precision = precision[:-1]
        plt.plot(thresholds, precision, linewidth=2, label=name)
    plt.grid()
    plt.xlabel("Threshold")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="lower left")
    plt.title(f"{prefix} PT")
    plt.savefig(os.path.join(save_dir, f"{prefix}_PT.png"), dpi=400, bbox_inches="tight", pad_inches=0)
    plt.close()

    # ROC
    plt.figure(figsize=(8, 4.5))
    plt.plot([0, 1], [0, 1], "k--")
    for name, scores in scores_dict.items():
        fpr, tpr, _ = roc_curve(no_def_labels, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={roc_auc:.4f})")
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.title(f"{prefix} ROC")
    plt.savefig(os.path.join(save_dir, f"{prefix}_ROC.png"), dpi=400, bbox_inches="tight", pad_inches=0)
    plt.close()


def build_and_predict(weight_path, feature_indices, known_classes, data_path, batch_size=512):
    data, labels = load_csv(data_path, feature_indices)
    data, _ = minmax_normalize(data)
    # 按场景指定输出维度，避免尝试多维度带来的不确定性
    output_dim = known_classes
    model = ResNetModelVar(dim=len(feature_indices), output_dim=output_dim,
                           selected_features=feature_indices, seed=25)
    _ = model(tf.zeros((1, len(feature_indices)), dtype=tf.float32), training=False)
    model.load_weights(weight_path)
    print(f"[info] load_weights success: {weight_path}, output_dim={output_dim}")

    scores, preds = predict_scores(model, data, batch_size=batch_size)
    no_def_label = (labels < known_classes).astype(np.int32)
    acc = (preds == labels).mean()
    print(f"{weight_path} | data={os.path.basename(data_path)} | known_classes={known_classes} | closed-set acc={acc:.4f}")
    return no_def_label, scores


def main():
    parser = argparse.ArgumentParser(description="Compare open-world models across feature selectors.")
    parser.add_argument("--output", type=str, default="evaluateRes/compare", help="输出目录")
    parser.add_argument("--batch_size", type=int, default=512, help="推理 batch 大小")
    parser.add_argument("--data1", type=str, default="../../test_data.csv", help="场景1 测试集 (20->29)")
    parser.add_argument("--data2", type=str, default="../../test_data.csv", help="场景2 测试集 (25->29)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    methods = [
        # name, weights, indices, known_classes, data_path
        # ("factor1", "checkpoints/ow_resnet2_k32_20260114033635.weights.h5", indices_factor, 20, args.data1),
        ("factor2", "checkpoints/factorOW2_resnet2_k32_20260113140851.weights.h5", indices_factor, 25, args.data2),
        # ("pso1", "checkpoints/ow_resnet2_k32_20260114034001.weights.h5", indices_pso, 20, args.data1),
        ("pso2", "checkpoints/PSOOW2_resnet2_k32_20260113133445.weights.h5", indices_pso, 25, args.data2),
        # ("fpa1", "checkpoints/ow_resnet2_k32_20260114033852.weights.h5", indices_fpa, 20, args.data1),
        ("fpa2", "checkpoints/fpaOW2_resnet2_k32_20260113133155.weights.h5", indices_fpa, 25, args.data2),
        # ("inf1", "checkpoints/infOW1_resnet2_k32_20260113133623.weights.h5", indices_inf, 20, args.data1),
        ("inf2", "checkpoints/infOW2_resnet2_k32_20260113133550.weights.h5", indices_inf, 25, args.data2),
    ]

    # 场景1: known_classes=20
    scenario1_scores = {}
    scenario1_labels = None
    for name, w, idxs, kc, dpath in methods:
        if kc != 20:
            continue
        if not os.path.exists(w):
            print(f"[WARN] weights not found, skip {name}: {w}")
            continue
        no_def_label, scores = build_and_predict(w, idxs, kc, dpath, batch_size=args.batch_size)
        scenario1_scores[name] = scores
        scenario1_labels = no_def_label  # 同一数据集，label 相同
    if scenario1_scores:
        plot_curves(scenario1_labels, scenario1_scores, args.output, prefix="scenario1_open20")

    # 场景2: known_classes=25
    scenario2_scores = {}
    scenario2_labels = None
    for name, w, idxs, kc, dpath in methods:
        if kc != 25:
            continue
        if not os.path.exists(w):
            print(f"[WARN] weights not found, skip {name}: {w}")
            continue
        no_def_label, scores = build_and_predict(w, idxs, kc, dpath, batch_size=args.batch_size)
        scenario2_scores[name] = scores
        scenario2_labels = no_def_label
    if scenario2_scores:
        plot_curves(scenario2_labels, scenario2_scores, args.output, prefix="scenario2_open25")


if __name__ == "__main__":
    main()
