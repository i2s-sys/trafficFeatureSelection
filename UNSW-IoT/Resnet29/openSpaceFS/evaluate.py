"""
Open-world evaluation script for ResNet2.

功能：
1) 加载 secondTrain.py 训练好的权重（默认取 checkpoints 目录下最新的 *.weights.h5）。
2) 读取 CSV 数据（与 firstResnetPacketSeed.py 的 init_data 相同格式：特征列 + 最后一列为整型标签）。
3) 仅保留与训练一致的前 K 个特征，按 min-max 归一化。
4) 计算每条样本的最大 softmax 概率作为“已知类别”置信度，标签二值化：label < close_world_class_num 记为 1，否则 0。
5) 生成 PR、PT（Precision-Threshold）、ROC 曲线，保存到当前目录下 evaluateRes 文件夹。
"""
import argparse
import glob
import os
import time
import numpy as np
import tensorflow as tf
import matplotlib

# 非交互式后端，便于无界面环境保存图片
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve, auc  # noqa: E402

from pcapResnetPacketSeed import ResNetModel2, OUTPUT_DIM

# 复制 secondTrain.py 中的特征选择
K = 32
sorted_indices = [
    10, 56, 15, 19, 20, 69, 11, 64, 39, 4, 55, 70, 14, 38, 35, 44,
    17, 71, 47, 8, 54, 43, 18, 62, 31, 48, 50, 24, 51, 27, 46, 52
]
TOP_K_INDICES = sorted_indices[:K]

DEFAULT_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../test_data.csv")) # 直接用test_data.csv即可 因为是完整数据集
DEFAULT_CKPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "checkpoints"))
DEFAULT_SAVE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "evaluateRes"))
DEFAULT_WEIGHTS = os.path.abspath(os.path.join(DEFAULT_CKPT_DIR, "resnet2_k32_20260112093427.weights.h5"))

def configure_gpu():
    """尽量开启按需分配，避免一次性占满显存。"""
    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU detected: {len(gpus)} device(s), memory growth enabled.")
        else:
            print("No GPU detected, using CPU.")
    except Exception as e:  # noqa: BLE001
        print(f"GPU config failed: {e}")


def find_latest_checkpoint(ckpt_dir: str):
    pattern = os.path.join(ckpt_dir, "resnet2_k*_*.weights.h5")
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def load_csv(path: str, feature_indices):
    data = []
    labels = []
    with open(path, "r", encoding="utf-8") as f:
        for row in f:
            row = row.strip().split(",")
            if not row or len(row) < 2:
                continue
            # 选取特征 + 标签
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


def minmax_normalize(data: np.ndarray):
    mmax = np.max(data, axis=0)
    mmin = np.min(data, axis=0)
    # 防止除零
    mmax = np.where(mmax == mmin, mmax + 1e-6, mmax)
    return (data - mmin) / (mmax - mmin), (mmin, mmax)


def build_model(input_dim: int, ckpt_path: str):
    model = ResNetModel2(dim=input_dim, selected_features=TOP_K_INDICES, seed=25)
    # 先跑一次前向，构建变量，再加载权重
    _ = model(tf.zeros((1, input_dim), dtype=tf.float32), training=False)
    model.load_weights(ckpt_path)
    print(f"Loaded weights from: {ckpt_path}")
    return model


def predict_scores(model, data: np.ndarray, batch_size: int = 256):
    scores = []
    preds = []
    total = data.shape[0]
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = tf.constant(data[start:end], dtype=tf.float32)
        logits = model(batch, training=False).numpy()
        prob = tf.nn.softmax(logits, axis=1).numpy()
        max_prob = prob.max(axis=1)
        pred_cls = prob.argmax(axis=1)
        scores.append(max_prob)
        preds.append(pred_cls)
    scores = np.concatenate(scores)
    preds = np.concatenate(preds)
    return scores, preds


def plot_pr(no_def_label, scores, save_dir, prefix):
    precision, recall, _ = precision_recall_curve(no_def_label, scores)
    pr_auc = auc(recall, precision)
    plt.figure(figsize=(8, 4.5))
    plt.plot(recall, precision, linewidth=3, label=f"PR-AUC={pr_auc:.4f}")
    plt.grid()
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="lower left")
    plt.title("Precision-Recall")
    out_path = os.path.join(save_dir, f"{prefix}_PR.png")
    plt.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved PR curve: {out_path}")


def plot_pt(no_def_label, scores, save_dir, prefix):
    precision, _, thresholds = precision_recall_curve(no_def_label, scores)
    # precision 比 thresholds 多 1 个点，去掉最后一个与 thresholds 对齐
    precision = precision[:-1]
    plt.figure(figsize=(8, 4.5))
    plt.plot(thresholds, precision, linewidth=3, label="Precision")
    plt.grid()
    plt.xlabel("Classification Threshold")
    plt.ylabel("Precision")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Precision-Threshold")
    out_path = os.path.join(save_dir, f"{prefix}_PT.png")
    plt.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved P-T curve: {out_path}")


def plot_roc(no_def_label, scores, save_dir, prefix):
    fpr, tpr, _ = roc_curve(no_def_label, scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 4.5))
    plt.plot([0, 1], [0, 1], "k--")
    plt.plot(fpr, tpr, linewidth=3, label=f"AUC={roc_auc:.4f}")
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="lower right")
    plt.title("ROC")
    out_path = os.path.join(save_dir, f"{prefix}_ROC.png")
    plt.savefig(out_path, dpi=400, bbox_inches="tight", pad_inches=0)
    plt.close()
    print(f"Saved ROC curve: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Open-world evaluation for ResNet2.")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA_PATH, help="CSV 数据路径，最后一列为标签。")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS, help="模型权重路径（.weights.h5）。默认使用 xx.weights.h5；若不存在则回落到 checkpoints 最新权重。")
    parser.add_argument("--output", type=str, default=DEFAULT_SAVE_DIR, help="结果保存目录。")
    parser.add_argument("--known_classes", type=int, default=25, help="封闭世界类别数（label < N 视为已知）。")
    parser.add_argument("--batch_size", type=int, default=256, help="预测 batch 大小。")
    args = parser.parse_args()

    configure_gpu()

    data_path = os.path.abspath(args.data)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    ckpt_path = args.weights
    if not ckpt_path or not os.path.exists(ckpt_path):
        ckpt_path = find_latest_checkpoint(DEFAULT_CKPT_DIR)
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint found in {DEFAULT_CKPT_DIR}")
    ckpt_path = os.path.abspath(ckpt_path)

    os.makedirs(args.output, exist_ok=True)

    print(f"Loading data from: {data_path}")
    data, labels = load_csv(data_path, TOP_K_INDICES)
    data, (mmin, mmax) = minmax_normalize(data)
    print(f"Data shape: {data.shape}, labels shape: {labels.shape}")

    model = build_model(input_dim=data.shape[1], ckpt_path=ckpt_path)

    scores, preds = predict_scores(model, data, batch_size=args.batch_size)

    # 开放集标签：已知类(1) vs 未知类(0)
    no_def_label = (labels < args.known_classes).astype(np.int32)

    # 计算并打印简单指标
    acc = (preds == labels).mean()
    print(f"Closed-set Top1 accuracy: {acc:.4f}")

    # 绘制曲线
    prefix = f"{os.path.splitext(os.path.basename(ckpt_path))[0]}_{os.path.splitext(os.path.basename(data_path))[0]}"
    plot_pr(no_def_label, scores, args.output, prefix)
    plot_pt(no_def_label, scores, args.output, prefix)
    plot_roc(no_def_label, scores, args.output, prefix)


if __name__ == "__main__":
    main()
