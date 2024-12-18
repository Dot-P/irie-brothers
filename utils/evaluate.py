import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt


def get_predictions(net, data_loader, device):
    """
    モデルの予測値とクラス確率を取得する関数

    Args:
        net: 学習済みPyTorchモデル
        data_loader: 評価データ用のDataLoader
        device: 使用するデバイス ('cpu' または 'cuda')

    Returns:
        y_pred: 予測ラベル (NumPy配列)
        y_prob: 各クラスの確率 (NumPy配列)
    """
    net.eval()
    y_pred, y_prob = [], []

    with torch.no_grad():
        for inputs, _ in data_loader:  
            inputs = inputs.to(device)
            outputs = net(inputs)
            probabilities = F.softmax(outputs, dim=1)

            y_prob.append(probabilities.cpu().numpy())
            y_pred.append(outputs.argmax(dim=1).cpu().numpy())

    return np.concatenate(y_pred), np.concatenate(y_prob)


def select_predictions(y_pred1, y_prob1, y_pred2, y_prob2):
    """
    2つの予測結果と信頼度を比較し、最大信頼度が高い方を選択する。
    確率が同じ場合は、先に与えられたものを優先する。

    Args:
        y_pred1 (np.ndarray): 最初の予測結果 (1次元: クラスラベル)
        y_prob1 (np.ndarray): 最初のクラスごとの信頼度 (2次元: サンプル数×クラス数)
        y_pred2 (np.ndarray): 2番目の予測結果 (1次元: クラスラベル)
        y_prob2 (np.ndarray): 2番目のクラスごとの信頼度 (2次元: サンプル数×クラス数)

    Returns:
        tuple: (選択された予測結果ベクトル, 選択された信頼度ベクトル)
    """
    # 各サンプルごとの最大信頼度を計算
    max_prob1 = np.max(y_prob1, axis=1)  # y_prob1の最大信頼度
    max_prob2 = np.max(y_prob2, axis=1)  # y_prob2の最大信頼度

    # 最大信頼度を比較して、大きい方を選択
    mask = max_prob2 > max_prob1  # Trueならy_prob2を選ぶ

    # y_predとy_probをmaskに基づいて選択
    y_pred = np.where(mask, y_pred2, y_pred1)
    y_prob = np.where(mask[:, None], y_prob2, y_prob1)  # クラスごとの信頼度を選択
    return y_pred, y_prob


def evaluate_model(y_val, y_pred):
    """
    検証データの正解ラベルと予測ラベルを用いて混同行列を可視化する関数

    Args:
        y_val: 検証データの正解ラベル (1次元配列)
        y_pred: モデルが予測したラベル (1次元配列)

    Outputs:
        混同行列をプロットして表示
    """
    ConfusionMatrixDisplay.from_predictions(y_val, y_pred, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.show()


def miss_images(X_val, y_val, y_pred, y_prob, root_dir, num_images_per_fig=5):
    """
    間違っている画像を複数の図に分けて表示し、正解ラベル・予測ラベル・y_pred・y_probをキャプションに表示する

    Args:
        X_val (np.array): 検証用画像データ
        y_val (np.array): 正解ラベル
        y_pred (np.array): 予測ラベル
        y_prob (np.array): 予測確率
        root_dir (str): データセットのルートディレクトリ
        num_images_per_fig (int): 1つの図に表示する画像の最大数
    """

    label_map = {
        "golf": 2,
        "baseball": 0,
        "basketball": 1,
        "judo": 3,
        "swimming": 6,
        "rugby": 5,
        "olympic wrestling": 4,
    }
    label_names = {v: k for k, v in label_map.items()} 

    # 間違っているインデックスを取得
    misclassified_indices = np.where(y_val != y_pred)[0]
    print(f"Total misclassified images: {len(misclassified_indices)}")

    # 画像を複数の図に分けて表示
    total_images = len(misclassified_indices)
    for fig_idx in range(0, total_images, num_images_per_fig):
        fig, axes = plt.subplots(1, num_images_per_fig, figsize=(15, 5))
        plt.suptitle(f"Misclassified Images {fig_idx+1} - {min(fig_idx+num_images_per_fig, total_images)}", fontsize=16)

        # 1つの図に表示する画像数
        for i in range(num_images_per_fig):
            if fig_idx + i >= total_images:
                break
            
            idx = misclassified_indices[fig_idx + i]
            img = X_val[idx].reshape(224, 224, 3).astype(np.uint8)
            true_label = y_val[idx]
            pred_label = y_pred[idx]
            pred_prob = y_prob[idx][pred_label]  # 予測ラベルの確率を取得
            
            # 画像を表示
            axes[i].imshow(img)
            axes[i].axis('off')
            axes[i].set_title(f"True: {label_names[true_label]}\nPred: {label_names[pred_label]}\n({pred_prob:.2f})")

        # 不足するサブプロットを非表示にする
        for j in range(i + 1, num_images_per_fig):
            axes[j].axis('off')

        plt.tight_layout()
        plt.show()