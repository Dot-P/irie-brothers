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

