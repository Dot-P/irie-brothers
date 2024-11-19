import numpy as np
import torch
import matplotlib.pyplot as plt

def fit(net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history):
    """
    モデルの訓練を行い、訓練データと検証データの損失や精度を記録する関数

    Args:
        net: PyTorchモデル
        optimizer: モデルの重みを更新するための最適化アルゴリズム
        criterion: 損失関数
        num_epochs: 訓練を行うエポック数
        train_loader: 訓練データのデータローダー
        test_loader: 検証データのデータローダー
        device: 使用するデバイス ('cpu' または 'cuda')
        history: 過去の訓練履歴を保持する配列 (初期は空の配列)

    Returns:
        訓練履歴を含む配列 (各エポックごとの損失と精度が記録される)
    """
    base_epochs = len(history)

    for epoch in range(base_epochs, num_epochs + base_epochs):
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0

        # 訓練フェーズ
        net.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = outputs.max(1)
            train_acc += (preds == labels).sum().item()

        # 検証フェーズ
        net.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = outputs.max(1)
                val_acc += (preds == labels).sum().item()

        # 平均損失と精度の計算
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_acc = train_acc / len(train_loader.dataset)
        avg_val_loss = val_loss / len(test_loader.dataset)
        avg_val_acc = val_acc / len(test_loader.dataset)

        # エポック結果表示
        print(f"Epoch {epoch+1}/{num_epochs+base_epochs}: "
              f"Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")

        # 履歴更新
        history = np.vstack((history, [epoch + 1, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc]))

    return history

def evaluate_history(history):
    """
    訓練履歴を基に、訓練と検証の損失や精度を評価し、学習曲線をプロットする関数

    Args:
        history: 訓練履歴を記録した配列
                 各行が [エポック番号, 訓練損失, 訓練精度, 検証損失, 検証精度] の形式

    Outputs:
        訓練と検証の損失と精度を学習曲線としてプロット
    """
    #損失と精度の確認
    print(f'[初期状態] loss: {history[0,3]:.5f}, accuracy: {history[0,4]:.5f}')
    print(f'[最終状態] loss: {history[-1,3]:.5f}, accuracy: {history[-1,4]:.5f}' )

    num_epochs = len(history)
    unit = num_epochs / 10

    # 学習曲線の表示 (損失)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,1], 'b', label='training')
    plt.plot(history[:,0], history[:,3], 'k', label='validation')
    plt.xticks(np.arange(0,num_epochs+1, unit))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning curve (loss)')
    plt.legend()
    plt.show()

    # 学習曲線の表示 (精度)
    plt.figure(figsize=(9,8))
    plt.plot(history[:,0], history[:,2], 'b', label='training')
    plt.plot(history[:,0], history[:,4], 'k', label='validation')
    plt.xticks(np.arange(0,num_epochs+1,unit))
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Learning curve (accuracy)')
    plt.legend()
    plt.show()

def torch_seed(seed=123):
    """
    PyTorchの乱数シードを固定し、モデルの学習再現性を確保する関数

    Args:
        seed: 固定するシード値 (デフォルト: 123)

    Effects:
        乱数シードを固定することで、モデルの学習結果が同じ環境で再現可能となる
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
