import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
from torchvision.models.segmentation import deeplabv3_resnet101
import random
import numpy as np
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, X, y=None, type='train'):
        self.X = X  # データは[サンプル数, 高さ, 幅, チャネル数]の形式
        self.y = y

        # データ拡張や変換
        if type == 'train':
            self.trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),  # PILからテンソルに変換
                transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif type == 'val':
            self.trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif type == 'unlabeled':
            self.trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        elif type == 'no_trans':
            self.trans = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
        else:
            raise ValueError('type must be "train", "val", or "unlabeled".')

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        image = self.X[idx]
        image = self.trans(image)
        if self.y is not None:
            label = torch.tensor(self.y[idx], dtype=torch.long)
            return image, label
        else:
            return image

class RandomFillBackground:
    def __init__(self, fill_color=(0, 0, 0), p=0.5):
        """
        画像の背景を確率pで指定した色に塗りつぶす

        Args:
            fill_color (tuple): 背景を塗りつぶす色 (R, G, B)
            p (float): 背景を塗りつぶす確率 (0.0 ~ 1.0)
        """
        self.fill_color = fill_color
        self.p = p

    def __call__(self, img):
        """
        背景を塗りつぶして新しい画像を返す

        Args:
            img (PIL.Image or Tensor): 入力画像

        Returns:
            Tensor: 背景を塗りつぶしたTensor形式の画像
        """
        # 事前学習済みモデルのロード
        model = deeplabv3_resnet101(pretrained=True)
        model.eval()
        # 確率pで適用しない場合
        if random.random() > self.p:
            return img

        # TensorをPILに変換（必要な場合のみ）
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)

        # 前処理
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(img).unsqueeze(0)

        # セグメンテーションの予測
        with torch.no_grad():
            output = model(input_tensor)["out"][0]
        output_predictions = output.argmax(0).byte().cpu().numpy()

        # 背景のマスク（背景はラベル=0）
        mask = output_predictions == 0

        # 元の画像をNumPy配列に変換
        img_np = np.array(img.resize((224, 224)))  # 224x224にリサイズ
        img_np[mask] = self.fill_color  # 背景部分を塗りつぶす

        # Tensor形式に戻して返す
        return transforms.ToTensor()(Image.fromarray(img_np))
